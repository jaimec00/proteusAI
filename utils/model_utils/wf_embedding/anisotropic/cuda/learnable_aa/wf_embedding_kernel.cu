
/* 
author:         jaime cardenas
title:          wf_embedding_kernel.cu
descripiton:    cuda kernel to embed 3d coordinates to target feature space using 
				custom anisotropic version of green's function solution to the 
				helmholtz equation, modeling each token as a point source in 3d space 
				with an orientation vector that is scaled by the source's aa identity,
				and superposing the effects of the point sources
				amino acid scales per wavenumber are learnable
*/

#include <cuda_runtime.h>
#include <cstdint>
#include <cuda_fp16.h>

// define function to sum the values of threads within a warp
// this is used to superpose the wavefunctions computed by individual threads in a warp
// scales O(log_2(32)) = O(5), very fast
__device__ __half superpose(float value_f) {
	unsigned mask = 0xFFFFFFFF; // mask for active threads in the warp (32 1's)

	for (int offset = 16; offset > 0; offset /= 2) {
		value_f += __shfl_down_sync(mask, value_f, offset);
	}

	__half value = __float2half(value_f);

	return value; 
}

__device__ uint32_t hash5(int a, int b, int c, int d, uint32_t seed) {
	// PCG-XSH-RR based hash algorithm used for dropout
    uint32_t h = seed;
    h ^= (static_cast<uint32_t>(a) * 0x85ebca6b) + 0x9e3779b9;
    h ^= (static_cast<uint32_t>(b) * 0xc2b2ae35) + 0x165667b1;
    h ^= (static_cast<uint32_t>(c) * 0x27d4eb2f) + 0xd6e8feb8;
    h ^= (static_cast<uint32_t>(d) * 0x85ebca6b) + 0x1b873593;
    h = (h ^ (h >> 16)) * 0x85ebca6b;
    h ^= (h >> 13);
    h = (h ^ (h >> 16)) * 0xc2b2ae35;
    h ^= (h >> 16);
    return h;
}

__device__ bool dropout(int Z, int I, int J, int K, uint32_t seed, float dropout_prob) {
	// uses light hash algorithm for dropout to avoid cuda built in RNG
	// reproducible, but not necessary, since bwd tensor is precomputed in fwd
    uint32_t h = hash5(Z, I, J, K, seed);
    float normalized = h / float(UINT32_MAX); // Convert to [0,1]
    return normalized >= dropout_prob;
}

__device__ float get_AA_scale(int16_t aa_idx, __half aa_magnitude) {

	// each of the first tot_AA threads is assigned an aa and the corresponding magnitude
	// each thread is assigned an aa index for its cb
	// fetch the scale from corresponding thread (which holds the cbs scale) 
	unsigned mask = 0xFFFFFFFF;
	float scale = __half2float(__shfl_sync(mask, aa_magnitude, aa_idx)); // convert to float since everything else is float, only cast to half when writing superposed output

	return scale;
}

__device__ __half superpose_d_aa(float local_d_aa, int16_t aa_idx, int lane_id) {

	// initialize aa specific accumulators
	float global_d_aa_f = 0.0f;
	unsigned mask = 0xFFFFFFFF;

	// loops through all threads. 
	// there are tot_AA threads that act as accumulators for a specific AA.
	// need to check if the receiving thread (accumulator) matches the sending thread's assigned AA
	#pragma unroll 1
	for (int tid = 0; tid < warpSize; ++tid){
		bool need_aa = lane_id == __shfl_sync(mask, aa_idx, tid); // check if current sender AA idx matches accumulator's assigned aa idx
		global_d_aa_f += need_aa * __shfl_sync(mask, local_d_aa, tid); // if so, get the value, else accumulates 0
	}

	// cast to half to match smem output array dtype
	__half global_d_aa = __float2half(global_d_aa_f);

	return global_d_aa; // 0 for threads w/ lane id > tot_AA
}

// device kernel for wavefunction embedding
__global__ void wf_embedding_kernel(
    const float* coordsA_ptr, int stride_coordsA_Z, int stride_coordsA_S, int stride_coordsA_N,
    const float* coordsB_ptr, int stride_coordsB_Z, int stride_coordsB_S, int stride_coordsB_N,
    const int16_t* aa_labels_ptr, int stride_aa_labels_Z, int stride_aa_labels_N,
    const __half* aa_magnitudes_ptr, int stride_aa_magnitudes_K, int stride_aa_magnitudes_A,
    const float* wavenumbers_ptr, int stride_wavenumbers_K, 

    __half* out_ptr, int stride_out_Z, int stride_out_N, int stride_out_D,
    __half* d_aa_ptr, int stride_d_aa_Z, int stride_d_aa_N, int stride_d_aa_D, int stride_d_aa_A,

    int tot_Z, int tot_N, int d_model, int tot_AA,
	int magnitude_type, float dropout_p, uint32_t rng_seed
) {
	
	// calculate the unique local id for each thread (will evaluate to theadIdx.y)
	int thread_id = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;
	int num_threads = blockDim.x * blockDim.y * blockDim.z; // number of threads in a block

	// compute warp id (within a block), then the lane id (thread id within a warp)
	int warp_id = thread_id / warpSize;
	int lane_id = thread_id % warpSize;
	int num_warps = num_threads / warpSize;

	// compute offsets
	int offs_NI_block = blockIdx.y * blockDim.y; // first NI in the block, useful for computing NJ offset, since NJ are shared
	int offs_NI = blockIdx.y * blockDim.y + threadIdx.y;  // NI index, 4 NI per block
	int offs_Z = blockIdx.z * blockDim.z + threadIdx.z;  // batch index, 1 batch per block
	
	// init dynamic shared mem and split into seperate arrays
	// have float, half, and int16 types, aligning to 4 byte boundary
	// assigning float first, then half and int16 gurantees proper alignment
	extern __shared__ __align__(4) char smem[]; 

	// compute number of elements for each dtype

	// float elements
	int NIA_elements_f = blockDim.y*3;
	int NJA_elements_f = blockDim.x*3;
	int NJB_elements_f = blockDim.x*3;
	int WN_elements_f = d_model/2;

	// half elements
	int aa_elements_h = tot_AA*WN_elements_f;
	int out_elements_h = blockDim.y*d_model;
	int d_aa_elements_h = blockDim.y*(tot_AA*d_model); 

	// int16 elements
	int NJAA_elements_ih = blockDim.x;

	// compute number of bytes for each dtype
	int smem_float_bytes = sizeof(float)*(NIA_elements_f + NJA_elements_f + NJB_elements_f + WN_elements_f);
	int smem_half_bytes = sizeof(__half)*(out_elements_h + d_aa_elements_h + aa_elements_h);
	int smem_int16_bytes = sizeof(int16_t)*(NJAA_elements_ih);

	// compute total bytes
	int smem_bytes = smem_float_bytes + smem_half_bytes + smem_int16_bytes;
	
	// number of iterations
	int smem_iters = (smem_bytes + num_threads - 1) / num_threads;

	// initialize smem to all 0
	#pragma unroll 1 // disable loop unrolling so that threads reuse registers, implemented in every loop
	for (int mem_idx = 0; mem_idx < smem_iters; ++mem_idx){
		int thread_mem_idx = mem_idx*num_threads + thread_id;
		bool mem_idx_valid = thread_mem_idx < smem_bytes;
		if (mem_idx_valid){ 
			smem[thread_mem_idx] = 0;
		}
	}

	// sync threads when done initializing shared memory
	__syncthreads();

	// first partition the float arrays
	float* coordsA_NI_x_smem = reinterpret_cast<float*>(smem);
	float* coordsA_NI_y_smem = coordsA_NI_x_smem + blockDim.y;
	float* coordsA_NI_z_smem = coordsA_NI_y_smem + blockDim.y;

	float* coordsA_NJ_x_smem = coordsA_NI_z_smem + blockDim.y;
	float* coordsA_NJ_y_smem = coordsA_NJ_x_smem + blockDim.x;
	float* coordsA_NJ_z_smem = coordsA_NJ_y_smem + blockDim.x;

	float* coordsB_NJ_x_smem = coordsA_NJ_z_smem + blockDim.x;
	float* coordsB_NJ_y_smem = coordsB_NJ_x_smem + blockDim.x;
	float* coordsB_NJ_z_smem = coordsB_NJ_y_smem + blockDim.x;

	float* wavenumbers = coordsB_NJ_z_smem + blockDim.x;

	// now the half arrays
	__half* aa_magnitudes = reinterpret_cast<__half*>(smem + smem_float_bytes); // swizzle this along the wavenumbers, ie aa0-wn0, aa0-wn1, aa1-wn0, aa1-wn1, ...
	__half* d_aa = aa_magnitudes + aa_elements_h;		// swizzle this the same way as aa_magnitudes, as only the first 20 threads in a warp need to access
														// these, they are guranteed to be in different banks, as long as there are less aas than threads. just make all the A
														// section contiguous for a given NI x D
	__half* out = d_aa + d_aa_elements_h;	// need to swizzle for when writing to gmem, such that real and imag parts are in adjacent banks
											// same pattern as aa_magnitudes, but pairs are of NI, not of wavenumbers

	// now the int16 array
	// swizzle the labels by splitting into first half and second half and swizzling those. 
	// a little more complicated but necessary to avoid bank conflicts
	int16_t* aa_labels_NJ_smem = reinterpret_cast<int16_t*>(smem + smem_float_bytes + smem_half_bytes);
	
	__syncthreads(); // make sure smem is properly partitioned

	// initialize these here to store in rmem
	float coordsA_NI_x;
	float coordsA_NI_y;
	float coordsA_NI_z;
	bool mask_NI;

	// utilities to store in registers
	int NI_smem_idx = threadIdx.y; // this is used for assigning NI to every warp
	int NJ_smem_idx = threadIdx.x;
	int num_NJ_warps = blockDim.x/warpSize;
	int aa_labels_NJ_smem_idx = 2*(NJ_smem_idx % (blockDim.x/2)) + (NJ_smem_idx>=(blockDim.x/2)); // so dont recompute this complicated monstrosity every time
	__half aa_magnitude = __float2half(1.0f); // only first tot_AA get assigned a magnitude to store, init to 1 (no scaling)
	int pair_NI_idx = NI_smem_idx/2; // what NI pair it belongs to (0->(0,1), 1->(2,3), ...) for swizzling patterns

	// load the wavenumbers and wavenumber specific aa magnitudes from gmem, partition it among warps
	// in total it is (tot_AA+1)*num_wn elements
	// wavenumbers does not have a special swizzling pattern, but aa_magnitudes does (see above)
	// split the work evenly, since have 32 warps
	int wn_warps = 1 + ((1.0f / tot_AA) * num_warps); // evaluates to 2
	bool load_wn = warp_id < wn_warps; // warps 0 and 1 load wavenumbers, 2-31 load AAs
	int wn_threads = wn_warps*warpSize;
	int aa_threads = num_threads - wn_threads;

	if (load_wn) {
		// first load wavenumbers, stay in smem throughout
		int k_iters = (WN_elements_f + wn_threads - 1) / wn_threads;
		#pragma unroll 1
		for (int k = 0; k < k_iters; ++k){
			int wavenumber_idx = k*wn_threads + thread_id; // use thread id directly, since using the first 2 warps
			bool wavenumber_mask = wavenumber_idx < WN_elements_f;
			if (wavenumber_mask){
				wavenumbers[wavenumber_idx] = wavenumbers_ptr[wavenumber_idx];
			}
		}

	} else { // load aa magnitudes
		// stays in smem throughout. moved to registers when looping through corresponding k
		int aa_iters = (aa_elements_h + aa_threads - 1) / aa_threads;
		int aa_thread_id = thread_id - wn_threads; // move this so first thread is 0
		#pragma unroll 1
		for (int aa = 0; aa < aa_iters; ++aa){
			// compute gmem idx and mask
			int aa_magnitudes_gmem_idx = aa*aa_threads + aa_thread_id;
			bool aa_mask = aa_magnitudes_gmem_idx < aa_elements_h;

			// compute smem idx w/ special swizzling pattern, see the initialization of aa_magnitudes array for big picture description
			int wn_idx = aa_magnitudes_gmem_idx / tot_AA; // keep track of what wavenumber each thread is on for proper swizzling
			int aa_idx = aa_magnitudes_gmem_idx % tot_AA; // keep track of the aa
			int pair_wn_idx = wn_idx/2; // what wavenumber pair it belongs to (0->(0,1), 1->(2,3), ...)
 
			int aa_magnitudes_smem_idx = (pair_wn_idx*2*tot_AA) + (2*aa_idx) + ((wn_idx%2)!=0); // even wavenumbers are 2*idx, odd wn are 2*idx+1, within their wn pair

			if (aa_mask){
				aa_magnitudes[aa_magnitudes_smem_idx] = aa_magnitudes_ptr[aa_magnitudes_gmem_idx];
			}
		}
	}

	// now load the first NJ block, which includes the NI for this block
	// first move the coordsA, coordsB, and aa labels pointer to this batch
	const float* coordsA_base_ptr = coordsA_ptr + (offs_Z*stride_coordsA_Z);
	const float* coordsB_base_ptr = coordsB_ptr + (offs_Z*stride_coordsB_Z);
	const int16_t* aa_labels_base_ptr = aa_labels_ptr + (offs_Z*stride_aa_labels_Z);

	// loop through blocks of NJ
	int NJ_iters = (tot_N + blockDim.x - 1) / blockDim.x;
	bool block_mask = (offs_Z < tot_Z) && (offs_NI < tot_N);

	#pragma unroll 1
	for (int j = 0; j < NJ_iters; ++j){

		int thread_offset = (j*blockDim.x) + NJ_smem_idx;
		int offs_NJ = (offs_NI_block + thread_offset) % tot_N;	// cycles to beginning when reach the end of the sequence, 
															// since starting at this blocks NI to get NI and NJ in the same iteration
		bool thread_mask = block_mask && (thread_offset < tot_N); 	// thread_offset checks how many elements have 
																	// been processed to mask already computed values


		// partition the gmem fetches among the 32 warps. 
		// offs NJ cycles to lowest offset for every 8 warps, so need 8 warps to fetch these at a time
		// every warp does 2 coalesced gmem fetches, except 24-31, they do one each
		if (warp_id < num_NJ_warps) {
			coordsA_NJ_x_smem[NJ_smem_idx] = coordsA_base_ptr[0*stride_coordsA_S + offs_NJ]; 
			coordsA_NJ_y_smem[NJ_smem_idx] = coordsA_base_ptr[1*stride_coordsA_S + offs_NJ];
		} else if ((warp_id >= num_NJ_warps) && (warp_id < (2*num_NJ_warps))) {
			coordsA_NJ_z_smem[NJ_smem_idx] = coordsA_base_ptr[2*stride_coordsA_S + offs_NJ];
			coordsB_NJ_x_smem[NJ_smem_idx] = coordsB_base_ptr[0*stride_coordsB_S + offs_NJ]; 
		} else if ((warp_id >= (2*num_NJ_warps)) && (warp_id < (3*num_NJ_warps))) {
			coordsB_NJ_y_smem[NJ_smem_idx] = coordsB_base_ptr[1*stride_coordsB_S + offs_NJ]; 
			coordsB_NJ_z_smem[NJ_smem_idx] = coordsB_base_ptr[2*stride_coordsB_S + offs_NJ];
		} else if ((warp_id >= (3*num_NJ_warps)) && (warp_id < (4*num_NJ_warps))) {
			// this is swizzled
			aa_labels_NJ_smem[aa_labels_NJ_smem_idx] = aa_labels_base_ptr[offs_NJ];
		}

		__syncthreads();

		if (j==0) { // in the first iteration, thread0 loaded the NI token, so now distribute that information to the other threads
			
			if ((lane_id == 0) && ((warp_id%num_NJ_warps) == 0)){ // 8 adjacent warps work on same NI, so only need one thread to move the NJs to NI SMEM for every 8 warps
				coordsA_NI_x_smem[NI_smem_idx] = coordsA_NJ_x_smem[NI_smem_idx]; // alpha carbons are even indexes
				coordsA_NI_y_smem[NI_smem_idx] = coordsA_NJ_y_smem[NI_smem_idx];
				coordsA_NI_z_smem[NI_smem_idx] = coordsA_NJ_z_smem[NI_smem_idx];
			} // don't need coordsB for NI, just NJ

			// synchronize the threads before using the shared memory
			__syncthreads();

			// now all threads move it from shared mem to their register		 
			coordsA_NI_x = coordsA_NI_x_smem[NI_smem_idx]; // no bank conflicts, it is broadcast for each warp
			coordsA_NI_y = coordsA_NI_y_smem[NI_smem_idx];
			coordsA_NI_z = coordsA_NI_z_smem[NI_smem_idx];
			mask_NI = thread_mask && (coordsA_NI_x!=12345); // arbitrary constant (12345) to indicate if position is masked, use a range to acount for precision errors in conversion

		}

		// now move NJ vals from smem to rmem and compute masks
		float coordsA_NJ_x = coordsA_NJ_x_smem[NJ_smem_idx];
		float coordsA_NJ_y = coordsA_NJ_y_smem[NJ_smem_idx];
		float coordsA_NJ_z = coordsA_NJ_z_smem[NJ_smem_idx];
		float coordsB_NJ_x = coordsB_NJ_x_smem[NJ_smem_idx];
		float coordsB_NJ_y = coordsB_NJ_y_smem[NJ_smem_idx];
		float coordsB_NJ_z = coordsB_NJ_z_smem[NJ_smem_idx];
		int16_t aa_labels_NJ = aa_labels_NJ_smem[aa_labels_NJ_smem_idx];
		bool mask_IJ_pre = mask_NI && (coordsA_NJ_x!=12345);

		// and we finally are done moving stuff to registers, now for the computation...
		// the distance computations. sets masked IJ dists to 0 to ensure mask operations are consistent
		float distsA_x = mask_IJ_pre * (coordsA_NI_x - coordsA_NJ_x); 
		float distsA_y = mask_IJ_pre * (coordsA_NI_y - coordsA_NJ_y);
		float distsA_z = mask_IJ_pre * (coordsA_NI_z - coordsA_NJ_z);

		// compute the distance and the masks
		float distsA_raw = (distsA_x*distsA_x) + (distsA_y*distsA_y) + (distsA_z*distsA_z);
		bool mask_IJ = mask_IJ_pre && (distsA_raw!=0); // prevent div by 0
		float inv_distA = mask_IJ * rsqrtf(distsA_raw + (!mask_IJ));
		float distsA = distsA_raw * inv_distA; // fast approximation of sqrt

		// make these unit vectors, masked vals are zero
		float distA_x_unit = distsA_x * inv_distA;
		float distA_y_unit = distsA_y * inv_distA;
		float distA_z_unit = distsA_z * inv_distA;

		// idk how to do wggma and i am running out of time so this will do lol. wgmma would require more smem anyways (i think?)
		// will scale AdotB_unit in wavenumbers loop, as aa scale is dependant on wavenumber
		float AdotB_unit = (coordsB_NJ_x*distA_x_unit) + (coordsB_NJ_y*distA_y_unit) + (coordsB_NJ_z*distA_z_unit);
		
		float magnitude;
		switch (magnitude_type) {
			case 0: // no magnitude scaling, each observer sees the full effect of all sources
				magnitude = 1.0;
				break;
			case 1: // same magnitude as green's func, i.e. 1/|R|
				magnitude = inv_distA;
				break;
			case 2:	// take the log2 of the distance, i.e. 1/log2(|R|) to account more for distant interactions
				magnitude = 1.0 / log2f(distsA); // evaluates to 1 / log2(0) = 1/-inf = -0 for 0 dists.
				break;
			case 3: // take the sqrt of dists 1/sqrt(dists). not as aggressive as log2, but still accounts for distant sources
				magnitude = mask_IJ * rsqrtf(distsA + (!mask_IJ));
				break;
		}

		// loop over wavenumbers in shared memory
		#pragma unroll 1
		for (int k = 0; k < WN_elements_f; ++k) {

			// load this wavenumbers scales, first tot_AA threads in a warp get assigned an aa's scale to keep in registers
			// and distribute to the other threads within their warp for their assigned NJ
			if (lane_id < tot_AA) {
				int k_pair = k / 2; 
				aa_magnitude = aa_magnitudes[(2*tot_AA*k_pair) + (2*lane_id) + ((k%2)!=0)]; // swizzling logic, see how aa_magnitudes were loaded above
			}
			__syncwarp(); // make sure the new scales are updated before distributing based on each threads assigned aa label for NJ

			// load wavenumber
			float wavenumber = wavenumbers[k];

			// use the labels of each NJ to get the corresponding scale factor
			float AdotB_scale = get_AA_scale(aa_labels_NJ, aa_magnitude);

			// scale it
			float AdotB_scaled = AdotB_unit * AdotB_scale;

			// modulate the dists and compute the phase
			float phase = (distsA - AdotB_scaled) * wavenumber;

			// compute sine and cosine
			// multiply by mask to zero out invalid threads
			float sine, cosine;
			__sincosf(phase, &sine, &cosine); // compute trig ops in one call

			// compute real and imaginary parts
			float real = mask_IJ * cosine * magnitude;
			float imag = mask_IJ * sine * magnitude;

			if (dropout_p != 0.0) {
				// lightweight hash based dropout, avoids overhead of built in cuda rngs
				bool drop_val = dropout( offs_Z, offs_NI, offs_NJ, k, rng_seed, dropout_p);
				real = drop_val * real / (1-dropout_p);
				imag = drop_val * imag / (1-dropout_p);
			}
			
			// superpose the sources, casts to half to match smem array
			__half real_superposition = superpose(real);
			__half imag_superposition = superpose(imag);

			if (lane_id==0){ // first thread in the warp writes the outputs to smem

				// save the intermediate output in smem
				int real_smem_idx = (pair_NI_idx*2*d_model) + (2*(2*k)) + ((NI_smem_idx%2)!=0);
				int imag_smem_idx = (pair_NI_idx*2*d_model) + (2*(2*k+1)) + ((NI_smem_idx%2)!=0);

				atomicAdd(&out[real_smem_idx], real_superposition);
				atomicAdd(&out[imag_smem_idx], imag_superposition);
			}

			// compute grad wrt AA scale
			// V = A_hat dot B_hat
			// real d_aa = d(cos(k[|A| - V*aa])/|A|) = -kV*-sin(k[|A| - V*aa])/|A| = kV*sin(k[|A| - V*aa])/|A| = kV*imag
			// imag d_aa = d(sin(k[|A| - V*aa])/|A|) = -kV*cos(k[|A| - V*aa])/|A| = -kV*real
			float real_d_aa = imag * wavenumber * AdotB_unit; // no neg, as derivative of cos is -sin so it cancels
			float imag_d_aa = real * wavenumber * (-AdotB_unit);

			// superpose, but just the contribution of each AA individually, first tot_AA threads in a warp have a val corresponding to an AA
			__half real_d_aa_superposition = superpose_d_aa(real_d_aa, aa_labels_NJ, lane_id);
			__half imag_d_aa_superposition = superpose_d_aa(imag_d_aa, aa_labels_NJ, lane_id);

			if (lane_id < tot_AA) { // have the threads assigned as AA accumulators write to corresponding AA
				// the smem array is NI x D x A
				// but swizzled such that all A for an NI x D are in adjacent banks
				int real_d_aa_smem_idx = (pair_NI_idx*2*d_model*tot_AA) + (2*(2*k*tot_AA + lane_id)) + ((NI_smem_idx%2)!=0);
				int imag_d_aa_smem_idx = (pair_NI_idx*2*d_model*tot_AA) + (2*((2*k+1)*tot_AA + lane_id)) + ((NI_smem_idx%2)!=0);

				atomicAdd(&d_aa[real_d_aa_smem_idx], real_d_aa_superposition);
				atomicAdd(&d_aa[imag_d_aa_smem_idx], imag_d_aa_superposition);

			}
		}
	}

	// initialize pointers for output and move them to proper NI
	__half* block_out_ptr = out_ptr + (offs_Z*stride_out_Z) + (offs_NI*stride_out_N);
	__half* block_d_aa_ptr = d_aa_ptr + (offs_Z*stride_d_aa_Z) + (offs_NI*stride_d_aa_N);

	// sync threads when writing output from shared mem
	__syncthreads();

	// have threads write back to HBM in parallel and coalesced manner
	int out_iters = (d_model + blockDim.x - 1) / blockDim.x;
	#pragma unroll 1
	for (int o = 0; o < out_iters; ++o){

		int out_gmem_idx = o*blockDim.x + threadIdx.x;
		int out_smem_idx = (pair_NI_idx*2*d_model) + (2*out_gmem_idx) + ((NI_smem_idx%2)!=0);
		
		bool out_mask = out_gmem_idx < d_model;
		if (out_mask){
			block_out_ptr[out_gmem_idx] = out[out_smem_idx];
		}
	}

	int d_aa_iters = (d_model*tot_AA + blockDim.x - 1) / blockDim.x;
	#pragma unroll 1
	for (int aa = 0; aa < d_aa_iters; ++aa){
		int d_aa_gmem_idx = aa*blockDim.x + threadIdx.x;
		int d_aa_smem_idx = (pair_NI_idx*2*d_model*tot_AA) + (2*d_aa_gmem_idx) + ((NI_smem_idx%2)!=0);

		bool aa_mask = d_aa_gmem_idx < (d_model*tot_AA);
		if (aa_mask){
			block_d_aa_ptr[d_aa_gmem_idx] = d_aa[d_aa_smem_idx];
		}
	}
}

// Host function to configure and launch the CUDA kernel
void wf_embedding_kernel_forward(
    const float* coordsA_ptr, int stride_coordsA_Z, int stride_coordsA_S, int stride_coordsA_N,
    const float* coordsB_ptr, int stride_coordsB_Z, int stride_coordsB_S, int stride_coordsB_N,
    const int16_t* aa_labels_ptr, int stride_aa_labels_Z, int stride_aa_labels_N,
    const __half* aa_magnitudes_ptr, int stride_aa_magnitudes_K, int stride_aa_magnitudes_A,
    const float* wavenumbers_ptr, int stride_wavenumbers_K, 

    __half* out_ptr, int stride_out_Z, int stride_out_N, int stride_out_D,
    __half* d_aa_ptr, int stride_d_aa_Z, int stride_d_aa_N, int stride_d_aa_D, int stride_d_aa_A,

    int tot_Z, int tot_N, int d_model, int tot_AA,
	int magnitude_type, float dropout_p, uint32_t rng_seed,
	cudaStream_t stream
) {
	// define block and grid dimensions
	dim3 block_size(256, 4, 1); // NJ x NI x Z. NJ is looped through (256 at a time), NI is assigned per 256//32=8 warps, and Z is assigned per block. 
								// total of 32 warps perblock, smem limits (on h100) is also 2 blocks, so get 64 warps per SM w/ this config
								// inefficient for small sequences, but may be even more efficient than orig isotropic version for large sequences
	dim3 grid_size(
		1, // NJ is looped through
		(tot_N  + block_size.y - 1) / block_size.y, // each block operates on 4 NI
		tot_Z
	);

	// configure the kernel to allow the maximum sram to maximize blocks/SM. 
	// a100 allows 164 kB, h100 allows 228 kB, should work on other devices too, but much slower since this kernel uses
	// significant SRAM. default for most consumer devices is 48 kB, so the kernel will run but with significantly less blocks/SM,
	// meaning it won't fully utilize the hardware
	int device;	
	cudaGetDevice(&device);
	int maxSharedMemPerBlockOptin = 0;
	cudaDeviceGetAttribute(&maxSharedMemPerBlockOptin, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
	cudaFuncSetAttribute(wf_embedding_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, maxSharedMemPerBlockOptin);

	// define shared memory per block 
	int NI_mem = block_size.y * (sizeof(float)*3 + sizeof(__half)*(d_model + tot_AA*d_model)); // 3 coordsA, dmodel size output, and tot_AAxd_model for grad wrt to AAs
	int NJ_mem = block_size.x*(sizeof(float)*(3 + 3) + sizeof(int16_t)*1); // 3 coordsA, 3 coordsB, and 1 label each
	int AA_mem = sizeof(__half) * (tot_AA*d_model/2); //  tot_AAs per wavenumber
	int WN_mem = sizeof(float) * d_model/2; // 1 for each wavenumber
	int shared_mem = NI_mem + NJ_mem + AA_mem + WN_mem; // 2 for aa since store aa mags and gradients

	// Launch the kernel
	wf_embedding_kernel<<<grid_size, block_size, shared_mem, stream>>>(
		coordsA_ptr, stride_coordsA_Z, stride_coordsA_S, stride_coordsA_N,
		coordsB_ptr, stride_coordsB_Z, stride_coordsB_S, stride_coordsB_N,
		aa_labels_ptr, stride_aa_labels_Z, stride_aa_labels_N,
		aa_magnitudes_ptr, stride_aa_magnitudes_K, stride_aa_magnitudes_A,
		wavenumbers_ptr, stride_wavenumbers_K,
		out_ptr,  stride_out_Z, stride_out_N, stride_out_D,
		d_aa_ptr, stride_d_aa_Z, stride_d_aa_N, stride_d_aa_D, stride_d_aa_A,
		tot_Z, tot_N, d_model, tot_AA,
		magnitude_type, dropout_p, rng_seed
	);
}