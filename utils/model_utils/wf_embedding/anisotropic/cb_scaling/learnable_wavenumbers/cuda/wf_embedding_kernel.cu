
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
#include <cuda_fp16.h>

__constant__ unsigned FULL_MASK = 0xFFFFFFFF;

// define function to sum the values of threads within a warp
// this is used to superpose the wavefunctions computed by individual threads in a warp
// scales O(log_2(32)) = O(5), very fast
__device__ float superpose(float value) {

	for (int offset = 16; offset > 0; offset /= 2) {
		value += __shfl_down_sync(FULL_MASK, value, offset);
	}
	return value; 
}

// device kernel for wavefunction embedding
// __launch_bounds__(512, 4)
__global__ void wf_embedding_kernel(
    float* coordsA_ptr, int stride_coordsA_Z, int stride_coordsA_S, int stride_coordsA_N,
    float* coordsB_ptr, int stride_coordsB_Z, int stride_coordsB_S, int stride_coordsB_N,
    const float* cb_magnitudes_ptr, int stride_cb_magnitudes_K,
    const float* wavenumbers_ptr, int stride_wavenumbers_K, 

    float* out_ptr, int stride_out_Z, int stride_out_N, int stride_out_D,
    float* d_cb_ptr, int stride_d_cb_Z, int stride_d_cb_N, int stride_d_cb_D, 
    float* d_k_ptr, int stride_d_k_Z, int stride_d_k_N, int stride_d_k_D,

    int tot_Z, int tot_N, int d_model
) {
	
	// calculate the unique local id for each thread (will evaluate to theadIdx.y)
	int thread_id = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;
	int num_threads = blockDim.x * blockDim.y * blockDim.z; // number of threads in a block

	// compute warp id (within a block), then the lane id (thread id within a warp)
	// int warp_id = thread_id / warpSize; not used
	int lane_id = thread_id % warpSize;

	// compute offsets
	int offs_NI = blockIdx.y * blockDim.y + threadIdx.y;  // NI index, 4 NI per block
	int offs_Z = blockIdx.z * blockDim.z + threadIdx.z;  // batch index, 1 batch per block
	
	// init dynamic shared mem and split into seperate arrays
	// have float, half, and int16 types, aligning to 4 byte boundary
	// assigning float first, then half and int16 gurantees proper alignment
	extern __shared__ __align__(4) float smem[]; 

	// compute number of bytes for each dtype
	int num_wn = d_model/2;
	int smem_elements = 3 + 2*num_wn + d_model*3; // 3 coords, cbscaling and k, out, cbgrad, and kgrad
	
	// initialize smem to all 0
	#pragma unroll 1 // disable loop unrolling so that threads reuse registers, implemented in every loop
	for (int thread_mem_idx = thread_id; thread_mem_idx < smem_elements; thread_mem_idx += num_threads){
		smem[thread_mem_idx] = 0.0f;
	}

	// sync threads when done initializing shared memory
	__syncthreads();

	// first partition the float arrays
	float* coordsA_NI_smem = smem;
	float* wavenumbers = coordsA_NI_smem + 3;
	float* cb_magnitudes = wavenumbers + num_wn;
	float* out = cb_magnitudes + num_wn;
	float* d_cb = out + d_model;
	float* d_k = d_cb + d_model;

	// initialize these here to store in rmem
	float coordsA_NI_x;
	float coordsA_NI_y;
	float coordsA_NI_z;

	// load the wavenumbers and wavenumber specific aa magnitudes from gmem
	// in total it is (tot_AA+1)*num_wn elements. do not split among warps, so that the loop uses the same num_iters params
	// for when writing to gmem at the end. this way threads dont create new registers
	// stays in smem throughout. moved to registers when looping through corresponding k
	#pragma unroll 1
	for (int wn_gmem_idx = thread_id; wn_gmem_idx < num_wn; wn_gmem_idx += num_threads){

		// also load wavenumbers to maximize register efficiency, ie use the same variables for multiple things
		cb_magnitudes[wn_gmem_idx] = cb_magnitudes_ptr[wn_gmem_idx]; 
		wavenumbers[wn_gmem_idx] = wavenumbers_ptr[wn_gmem_idx];
	}

	// now load the first NJ block, which includes the NI for this block
	// first move the coordsA, coordsB, and aa labels pointer to this batch
	coordsA_ptr += offs_Z*stride_coordsA_Z;
	coordsB_ptr += offs_Z*stride_coordsB_Z;

	// loop through blocks of NJ
	int NJ_iters = (tot_N + num_threads - 1) / num_threads;
	#pragma unroll 1
	for (int j = 0; j < NJ_iters; ++j){

		int thread_offset = (j*num_threads) + thread_id;
		int offs_NJ = (offs_NI + thread_offset) % tot_N;	// cycles to beginning when reach the end of the sequence, 
																// since starting at this blocks NI to get NI and NJ in the same iteration
		bool thread_mask = thread_offset < tot_N; 	// thread_offset checks how many elements have 
													// been processed to mask already computed values

		float coordsA_NJ_x = coordsA_ptr[0*stride_coordsA_S + offs_NJ];
		float coordsA_NJ_y = coordsA_ptr[1*stride_coordsA_S + offs_NJ];
		float coordsA_NJ_z = coordsA_ptr[2*stride_coordsA_S + offs_NJ];
		float coordsB_NJ_x = coordsB_ptr[0*stride_coordsB_S + offs_NJ]; 
		float coordsB_NJ_y = coordsB_ptr[1*stride_coordsB_S + offs_NJ]; 
		float coordsB_NJ_z = coordsB_ptr[2*stride_coordsB_S + offs_NJ];

		if (j==0) { // in the first iteration, thread0 loaded the NI token, so now distribute that information to the other threads
			
			if (thread_id == 0){
				coordsA_NI_smem[0] = coordsA_NJ_x;
				coordsA_NI_smem[1] = coordsA_NJ_y;
				coordsA_NI_smem[2] = coordsA_NJ_z;
			} // don't need coordsB for NI, just NJ

			// synchronize the threads before using the shared memory
			__syncthreads();

			// now all threads move it from shared mem to their register		 
			coordsA_NI_x = coordsA_NI_smem[0]; // no bank conflicts, it is broadcast for each warp
			coordsA_NI_y = coordsA_NI_smem[1];
			coordsA_NI_z = coordsA_NI_smem[2];
		}


		// and we finally are done moving stuff to registers, now for the computation...
		// the distance computations. sets masked IJ dists to 0 to ensure mask operations are consistent
		float distsA_x = coordsA_NI_x - coordsA_NJ_x; 
		float distsA_y = coordsA_NI_y - coordsA_NJ_y;
		float distsA_z = coordsA_NI_z - coordsA_NJ_z;

		// compute the distance and the masks
		float distsA_raw = (distsA_x*distsA_x) + (distsA_y*distsA_y) + (distsA_z*distsA_z);
		bool mask_IJ = thread_mask && (coordsA_NI_x!=12345) && (coordsA_NJ_x!=12345) && (distsA_raw!=0);
		float inv_distA = mask_IJ * rsqrtf(distsA_raw + (!mask_IJ));
		float distsA = distsA_raw * inv_distA; // fast approximation of sqrt

		// make these unit vectors, masked vals are zero
		// idk how to do wggma and i am running out of time so this will do lol. wgmma would require more smem anyways (i think?)
		// will scale AdotB_unit in wavenumbers loop, as aa scale is dependant on wavenumber
		float AdotB_unit = inv_distA*((coordsB_NJ_x*distsA_x) + (coordsB_NJ_y*distsA_y) + (coordsB_NJ_z*distsA_z));

		// loop over wavenumbers in shared memory
		#pragma unroll 1
		for (int k = 0; k < num_wn; ++k) {

			// load this wavenumbers scales, first tot_AA threads in a warp get assigned an aa's scale to keep in registers
			// and distribute to the other threads within their warp for their assigned NJ
			float cb_magnitude = cb_magnitudes[k]; // swizzling hurt performance more than bank conflicts

			// use the labels of each NJ to get the corresponding scale factor and scale it
			float AdotB_scaled = AdotB_unit * cb_magnitude;

			// modulate the dists and compute the phase
			float wavenumber = wavenumbers[k];
			float phase = (distsA - AdotB_scaled) * wavenumber;

			// compute sine and cosine
			// multiply by mask to zero out invalid threads
			float sine, cosine;
			__sincosf(phase, &sine, &cosine); // compute trig ops in one call

			// compute real and imaginary parts
			float real = cosine * inv_distA; // inv distA is 0 for invalid pairs
			float imag = sine * inv_distA;
			
			// superpose the sources, casts to half to match smem array
			float real_superposition = superpose(real);
			float imag_superposition = superpose(imag);

			float real_d_k = -imag*(distsA - AdotB_scaled);
			float imag_d_k = real*(distsA - AdotB_scaled);

			float real_d_k_superposition = superpose(real_d_k);
			float imag_d_k_superposition = superpose(imag_d_k);

			float real_d_cb = imag * wavenumber * AdotB_unit; // no neg, as derivative of cos is -sin so it cancels
			float imag_d_cb = real * wavenumber * (-AdotB_unit);

			// superpose, but just the contribution of each AA individually, first tot_AA threads in a warp have a val corresponding to an AA
			float real_d_cb_superposition = superpose(real_d_cb);
			float imag_d_cb_superposition = superpose(imag_d_cb);

			if (lane_id==0){ // first thread in the warp writes the outputs to smem
				atomicAdd(&out[2*k], real_superposition);
				atomicAdd(&out[2*k + 1], imag_superposition);
				atomicAdd(&d_k[2*k], real_d_k_superposition);
				atomicAdd(&d_k[2*k + 1], imag_d_k_superposition);
				atomicAdd(&d_cb[2*k], real_d_cb_superposition);
				atomicAdd(&d_cb[2*k + 1], imag_d_cb_superposition);
			}

		}
	}

	// initialize pointers for output and move them to appropriate NI
	out_ptr += (offs_Z*stride_out_Z) + (offs_NI*stride_out_N);
	d_cb_ptr += (offs_Z*stride_d_cb_Z) + (offs_NI*stride_d_cb_N);
	d_k_ptr += (offs_Z*stride_d_k_Z) + (offs_NI*stride_d_k_N);

	// sync threads when writing output from shared mem
	__syncthreads();

	// have threads write back to HBM in parallel and coalesced manner
	#pragma unroll 1
	for (int o = thread_id; o < d_model; o += num_threads){
		out_ptr[o] = out[o];
		d_cb_ptr[o] = d_cb[o];
		d_k_ptr[o] = d_k[o];
	}
}

// Host function to configure and launch the CUDA kernel
void wf_embedding_kernel_forward(
    float* coordsA_ptr, int stride_coordsA_Z, int stride_coordsA_S, int stride_coordsA_N,
    float* coordsB_ptr, int stride_coordsB_Z, int stride_coordsB_S, int stride_coordsB_N,
    const float* cb_magnitudes_ptr, int stride_cb_magnitudes_K, 
    const float* wavenumbers_ptr, int stride_wavenumbers_K, 

    float* out_ptr, int stride_out_Z, int stride_out_N, int stride_out_D,
    float* d_cb_ptr, int stride_d_cb_Z, int stride_d_cb_N, int stride_d_cb_D, 
    float* d_k_ptr, int stride_d_k_Z, int stride_d_k_N, int stride_d_k_D,

    int tot_Z, int tot_N, int d_model,
	cudaStream_t stream
) {
	// define block and grid dimensions
	dim3 block_size(512, 1, 1); // NJ x NI x Z. NJ is looped through (256 at a time), NI is assigned per 256//32=8 warps, and Z is assigned per block. 
								// total of 32 warps perblock, smem limits (on h100) is also 2 blocks, so get 64 warps per SM w/ this config
								// inefficient for small sequences, but may be even more efficient than orig isotropic version for large sequences
	dim3 grid_size(
		1, // NJ is looped through
		tot_N, // each block operates on 4 NI
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
	cudaFuncSetCacheConfig(wf_embedding_kernel, cudaFuncCachePreferShared); // prefer smem over L1 cache

	// define shared memory per block 
	int NI_mem = block_size.y * sizeof(float)* (3 + 3*d_model); // 3 coordsA, dmodel size output, and grads wrt cb and k
	int CB_mem = sizeof(float) * d_model/2; //  tot_AAs per wavenumber
	int WN_mem = sizeof(float) * d_model/2; // d_model//2 + d_model = d_model//2 * 3 
	int shared_mem = NI_mem + CB_mem + WN_mem; // 2 for aa since store aa mags and gradients

	// Launch the kernel
	wf_embedding_kernel<<<grid_size, block_size, shared_mem, stream>>>(
		coordsA_ptr, stride_coordsA_Z, stride_coordsA_S, stride_coordsA_N,
		coordsB_ptr, stride_coordsB_Z, stride_coordsB_S, stride_coordsB_N,
		cb_magnitudes_ptr, stride_cb_magnitudes_K, 
		wavenumbers_ptr, stride_wavenumbers_K,
		out_ptr,  stride_out_Z, stride_out_N, stride_out_D,
		d_cb_ptr, stride_d_cb_Z, stride_d_cb_N, stride_d_cb_D,
		d_k_ptr, stride_d_k_Z, stride_d_k_N, stride_d_k_D,
		tot_Z, tot_N, d_model
	);
}