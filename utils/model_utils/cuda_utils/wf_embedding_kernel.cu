
/* 
author:         jaime cardenas
title:          wf_embedding_kernel.cu
descripiton:    cuda kernel to embed 3d coordinates to target feature space using 
				green's function solution to the helmholtz equation, modeling each
				token as a point source in 3d space, and superposing the effects of 
				the point sources 
*/

#include <cuda_runtime.h>
#include <cuda_fp16.h> // for half-precision operations
#include <stdio.h>

// define function to sum the values of threads within a warp
// scales O(log_2(32)) = O(5), very fast
__device__ float warp_reduce_sum(float value) {
	unsigned mask = 0xFFFFFFFF; // mask for active threads in the warp

	for (int offset = 16; offset > 0; offset /= 2) {
		value += __shfl_down_sync(mask, value, offset); // Perform float operations
	}

	return value; 
}

// device kernel for wavefunction embedding
__global__ void wf_embedding_kernel(
	const float* coords_ptr, int stride_coords_Z, int stride_coords_N, int stride_coords_S,
	const __half* wavenumbers_ptr, int stride_wavenumbers_K, 
	const bool* mask_ptr, int stride_mask_Z, int stride_mask_N,

	__half* out_ptr, int stride_out_Z, int stride_out_N, int stride_out_D,
	__half* cos_sums_ptr, int stride_cos_sums_Z, int stride_cos_sums_N, int stride_cos_sums_K,
	__half* sin_sums_ptr, int stride_sin_sums_Z, int stride_sin_sums_N, int stride_sin_sums_K,

	int tot_Z, int tot_N, int d_model
) {
	
	// compute global thread index. only computes the starting offset of the block, as i will partition NI, NJ loads to 
	// specific warps to read from global memory before making use of threadIdxs
	int offs_NI = blockIdx.y * blockDim.y;  // NI index, 4 unique thread ids (not used yet)
	int offs_Z = blockIdx.z * blockDim.z;  // batch index, 1 thread id

	// calculate the unique local id for each thread
	int thread_id = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;
	int num_threads = blockDim.x * blockDim.y * blockDim.z; // number of threads in a block
	
	// compute warp id (within a block), then the lane id (thread id within a warp)
	int warp_id = thread_id / warpSize;
	int lane_id = thread_id % warpSize;

	// shared memory for these values, warps fetch these from global memory in parallel (static to be able to define multiple)
	__shared__ __align__(4) __half wavenumbers[256]; 

	__shared__ __align__(4) float coords_NI_x[4]; 
	__shared__ __align__(4) float coords_NI_y[4];
	__shared__ __align__(4) float coords_NI_z[4];
	__shared__ __align__(4) bool mask_NI[4];

	__shared__ __align__(4) float coords_NJ_x[32];  
	__shared__ __align__(4) float coords_NJ_y[32];  
	__shared__ __align__(4) float coords_NJ_z[32]; 
	__shared__ __align__(4) unsigned int mask_NJ; // to avoid bank conflicts, all threads load the same 32 bit int, but perform operations to modify/read the position of interest

	// init outputs in shared memory
	__shared__ __align__(4) __half out[2048]; 	// format is not the same as the actual output. actual output is interleaved real and imag,
												// but since this is half type, this would give a bank conflict if stored them like this
												// instead do real1, real_(d/2), imag1, imag_(d/2), real3, real_(d/2+1),...
												// this spaces 
	__shared__ __align__(4) __half trig_sums[2048]; // combine cos sums and sin sums to single array w/ stride 2 to avoid bank conflicts (cos1, sin1, cos2, sin2)

	// first load wavenumbers
	int num_wn = d_model/2;

	// number of iterations needed
	int k_iters = (num_wn + num_threads - 1) / num_threads; // cdiv

	// loop through wavenumbers to load to SRAM, stays there throughout
	for (int k = 0; k < k_iters; ++k){
		wavenumbers[k*num_threads + thread_id] = wavenumbers_ptr[k*num_threads + thread_id];
	}

	// now load the first NJ block, which includes the NI for this block
	// first move the coords and mask pointer to this batch
	const float* coords_base_ptr = coords_ptr + (offs_Z*stride_coords_Z);
	const bool* mask_base_ptr = mask_ptr + (offs_Z*stride_mask_Z);

	// loop through blocks of NJ
	int NJ_iters = (tot_N + blockDim.x - 1) / blockDim.x;
	for (int j = 0; j < NJ_iters; ++j){

		int offs_NJ = (offs_NI + j*warpSize + lane_id) % tot_N;   // cycles to beginning when reach the end of the sequence
		bool thread_mask = (offs_Z < tot_Z) && ((j*warpSize + lane_id) < tot_N) && (offs_NI < tot_N); 	// j*warpSize+laneid checks how many elements have 
																				// been processed to mask already computed values

		// now assign one warp for x, another y, another z, and another mask to read from HBM in coalesced manner and in parallel
		// bank conflcts since dealing w/ half and bools
		// stride of 2 for coords_xy (0 dim is x, 1 dim is y), and stride 2 for z ()
		// invalid threads read the first element
		switch (warp_id) {
			// read x
			case 0: coords_NJ_x[lane_id] = coords_base_ptr[0*stride_coords_S + offs_NJ*thread_mask]; break;
			// read y
			case 1: coords_NJ_y[lane_id] = coords_base_ptr[1*stride_coords_S + offs_NJ*thread_mask]; break;
			// read z
			case 2: coords_NJ_z[lane_id] = coords_base_ptr[2*stride_coords_S + offs_NJ*thread_mask]; break;
			// read mask
			case 3: mask_NJ = __ballot_sync(0xFFFFFFFF, !mask_base_ptr[offs_NJ*thread_mask]); break; // creates 32-bit bit mask based on each threads value w/ in the warp				
		}

		// synchronize the threads before using the shared memory
		__syncthreads();
	
		if (j==0) { // in the first iteration, we loaded the NI tokens to shared mem as the first 4 tokens in NJ
			if (lane_id==0){ // only first thread in the warp writes NI to shared mem 
							// these are ordered, so first 4 (if have four warps) are the NI
				coords_NI_x[warp_id] = coords_NJ_x[warp_id];
				coords_NI_y[warp_id] = coords_NJ_y[warp_id];
				coords_NI_z[warp_id] = coords_NJ_z[warp_id];
				mask_NI[warp_id] = (mask_NJ & (1U << warp_id)) != 0; // converts back to bool

			}
			// and again
			__syncthreads();
		}

		// the distance computations. each warp has a unique NI, shared among its threads, but each thread among the warp has a unique NJ 
		// think it is best to convert these to full, do everything, then convert to half when writing to shared
		float dist_x = coords_NI_x[warp_id] - coords_NJ_x[lane_id];
		float dist_y = coords_NI_y[warp_id] - coords_NJ_y[lane_id];
		float dist_z = coords_NI_z[warp_id] - coords_NJ_z[lane_id];

		// compute the distance and the masks
		float dists_raw = dist_x*dist_x + dist_y*dist_y + dist_z*dist_z;
		bool mask_IJ = (thread_mask) && (mask_NI[warp_id]) && ((mask_NJ & (1U << lane_id)) != 0) && (dists_raw!=0); 
		float dists = sqrtf(dists_raw + (1-mask_IJ)); // fast approximation of sqrt (1-mask avoids div by 0)

		// loop over wavenumbers in shared memory
		for (int k = 0; k < num_wn; ++k) {

			// all threads access the same k from shared memory
			float wavenumber = __half2float(wavenumbers[k]);

			// compute the phase
			float phase = dists * wavenumber;

			// compute sine and cosine
			// multiply by mask to zero out invalid threads
			float cos = mask_IJ * cosf(phase);  // Fast approximation of cosine
			float sin = mask_IJ * sinf(phase);  // Fast approximation of sine

			// compute real and imaginary parts
			// divide by one for invalid to avoid nans
			// note that cos and sin are 0 for invalid threads already
			float real = cos / (dists + (1-mask_IJ));
			float imag = sin / (dists + (1-mask_IJ));

			// have each warp sum the contributions of its threads
			float real_superposition = warp_reduce_sum(real);
			float imag_superposition = warp_reduce_sum(imag);
			float cos_sums = warp_reduce_sum(cos);
			float sin_sums = warp_reduce_sum(sin);

			if (lane_id==0){ // first thread in the warp writes to mem

				// convert back to half precision and save in shared
				// no bank conflicts, since only the first thread in the warp is writing

				// save the intermediate output
				// writing as real1, real_d/2, imag1, imag_d/2, real2, real_d/2+1, ...
				// for efficient shared mem access w/ no bank conflicts in shared memory
				// 4*k & (d_model - 1) is equivilant to 4*k % d_model but faster
				int base_idx = warp_id*d_model + ((4 * k) & (d_model - 1));
				out[base_idx + ((4*k) >= d_model)] = __hadd(out[base_idx + ((4*k) >= d_model)], __float2half(real_superposition));
				out[base_idx + 2 + ((4*k) >= d_model)] = __hadd(out[base_idx + 2 + ((4*k) >= d_model)],__float2half(imag_superposition));

				// save these for bwd
				trig_sums[warp_id*d_model + (2*k)] = __hadd(trig_sums[warp_id*d_model + (2*k)], __float2half(cos_sums));
				trig_sums[warp_id*d_model + (2*k + 1)] = __hadd(trig_sums[warp_id*d_model + (2*k + 1)], __float2half(sin_sums));


			}


		}
	}

	// initialize pointers for output and move them to proper NI
	__half* block_out_ptr = out_ptr + (offs_Z*stride_out_Z) + ((offs_NI+warp_id)*stride_out_N);
	__half* block_cos_sums_ptr = cos_sums_ptr + (offs_Z*stride_cos_sums_Z) + ((offs_NI+warp_id)*stride_cos_sums_N);
	__half* block_sin_sums_ptr = sin_sums_ptr + (offs_Z*stride_sin_sums_Z) + ((offs_NI+warp_id)*stride_sin_sums_N);

	// have threads write back to HBM in parallel
	// two seperate loops, since out and the sums are different sizes in last dim
	if ((offs_NI+warp_id) < tot_N){
	
		int d_iters = (num_wn + warpSize - 1) / warpSize; // cdiv
		
		for (int k = 0; k < d_iters; ++k){
		// write to output
		// note that the format is real1, real_d/2, imag1, imag_d/2, real2, real_d/2+1 ... in shared mem,
		// but in global it is real1, imag1, real2, imag2 ...
		// do two writes per iter, one for the first half and one for the second half
		// the read from sram avoids bank conflicts (note that working w/ half), and the write to hbm is coalesced 
			block_out_ptr[warpSize*k + lane_id] = out[warp_id*d_model + warpSize*k + (2*lane_id)];
			block_out_ptr[(d_model/2) + warpSize*k + lane_id] = out[warp_id*d_model + warpSize*k + (2*lane_id + 1)];
		
			// store for bwd
			block_cos_sums_ptr[warpSize*k + lane_id] = trig_sums[warp_id*d_model + warpSize*k + (2*lane_id)];
			block_sin_sums_ptr[warpSize*k + lane_id] = trig_sums[warp_id*d_model + warpSize*k + (2*lane_id + 1)];
		
		}
	}
}

// Host function to configure and launch the CUDA kernel
void wf_embedding_kernel_forward(
	const float* coords_ptr, int stride_coords_Z, int stride_coords_S, int stride_coords_N,
	const __half* wavenumbers_ptr, int stride_wavenumbers_K, 
	const bool* mask_ptr, int stride_mask_Z, int stride_mask_N,

	__half* out_ptr, int stride_out_Z, int stride_out_N, int stride_out_D,
	__half* cos_sums_ptr, int stride_cos_sums_Z, int stride_cos_sums_N, int stride_cos_sums_K,
	__half* sin_sums_ptr, int stride_sin_sums_Z, int stride_sin_sums_N, int stride_sin_sums_K,

	int tot_Z, int tot_N, int d_model, 
	cudaStream_t stream
) {
	// define block and grid dimensions
	dim3 block_size(32, 4, 1); // 4 threads for NI, 32 NJ, 1 for Z (1 for Z so that all NI can make use of each NJ loaded in the block)
	dim3 grid_size(
		1, // choosing this as x to make it easier to assign warps. will be looping through NJ
		(tot_N + block_size.y - 1) / block_size.y,
		(tot_Z + block_size.z - 1) / block_size.z 
	);

	// configure the kernel to allow 96kb sram. not necessary since i am using static
	// cudaFuncSetAttribute(wf_embedding_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 96 * 1024);

	// define shared memory per block (switched to static but keeping it here in case switch back)
	int shared_mem = 2*(block_size.y*(8 + 2*d_model/2 + d_model) + block_size.x*(8) + d_model/2);   // NI (y dim) stores NIxd_model output, 
																									// and two NIxd_model/2 (one for cos sums and another for sin) 
																									// and requires 8 floating point numbers per NI. NJ (x dim) also requires 8 per,
																									// and the wavenumbers requires another d_model/2 floating point numbers.
																									// using fp 16 so multiple by 2 bytes

	// Launch the kernel
	wf_embedding_kernel<<<grid_size, block_size, shared_mem, stream>>>(
		coords_ptr, stride_coords_Z, stride_coords_N, stride_coords_S,
		wavenumbers_ptr, stride_wavenumbers_K,
		mask_ptr, stride_mask_Z, stride_mask_N,
		out_ptr,  stride_out_Z, stride_out_N, stride_out_D,
		cos_sums_ptr, stride_cos_sums_Z, stride_cos_sums_N, stride_cos_sums_K,
		sin_sums_ptr, stride_sin_sums_Z, stride_sin_sums_N, stride_sin_sums_K,
		tot_Z, tot_N, d_model
	);
}