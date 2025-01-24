
/* 
author:         jaime cardenas
title:          wf_embedding_kernel.cu
descripiton:    cuda kernel to embed 3d coordinates to target feature space using 
				green's function solution to the helmholtz equation, modeling each
				token as a point source in 3d space, and superposing the effects of 
				the point sources 
*/

#include <cuda_runtime.h>

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
	const float* wavenumbers_ptr, int stride_wavenumbers_K, 

	float* out_ptr, int stride_out_Z, int stride_out_N, int stride_out_D,
	float* cos_sums_ptr, int stride_cos_sums_Z, int stride_cos_sums_N, int stride_cos_sums_K,
	float* sin_sums_ptr, int stride_sin_sums_Z, int stride_sin_sums_N, int stride_sin_sums_K,

	int tot_Z, int tot_N, int d_model
) {
	
	// compute global thread index. only computes the starting offset of the block, as i will partition NI, NJ loads to 
	// specific warps to read from global memory before making use of threadIdxs
	int offs_NI = blockIdx.x * blockDim.x;  // NI index, 4 unique thread ids (not used yet)
	int offs_Z = blockIdx.z * blockDim.z;  // batch index, 1 thread id

	// calculate the unique local id for each thread (will evaluate to theadIdx.y)
	int thread_id = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;
	int num_threads = blockDim.x * blockDim.y * blockDim.z; // number of threads in a block
	
	// compute warp id (within a block), then the lane id (thread id within a warp)
	// int warp_id = thread_id / warpSize; // not used
	int lane_id = thread_id % warpSize;

	// shared memory for these values, warps fetch these from global memory in parallel
	extern __shared__ __align__(4) float shared_mem[];

	// init wavenumbers, stay in sram throughout
	float* wavenumbers = shared_mem;
	
	// init outputs in shared memory
	float* out = wavenumbers + d_model/2;
	float* cos_sums = out + d_model;
	float* sin_sums = cos_sums + d_model/2;
	
	// shared mem array for NI. only working with a single NI, 
	// which is loaded with the other NJ at the same time. 
	// during first iter have first thread put it in shared so
	// other threads can move it to their register
	float* coords_NI = sin_sums + 3;

	// initialize these here to avoid compilation error, will be in registers
	float coords_NI_x;
	float coords_NI_y;
	float coords_NI_z;
	bool is_inf_NI;
	bool mask_NI;

	// first load wavenumbers
	int num_wn = d_model/2;

	// number of iterations needed
	int k_iters = (num_wn + num_threads - 1) / num_threads; // cdiv

	// loop through wavenumbers to load to SRAM, stays there throughout
	for (int k = 0; k < k_iters; ++k){
		wavenumbers[k*num_threads + thread_id] = wavenumbers_ptr[k*num_threads + thread_id];
	}

	// now load the first NJ block, which includes the NI for this block
	// first move the coords pointer to this batch
	const float* coords_base_ptr = coords_ptr + (offs_Z*stride_coords_Z);

	// loop through blocks of NJ
	int NJ_iters = (tot_N + blockDim.x - 1) / blockDim.x;
	for (int j = 0; j < NJ_iters; ++j){

		int offs_NJ = (offs_NI + j*num_threads + thread_id) % tot_N;   // cycles to beginning when reach the end of the sequence
		bool thread_mask = (offs_Z < tot_Z) && ((j*num_threads + thread_id) < tot_N) && (offs_NI < tot_N); 	// j*num_threads+thread_id checks how many elements have 
																										// been processed to mask already computed values

		// each thread has a single NJ value in its register
		float coords_NJ_x = coords_base_ptr[0*stride_coords_S + offs_NJ*thread_mask];
		float coords_NJ_y = coords_base_ptr[1*stride_coords_S + offs_NJ*thread_mask];
		float coords_NJ_z = coords_base_ptr[2*stride_coords_S + offs_NJ*thread_mask];
		bool is_inf_NJ = coords_NJ_x > 1e30; // convenience for later
		bool mask_NJ = thread_mask && (!is_inf_NJ); 
	
		if (j==0) { // in the first iteration, thread0 loaded the NI token, so now distribute that information to the other threads
			
			// have the first thread move this to shared mem
			if (thread_id==0){ // introduces thread divergence, but only in the first iteration, after that it is very quick
				coords_NI[0] = coords_NJ_x;
				coords_NI[1] = coords_NJ_y;
				coords_NI[2] = coords_NJ_z;
			}

			// synchronize the threads before using the shared memory
			__syncthreads();

			// now all threads move it from shared mem to their register		 
			coords_NI_x = coords_NI[0]; // no bank conflicts, it is broadcast
			coords_NI_y = coords_NI[1];
			coords_NI_z = coords_NI[2];
			is_inf_NI = coords_NI_x > 1e30; 
			mask_NI = thread_mask && (!is_inf_NI); 

			// if NI is masked, stop the program. note all threads have the same NI, so the whole block will terminate
			if (!mask_NI) return;

		}

		// the distance computations. sets inf vals to zero (1/inf) to avoid NaNs. already checked if NI is valid, so only need to do for NJ
		float dist_x = coords_NI_x - (coords_NJ_x*(!is_inf_NJ) + (1/coords_NJ_x)*is_inf_NJ);
		float dist_y = coords_NI_y - (coords_NJ_y*(!is_inf_NJ) + (1/coords_NJ_y)*is_inf_NJ);
		float dist_z = coords_NI_z - (coords_NJ_z*(!is_inf_NJ) + (1/coords_NJ_z)*is_inf_NJ);

		// compute the distance and the masks
		float dists_raw = dist_x*dist_x + dist_y*dist_y + dist_z*dist_z;
		bool mask_IJ = mask_NI && mask_NJ && (dists_raw!=0); 
		float dists = sqrtf(dists_raw + (1-mask_IJ)); // fast approximation of sqrt (1-mask avoids div by 0)

		// loop over wavenumbers in shared memory
		for (int k = 0; k < num_wn; ++k) {

			// compute the phase
			// all threads access the same k from shared memory
			float phase = dists * wavenumbers[k];

			// compute sine and cosine
			// multiply by mask to zero out invalid threads
			float cosine = mask_IJ * cosf(phase);  // Fast approximation of cosine
			float sine = mask_IJ * sinf(phase);  // Fast approximation of sine

			// compute real and imaginary parts
			// divide by one for invalid to avoid nans
			// note that cos and sin are 0 for invalid threads already
			float real = cosine / (dists + (1-mask_IJ));
			float imag = sine / (dists + (1-mask_IJ));

			// have each warp sum the contributions of its threads
			float real_superposition = warp_reduce_sum(real);
			float imag_superposition = warp_reduce_sum(imag);
			float cos_superposition = warp_reduce_sum(cosine);
			float sin_superposition = warp_reduce_sum(sine);

			if (lane_id==0){ // first thread in the warp writes to mem

				// save the intermediate output in shared mem
				atomicAdd(&out[2*k], real_superposition);
				atomicAdd(&out[2*k + 1], imag_superposition);

				// save these for bwd
				atomicAdd(&cos_sums[k], cos_superposition);
				atomicAdd(&sin_sums[k], sin_superposition);

			}
		}
	}

	// initialize pointers for output and move them to proper NI
	float* block_out_ptr = out_ptr + (offs_Z*stride_out_Z) + (offs_NI*stride_out_N);
	float* block_cos_sums_ptr = cos_sums_ptr + (offs_Z*stride_cos_sums_Z) + (offs_NI*stride_cos_sums_N);
	float* block_sin_sums_ptr = sin_sums_ptr + (offs_Z*stride_sin_sums_Z) + (offs_NI*stride_sin_sums_N);

	// have threads write back to HBM in parallel and coalesced manner		
	for (int k = 0; k < k_iters; ++k){
		// write to output. do two iters of out so it fits in the same loop as the trig sums. 
		block_out_ptr[num_threads*(2*k) + thread_id] = out[num_threads*(2*k) + thread_id];
		block_out_ptr[num_threads*(2*k+1) + thread_id] = out[num_threads*(2*k+1) + thread_id];
	
		// store for bwd
		block_cos_sums_ptr[num_threads*k + thread_id] = cos_sums[num_threads*k + thread_id];
		block_sin_sums_ptr[num_threads*k + thread_id] = sin_sums[num_threads*k + thread_id];		
	}
}

// Host function to configure and launch the CUDA kernel
void wf_embedding_kernel_forward(
	const float* coords_ptr, int stride_coords_Z, int stride_coords_S, int stride_coords_N,
	const float* wavenumbers_ptr, int stride_wavenumbers_K, 

	float* out_ptr, int stride_out_Z, int stride_out_N, int stride_out_D,
	float* cos_sums_ptr, int stride_cos_sums_Z, int stride_cos_sums_N, int stride_cos_sums_K,
	float* sin_sums_ptr, int stride_sin_sums_Z, int stride_sin_sums_N, int stride_sin_sums_K,

	int tot_Z, int tot_N, int d_model, 
	cudaStream_t stream
) {
	// define block and grid dimensions
	dim3 block_size(1, 64, 1); // 
	dim3 grid_size(
		tot_N, // each block operates on a single NI
		1, // NJ is looped through
		tot_Z
	);

	// configure the kernel to allow 164kb sram. to maximize blocks/SM. only works on a100
	cudaFuncSetAttribute(wf_embedding_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 164 * 1000);

	// define shared memory per block 
	int shared_mem = 4*(block_size.x*(3 + 2*d_model/2 + d_model) + d_model/2);   // NI (x dim) stores NIxd_model output, 
																				// and two NIxd_model/2 (one for cos sums and another for sin) 
																				// and requires 3 floating point numbers per NI. 
																				// NJ (y dim) is free, as everything is in registers 
																				// and the wavenumbers requires another d_model/2 floating point numbers.
																				// multiply by 4 bytes per fp32

	// Launch the kernel
	wf_embedding_kernel<<<grid_size, block_size, shared_mem, stream>>>(
		coords_ptr, stride_coords_Z, stride_coords_N, stride_coords_S,
		wavenumbers_ptr, stride_wavenumbers_K,
		out_ptr,  stride_out_Z, stride_out_N, stride_out_D,
		cos_sums_ptr, stride_cos_sums_Z, stride_cos_sums_N, stride_cos_sums_K,
		sin_sums_ptr, stride_sin_sums_Z, stride_sin_sums_N, stride_sin_sums_K,
		tot_Z, tot_N, d_model
	);
}