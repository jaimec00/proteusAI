
/* 
author:         jaime cardenas
title:          wf_embedding_kernel.cu
descripiton:    cuda kernel to embed 3d coordinates to target feature space using 
				green's function solution to the helmholtz equation, modeling each
				token as a point source in 3d space, and superposing the effects of 
				the point sources 
*/

#include <cuda_runtime.h>
#include <iostream>

// define function to sum the values of threads within a warp
// this is used to superpose the wavefunctions computed by individual threads in a warp
// scales O(log_2(32)) = O(5), very fast
__device__ float warp_sum(float value) {
	unsigned mask = 0xFFFFFFFF; // mask for active threads in the warp (32 1's)

	for (int offset = 16; offset > 0; offset /= 2) {
		value += __shfl_down_sync(mask, value, offset);
	}

	return value; 
}

// device kernel for wavefunction embedding
__global__ void wf_embedding_kernel(
	const float* coords_ptr, int stride_coords_Z, int stride_coords_S, int stride_coords_N,
	const float* wavenumbers_ptr, int stride_wavenumbers_K, 

	float* out_ptr, int stride_out_Z, int stride_out_N, int stride_out_D,
	float* cos_sums_ptr, int stride_cos_sums_Z, int stride_cos_sums_N, int stride_cos_sums_K,
	float* sin_sums_ptr, int stride_sin_sums_Z, int stride_sin_sums_N, int stride_sin_sums_K,

	int tot_Z, int tot_N, int d_model, int magnitude_type
) {
	
	// compute global thread index
	int offs_NI = blockIdx.x * blockDim.x + threadIdx.x;  // NI index, 1 thread id
	int offs_Z = blockIdx.z * blockDim.z + threadIdx.z;  // batch index, 1 thread id

	// calculate the unique local id for each thread (will evaluate to theadIdx.y)
	int thread_id = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;
	int num_threads = blockDim.x * blockDim.y * blockDim.z; // number of threads in a block
	
	// compute warp id (within a block), then the lane id (thread id within a warp)
	// int warp_id = thread_id / warpSize; // not used
	int lane_id = thread_id % warpSize;

	// init dynamic shared mem and split into seperate arrays
	extern __shared__ __align__(4) float shared_mem[];

	// it seems shared memory in some configurations
	// has residual values in it, so explicitly set all values to zero before any computations
	int shared_mem_elements = d_model + d_model/2 + d_model/2 + d_model/2 + 3;
	int shared_mem_iters = (shared_mem_elements + num_threads - 1) / num_threads;
	#pragma unroll 1 // disable loop unrolling so that threads reuse registers, implemented in every loop
	for (int mem_idx = 0; mem_idx < shared_mem_iters; ++mem_idx){
		int thread_mem_idx = mem_idx*num_threads + thread_id;
		bool mem_idx_valid = thread_mem_idx < shared_mem_elements;
		if (mem_idx_valid){ // better to have invalid threads wait than do redundant global mem reads
							// minimal thread divergence, as invalid threads do not take a different branch,
							// they just wait for the valid threads to finish
			shared_mem[thread_mem_idx] = 0.0f;
		}
	}

	// sync threads when done initializing shared memory
	__syncthreads();

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
	float* coords_NI = sin_sums + d_model/2;

	// initialize these here to avoid compilation error, will be in each thread's register
	float coords_NI_x;
	float coords_NI_y;
	float coords_NI_z;
	bool mask_NI;

	// first load wavenumbers
	int num_wn = d_model/2;

	// number of iterations needed
	int k_iters = (num_wn + num_threads - 1) / num_threads;

	// loop through wavenumbers to load to SRAM, stays there throughout
	#pragma unroll 1
	for (int k = 0; k < k_iters; ++k){
		int wavenumber_idx = k*num_threads + thread_id;
		bool wavenumber_mask = wavenumber_idx < num_wn;
		if (wavenumber_mask){
			wavenumbers[wavenumber_idx] = wavenumbers_ptr[wavenumber_idx];
		}
	}

	// now load the first NJ block, which includes the NI for this block
	// first move the coords pointer to this batch
	const float* coords_base_ptr = coords_ptr + (offs_Z*stride_coords_Z);

	// loop through blocks of NJ
	int NJ_iters = (tot_N + blockDim.y - 1) / blockDim.y;
	bool block_mask = (offs_Z < tot_Z) && (offs_NI < tot_N); // since each block gets one token, this will always be true, but just for clarity
	#pragma unroll 1
	for (int j = 0; j < NJ_iters; ++j){

		int thread_offset = (j*num_threads) + thread_id;
		int offs_NJ = (offs_NI + thread_offset) % tot_N;	// cycles to beginning when reach the end of the sequence, 
															// since starting at this blocks NI to get NI and NJ in the same iteration
		bool thread_mask = block_mask && (thread_offset < tot_N); 	// thread_offset checks how many elements have 
																								// been processed to mask already computed values

		// each thread has a single NJ value in its register
		// coords was transposed by pytorch to get 3xN tensor, so all x values are contiguous, same w/ y and z 
		// invalid threads still read from global mem, as this is still one memory transaction per warp since the reads are 
		// coalesced and we are guranteed to be in bounds since we cycle to the beginning (w/ mod tot_N)
		// possible that one iteration will have some threads in a warp at the end of the sequence and the others at beginning,
		// leading to uncoalesced mem access, but guranteed to happen <= 1 time per block, so no noticeable performance loss
		float coords_NJ_x = coords_base_ptr[0*stride_coords_S + offs_NJ]; 
		float coords_NJ_y = coords_base_ptr[1*stride_coords_S + offs_NJ];
		float coords_NJ_z = coords_base_ptr[2*stride_coords_S + offs_NJ];
		bool mask_NJ = thread_mask && (coords_NJ_x!=12345); // baked the mask into coords w/ this val, arbitrary, but inf makes it harder to avoid NaNs

		if (j==0) { // in the first iteration, thread0 loaded the NI token, so now distribute that information to the other threads
			
			// have the first thread move this to shared mem
			if (thread_id==0){ // introduces thread divergence, but only in the first iteration, after that it is very quick
				coords_NI[0] = coords_NJ_x;
				coords_NI[1] = coords_NJ_y;
				coords_NI[2] = coords_NJ_z;
			}

			// synchronize the threads before using the shared memory, only happens in the first iteration
			// this also happens before wavenumbers are read, so two birds w/ one stone
			__syncthreads();

			// now all threads move it from shared mem to their register		 
			coords_NI_x = coords_NI[0]; // no bank conflicts, it is broadcast
			coords_NI_y = coords_NI[1];
			coords_NI_z = coords_NI[2];
			mask_NI = thread_mask && (coords_NI_x!=12345); 

			// if NI is masked, stop the program. note all threads have the same NI, so the whole block will terminate
			if (!mask_NI) return;

		}

		// the distance computations. sets masked NJ dists to 0 to ensure mask operations are consistent
		float dist_x = mask_NJ*(coords_NI_x - coords_NJ_x); 
		float dist_y = mask_NJ*(coords_NI_y - coords_NJ_y);
		float dist_z = mask_NJ*(coords_NI_z - coords_NJ_z);

		// compute the distance and the masks
		float dists_raw = dist_x*dist_x + dist_y*dist_y + dist_z*dist_z;
		bool mask_IJ = mask_NI && mask_NJ && (dists_raw!=0); // prevent div by 0
		float dists = mask_IJ * dists_raw * rsqrtf(dists_raw + (!mask_IJ)); // fast approximation of sqrt ( + !mask avoids div by 0, 
																			// mult by mask ensures invalid dists are 0 for consistent masking)

		float magnitude;
		switch (magnitude_type) {
			case 0: // no magnitude scaling, each observer sees the full effect of all sources
				magnitude = 1.0;
				break;
			case 1: // same magnitude as green's func, i.e. 1/|R|
				magnitude = 1.0 / (dists + (!mask_IJ));
				break;
			case 2:	// take the log2 of the distance, i.e. 1/log2(|R|) to account more for distant interactions
				magnitude = 1.0 / log2f(dists + 2*(!mask_IJ)); // evaluates to 1 / log2(0+2*1) = 1/1 = 1 for 0 dists. cos and sine are already masked
				break;
			case 3: // take the sqrt of dists 1/sqrt(dists). 
				magnitude = mask_IJ * rsqrtf(dists + (!mask_IJ));
				break;
		}

		// loop over wavenumbers in shared memory
		#pragma unroll 1
		for (int k = 0; k < num_wn; ++k) {

			// compute the phase
			// all threads access the same k from shared memory
			float phase = dists * wavenumbers[k];

			// compute sine and cosine
			// multiply by mask to zero out invalid threads
			float cosine = mask_IJ * __cosf(phase);  // Fast approximation of cosine
			float sine = mask_IJ * __sinf(phase);  // Fast approximation of sine

			// compute real and imaginary parts
			float real = cosine*magnitude;
			float imag = sine*magnitude;

			float real_superposition = warp_sum(real);
			float imag_superposition = warp_sum(imag);

			float cos_sum = warp_sum(real*dists); // d_imag
			float sin_sum = warp_sum(imag*dists); // d_real

			if (lane_id==0){ // first thread in the warp writes to mem

				// save the intermediate output in shared mem
				// no bank conflicts, a single thread per warp writes these
				// minimal contention, as only two threads overall are performing atomic adds
				atomicAdd(&out[2*k], real_superposition);
				atomicAdd(&out[2*k + 1], imag_superposition);

				// save these for bwd
				atomicAdd(&cos_sums[k], cos_sum);
				atomicAdd(&sin_sums[k], sin_sum);

			}
		}
	}

	// initialize pointers for output and move them to proper NI
	float* block_out_ptr = out_ptr + (offs_Z*stride_out_Z) + (offs_NI*stride_out_N);
	float* block_cos_sums_ptr = cos_sums_ptr + (offs_Z*stride_cos_sums_Z) + (offs_NI*stride_cos_sums_N);
	float* block_sin_sums_ptr = sin_sums_ptr + (offs_Z*stride_sin_sums_Z) + (offs_NI*stride_sin_sums_N);

	// sync threads when writing output from shared mem
	__syncthreads();

	// have threads write back to HBM in parallel and coalesced manner
	#pragma unroll 1
	for (int k = 0; k < k_iters; ++k){
		// write to output. do two iters of out so it fits in the same loop as the trig sums. 
		// all memory writes are coalesced, and no bank conflicts, since each thread reads adjacent 32bit floats from out
		// in shared mem and writes to adjacent indexes in HBM
		int out1_idx = (num_threads*(2*k)) + thread_id;
		int out2_idx = (num_threads*(2*k + 1)) + thread_id;

		bool out1_mask = out1_idx < (d_model);
		bool out2_mask = out2_idx < (d_model);

		if (out1_mask){
			block_out_ptr[out1_idx] = out[out1_idx];
		}
		if (out2_mask){
			block_out_ptr[out2_idx] = out[out2_idx];
		}

		// store for bwd
		int trig_idx = num_threads*k + thread_id;
		bool trig_mask = trig_idx < num_wn;

		if (trig_mask){
			block_cos_sums_ptr[trig_idx] = cos_sums[trig_idx];
			block_sin_sums_ptr[trig_idx] = sin_sums[trig_idx];	
		}
	}
}

// Host function to configure and launch the CUDA kernel
void wf_embedding_kernel_forward(
	const float* coords_ptr, int stride_coords_Z, int stride_coords_S, int stride_coords_N,
	const float* wavenumbers_ptr, int stride_wavenumbers_K, 

	float* out_ptr, int stride_out_Z, int stride_out_N, int stride_out_D,
	float* cos_sums_ptr, int stride_cos_sums_Z, int stride_cos_sums_N, int stride_cos_sums_K,
	float* sin_sums_ptr, int stride_sin_sums_Z, int stride_sin_sums_N, int stride_sin_sums_K,

	int tot_Z, int tot_N, int d_model, int magnitude_type,
	cudaStream_t stream
) {
	// define block and grid dimensions
	dim3 block_size(1, 64, 1); // 
	dim3 grid_size(
		tot_N, // each block operates on a single NI
		1, // NJ is looped through
		tot_Z
	);

	// configure the kernel to allow the maximum sram to maximize blocks/SM. 
	// a100 allows 164 kB, h100 allows 228 kB, should work on other devices too, but much slower since this kernel uses
	// significant SRAM. default for most consumer devices is 48 kB, so the kernel will run but with significantly less blocks/SM,
	// meaning it won't fully utilize the hardware
	// for d_model=512, each block uses 5.2 kB SRAM (+ 1 kB for cuda driver shared mem). see shared_mem calculation in next code block
	// a100 get 26/32 blocks per SM while h100 gets 35/32=>32/32 blocks per SM. h100 also has more SMs and is generally faster
	// shared memory per block is dependant on d_model, so if this is a limiting factor, you should decrease d_model
	int device;	
	cudaGetDevice(&device);
	int maxSharedMemPerBlockOptin = 0;
	cudaDeviceGetAttribute(&maxSharedMemPerBlockOptin, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
	cudaFuncSetAttribute(wf_embedding_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, maxSharedMemPerBlockOptin);

	// define shared memory per block 
	int shared_mem = sizeof(float)*(block_size.x*(3 + d_model/2 + d_model/2 + d_model) + d_model/2);   // NI (x dim) stores NIxd_model output, 
																				// and two NIxd_model/2 (one for cos sums and another for sin) 
																				// and requires 3 floating point numbers per NI (x,y,z). 
																				// NJ is free, as everything is in registers 
																				// and the wavenumbers requires another d_model/2 floating point numbers.
																				// multiply by 4 bytes per fp32

	// Launch the kernel
	wf_embedding_kernel<<<grid_size, block_size, shared_mem, stream>>>(
		coords_ptr, stride_coords_Z, stride_coords_S, stride_coords_N,
		wavenumbers_ptr, stride_wavenumbers_K,
		out_ptr,  stride_out_Z, stride_out_N, stride_out_D,
		cos_sums_ptr, stride_cos_sums_Z, stride_cos_sums_N, stride_cos_sums_K,
		sin_sums_ptr, stride_sin_sums_Z, stride_sin_sums_N, stride_sin_sums_K,
		tot_Z, tot_N, d_model, magnitude_type
	);
}