
/* 
author:         jaime cardenas
title:          wf_embedding_kernel.cu
descripiton:    cuda kernel to embed 3d coordinates to target feature space using 
				green's function solution to the helmholtz equation, modeling each
				token as a point source in 3d space, and superposing the effects of 
				the point sources 
*/

#include <cuda_runtime.h>
#include <cstdint>

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

__device__ uint32_t hash5(int a, int b, int c, int d, uint32_t seed) {

	// PCG-XSH-RR based hash algorithm
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
    uint32_t h = hash5(Z, I, J, K, seed);
    float normalized = h / float(UINT32_MAX); // Convert to [0,1]
    return normalized >= dropout_prob;
}

// device kernel for wavefunction embedding
__global__ void wf_embedding_kernel(
	const float* coordsA_ptr, int stride_coordsA_Z, int stride_coordsA_S, int stride_coordsA_N,
	const float* coordsB_ptr, int stride_coordsB_Z, int stride_coordsB_S, int stride_coordsB_N,
	const float* wavenumbers_ptr, int stride_wavenumbers_K, 

	float* out_ptr, int stride_out_Z, int stride_out_N, int stride_out_D,
	float* d_imag_ptr, int stride_d_imag_Z, int stride_d_imag_N, int stride_d_imag_K,
	float* d_real_ptr, int stride_d_real_Z, int stride_d_real_N, int stride_d_real_K,

	int tot_Z, int tot_N, int d_model, int magnitude_type, float dropout_p, uint32_t rng_seed
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
	float* d_imag = out + d_model;
	float* d_real = d_imag + d_model/2;
	
	// shared mem array for NI. only working with a single NI, 
	// which is loaded with the other NJ at the same time. 
	// during first iter have first thread put it in shared so
	// other threads can move it to their register
	float* coordsA_NI = d_real + d_model/2;

	// initialize these here to avoid compilation error, will be in each thread's register
	float coordsA_NI_x;
	float coordsA_NI_y;
	float coordsA_NI_z;
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
	// first move the coordsA pointer to this batch
	const float* coordsA_base_ptr = coordsA_ptr + (offs_Z*stride_coordsA_Z);

	// also move the coordsB pointer
	const float* coordsB_base_ptr = coordsB_ptr + (offs_Z*stride_coordsB_Z);

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
		// coordsA was transposed by pytorch to get 3xN tensor, so all x values are contiguous, same w/ y and z 
		// invalid threads still read from global mem, as this is still one memory transaction per warp since the reads are 
		// coalesced and we are guranteed to be in bounds since we cycle to the beginning (w/ mod tot_N)
		// possible that one iteration will have some threads in a warp at the end of the sequence and the others at beginning,
		// leading to uncoalesced mem access, but guranteed to happen <= 1 time per block, so no noticeable performance loss
		float coordsA_NJ_x = coordsA_base_ptr[0*stride_coordsA_S + offs_NJ]; 
		float coordsA_NJ_y = coordsA_base_ptr[1*stride_coordsA_S + offs_NJ];
		float coordsA_NJ_z = coordsA_base_ptr[2*stride_coordsA_S + offs_NJ];
		bool mask_NJ = thread_mask && (coordsA_NJ_x!=12345); // baked the mask into coordsA w/ this val, arbitrary, but inf makes it harder to avoid NaNs

		if (j==0) { // in the first iteration, thread0 loaded the NI token, so now distribute that information to the other threads
			
			// have the first thread move this to shared mem
			if (thread_id==0){ // introduces thread divergence, but only in the first iteration, after that it is very quick
				coordsA_NI[0] = coordsA_NJ_x;
				coordsA_NI[1] = coordsA_NJ_y;
				coordsA_NI[2] = coordsA_NJ_z;
			} // don't need coordsB for NI, just NJ

			// synchronize the threads before using the shared memory, only happens in the first iteration
			// this also happens before wavenumbers are read, so two birds w/ one stone
			__syncthreads();

			// now all threads move it from shared mem to their register		 
			coordsA_NI_x = coordsA_NI[0]; // no bank conflicts, it is broadcast
			coordsA_NI_y = coordsA_NI[1];
			coordsA_NI_z = coordsA_NI[2];
			mask_NI = thread_mask && (coordsA_NI_x!=12345); 

			// if NI is masked, stop the program. note all threads have the same NI, so the whole block will terminate
			if (!mask_NI) return;

		}

		// the distance computations. sets masked NJ dists to 0 to ensure mask operations are consistent
		float distA_x = mask_NJ*(coordsA_NI_x - coordsA_NJ_x); 
		float distA_y = mask_NJ*(coordsA_NI_y - coordsA_NJ_y);
		float distA_z = mask_NJ*(coordsA_NI_z - coordsA_NJ_z);

		// compute the distance and the masks
		float distsA_raw = distA_x*distA_x + distA_y*distA_y + distA_z*distA_z;
		bool mask_IJ = mask_NI && mask_NJ && (distsA_raw!=0); // prevent div by 0
		float inv_distA = mask_IJ * rsqrtf(distsA_raw + (!mask_IJ));
		float distsA = distsA_raw * inv_distA; // fast approximation of sqrt ( + !mask avoids div by 0, 
																			// mult by mask ensures invalid dists are 0 for consistent masking)

		// now compute dot product between distsA_xyz and coordsB, which is already computed by torch as the vector pointing from alpha to beta of source
		// call this dot product distsB
		float coordsB_NJ_x = coordsB_base_ptr[0*stride_coordsB_S + offs_NJ]; 
		float coordsB_NJ_y = coordsB_base_ptr[1*stride_coordsB_S + offs_NJ];
		float coordsB_NJ_z = coordsB_base_ptr[2*stride_coordsB_S + offs_NJ];

		// make these unit vectors, masked vals are zero
		float distA_x_norm = distA_x * inv_distA;
		float distA_y_norm = distA_y * inv_distA;
		float distA_z_norm = distA_z * inv_distA;

		// idk how to do wggma and i am running out of time so this will do lol
		float distsB = (coordsB_NJ_x*distA_x_norm) + (coordsB_NJ_y*distA_y_norm) + (coordsB_NJ_z*distA_z_norm);
		
		// subtract the dot product to create doppler-like effect
		float distsAB = distsA - distsB;
		
		float magnitude;
		switch (magnitude_type) {
			case 0: // no magnitude scaling, each observer sees the full effect of all sources
				magnitude = 1.0;
				break;
			case 1: // same magnitude as green's func, i.e. 1/|R|
				magnitude = inv_distA;
				break;
			case 2:	// take the log2 of the distance, i.e. 1/log2(|R|) to account more for distant interactions
				magnitude = 1.0 / log2f(distsA + 2*(!mask_IJ)); // evaluates to 1 / log2(0+2*1) = 1/1 = 1 for 0 dists. cos and sine are already masked
				break;
			case 3: // take the sqrt of dists 1/sqrt(dists). not as aggressive as log2, but still accounts for distant sources
				magnitude = mask_IJ * rsqrtf(distsA + (!mask_IJ));
				break;
		}

		// loop over wavenumbers in shared memory
		#pragma unroll 1
		for (int k = 0; k < num_wn; ++k) {

			// compute the phase
			// all threads access the same k from shared memory
			float phase = distsAB * wavenumbers[k];

			// compute sine and cosine
			// multiply by mask to zero out invalid threads
			float sine, cosine;
			__sincosf(phase, &sine, &cosine); // compute trig ops in one call
			cosine = mask_IJ*cosine;
			sine = mask_IJ*sine;

			// compute real and imaginary parts
			float real = cosine*magnitude;
			float imag = sine*magnitude;

			if (dropout_p != 0.0) {
				// lightweight hash based dropout, avoids overhead of built in cuda rngs
				bool drop_val = dropout( offs_Z, offs_NI, offs_NJ, k, rng_seed, dropout_p);
				real = drop_val * real / (1-dropout_p);
				imag = drop_val * imag / (1-dropout_p);
			}
			
			float real_superposition = warp_sum(real);
			float imag_superposition = warp_sum(imag);

			float cos_sum = warp_sum(real*distsAB); // d_imag
			float sin_sum = warp_sum(imag*distsAB); // d_real

			if (lane_id==0){ // first thread in the warp writes to mem

				// save the intermediate output in shared mem
				// no bank conflicts, a single thread per warp writes these
				// minimal contention, as only two threads overall are performing atomic adds
				atomicAdd(&out[2*k], real_superposition);
				atomicAdd(&out[2*k + 1], imag_superposition);

				// save these for bwd
				atomicAdd(&d_imag[k], cos_sum);
				atomicAdd(&d_real[k], sin_sum);

			}
		}
	}

	// initialize pointers for output and move them to proper NI
	float* block_out_ptr = out_ptr + (offs_Z*stride_out_Z) + (offs_NI*stride_out_N);
	float* block_d_imag_ptr = d_imag_ptr + (offs_Z*stride_d_imag_Z) + (offs_NI*stride_d_imag_N);
	float* block_d_real_ptr = d_real_ptr + (offs_Z*stride_d_real_Z) + (offs_NI*stride_d_real_N);

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
			block_d_imag_ptr[trig_idx] = d_imag[trig_idx];
			block_d_real_ptr[trig_idx] = d_real[trig_idx];	
		}
	}
}

// Host function to configure and launch the CUDA kernel
void wf_embedding_kernel_forward(
	const float* coordsA_ptr, int stride_coordsA_Z, int stride_coordsA_S, int stride_coordsA_N,
	const float* coordsB_ptr, int stride_coordsB_Z, int stride_coordsB_S, int stride_coordsB_N,
	const float* wavenumbers_ptr, int stride_wavenumbers_K, 

	float* out_ptr, int stride_out_Z, int stride_out_N, int stride_out_D,
	float* d_imag_ptr, int stride_d_imag_Z, int stride_d_imag_N, int stride_d_imag_K,
	float* d_real_ptr, int stride_d_real_Z, int stride_d_real_N, int stride_d_real_K,

	int tot_Z, int tot_N, int d_model, int magnitude_type, float dropout_p, uint32_t rng_seed,
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
		coordsA_ptr, stride_coordsA_Z, stride_coordsA_S, stride_coordsA_N,
		coordsB_ptr, stride_coordsB_Z, stride_coordsB_S, stride_coordsB_N,
		wavenumbers_ptr, stride_wavenumbers_K,
		out_ptr,  stride_out_Z, stride_out_N, stride_out_D,
		d_imag_ptr, stride_d_imag_Z, stride_d_imag_N, stride_d_imag_K,
		d_real_ptr, stride_d_real_Z, stride_d_real_N, stride_d_real_K,
		tot_Z, tot_N, d_model, magnitude_type, dropout_p, rng_seed
	);
}