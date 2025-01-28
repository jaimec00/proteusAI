
/* 
author:         jaime cardenas
title:          attn_fwd_kernel.cu
descripiton:    cuda kernel to perform geometric attention forward pass
*/

#include <cuda_runtime.h>

// device kernel for geometric attention forward pass
__global__ void attn_fwd_kernel(
    const float* Q_ptr, int stride_Q_Z, int stride_Q_H, int stride_Q_N, int stride_Q_D,
    const float* K_ptr, int stride_K_Z, int stride_K_H, int stride_K_N, int stride_K_D,
    const float* V_ptr, int stride_V_Z, int stride_V_H, int stride_V_N, int stride_V_D,
    const float* coords_ptr, int stride_coords_Z, int stride_coords_S, int stride_coords_N,
    const float* spreads_ptr, int stride_spreads_H,
    const float* rng_seed_ptr, int stride_rng_seed_Z, int stride_rng_seed_H,

    float* L, int stride_L_Z, int stride_L_H, int stride_L_N,
    float* O, int stride_O_Z, int stride_O_H, int stride_O_N, int stride_O_D,

    int softmax_scale, int dropout,
    int batch, int N, int nheads, int d_k
) {
	
	// compute global thread index
	int offs_I = blockIdx.x * blockDim.x + threadIdx.x;  // NI index, 1 thread id
	int offs_J = blockIdx.y * blockDim.y + threadIdx.y;  // NI index, 1 thread id
	int offs_ZH = blockIdx.z * blockDim.z + threadIdx.z;  // batch index, 1 thread id

	int offs_Z = offs_ZH / nheads 
	int offs_H = offs_ZH % nheads 

	// calculate the unique local id for each thread (will evaluate to theadIdx.y)
	int thread_id = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;
	int num_threads = blockDim.x * blockDim.y * blockDim.z; // number of threads in a block
	
	// compute warp id (within a block), then the lane id (thread id within a warp)
	int warp_id = thread_id / warpSize;
	int lane_id = thread_id % warpSize;

	// init dynamic shared mem and split into seperate arrays
	extern __shared__ __align__(4) float shared_mem[];

	// explicitly set all values to zero before any computations
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

	// now split into seperate shared mem arrays
	
}

// Host function to configure and launch the CUDA kernel
void attn_fwd(
    const float* Q_ptr, int stride_Q_Z, int stride_Q_H, int stride_Q_N, int stride_Q_D,
    const float* K_ptr, int stride_K_Z, int stride_K_H, int stride_K_N, int stride_K_D,
    const float* V_ptr, int stride_V_Z, int stride_V_H, int stride_V_N, int stride_V_D,
    const float* coords_ptr, int stride_coords_Z, int stride_coords_S, int stride_coords_N,
    const float* spreads_ptr, int stride_spreads_H,
    const float* rng_seed_ptr, int stride_rng_seed_Z, int stride_rng_seed_H,

    float* L, int stride_L_Z, int stride_L_H, int stride_L_N,
    float* O, int stride_O_Z, int stride_O_H, int stride_O_N, int stride_O_D,

    int softmax_scale, int dropout,
    int batch, int N, int nheads, int d_k

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
	attn_fwd_kernel<<<grid_size, block_size, shared_mem, stream>>>(
		Q_ptr, stride_Q_Z, stride_Q_H, stride_Q_N, stride_Q_D,
		K_ptr, stride_K_Z, stride_K_H, stride_K_N, stride_K_D,
		V_ptr, stride_V_Z, stride_V_H, stride_V_N, stride_V_D,

		coords_ptr, stride_coords_Z, stride_coords_S, stride_coords_N,
		spreads_ptr, stride_spreads_H,
		rng_seed_ptr, stride_rng_seed_Z, stride_rng_seed_H,

		L, stride_L_Z, stride_L_H, stride_L_N,
		O, stride_O_Z, stride_O_H, stride_O_N, stride_O_D,

		softmax_scale, dropout,
		batch, N, nheads, d_k
	);
}