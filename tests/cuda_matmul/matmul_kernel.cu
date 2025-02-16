
/* 
author:         jaime cardenas
title:          matmul_kernel.cu
descripiton:    matmul test
*/

#include <cuda_runtime.h>
#include <stdio.h>
#include <cuda_fp16.h>

#include "cutlass/arch/barrier.h"
#include "cutlass/pipeline/sm90_pipeline.hpp"
#include "cutlass/arch/mma_sm90.h"

__global__ void matmul_kernel(
    const half_t* A_ptr,
    const half_t* B_ptr,
    const half_t* C_ptr,
    half_t* D_ptr,

	int M, int N, int K

) {

}

// Host function to configure and launch the CUDA kernel
void main(
    const half_t A_ptr, const half_t B_ptr, half_t C_ptr, half_t D_ptr,
	cudaStream_t stream
) {

	// define tile sizes for wmma (will make bigger and define diff sizes, since will be using wgmma instead)
	int M = 64; // tile size
	int N = 128; // tile size
	int K = 16; // tile size

	// define block and grid dimensions
	dim3 block_size(32, num_warps, 1); // 16 warps per block, 8 producers, 8 consumers
	dim3 grid_size(
		(tot_N + BLOCK_I-1)/BLOCK_I, 
		1,
		1 
	);

	// configure the kernel to allow the maximum sram to maximize blocks/SM and more importantly warps/SM. 
	// h100 allows 228 kB, this is optimized for h100. will not test on other devices for now, but open to adjustments
	int device;	
	cudaGetDevice(&device);
	int maxSharedMemPerBlockOptin = 0;
	cudaDeviceGetAttribute(&maxSharedMemPerBlockOptin, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
	cudaFuncSetAttribute(attn_fwd_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, maxSharedMemPerBlockOptin);

	// define shared memory per block 
	int shared_mem = (BLOCK_I*K + BLOCK_J*BUFF_SIZE*K + BLOCK_I*BLOCK_J*K*2)*sizeof(__half)
	// for each row, store x,y,z coords, Li, and mi in fp32, and the Qi and Oi in fp16
	// for each column, store x,y,z coords in fp32, and the Kj and Vj in fp16, multiplied by the buffer size (how many iterations of Kj and Vj are stored in shared mem by the producers)
	// also store a num_consumers=num_warps/2 8 bit bitmasks for communication between consumers and producers. each consumer/producer warp pair get BUFF_SIZE bits, but have extra room in case want to increase buffer size
	// Launch the kernel
	matmul_kernel<<<grid_size, block_size, shared_mem, stream>>>(
		A_ptr, B_ptr, C_ptr, D_ptr,
		M, N, K
	);
}