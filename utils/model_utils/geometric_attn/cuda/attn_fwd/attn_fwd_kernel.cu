
/* 
author:         jaime cardenas
title:          attn_fwd_kernel.cu
descripiton:    cuda kernel to perform geometric attention forward pass. 
				adaptation of flash attention 3
*/

#include <cuda_runtime.h>
#include <cuda_fp16.h> // for half
#include <cstdint> // for unsigned 32 bit int


// device kernel for geometric attention forward pass
__global__ void attn_fwd_kernel(
    const __half* Q_ptr, int stride_Q_Z, int stride_Q_H, int stride_Q_N, int stride_Q_D,
    const __half* K_ptr, int stride_K_Z, int stride_K_H, int stride_K_N, int stride_K_D,
    const __half* V_ptr, int stride_V_Z, int stride_V_H, int stride_V_N, int stride_V_D,
    const float* coords_ptr, int stride_coords_Z, int stride_coords_S, int stride_coords_N,
    const float* spreads_ptr, int stride_spreads_H,

    float* L, int stride_L_Z, int stride_L_H, int stride_L_N,
    __half* O, int stride_O_Z, int stride_O_H, int stride_O_N, int stride_O_D,

    float softmax_scale, float dropout, uint32_t rng_seed, 
    int batch, int N, int nheads, int d_k,
	int TILE_SIZE, int BUFF_SIZE
) {

}

// Host function to configure and launch the CUDA kernel
void attn_fwd(
    const float* Q_ptr, int stride_Q_Z, int stride_Q_H, int stride_Q_N, int stride_Q_D,
    const float* K_ptr, int stride_K_Z, int stride_K_H, int stride_K_N, int stride_K_D,
    const float* V_ptr, int stride_V_Z, int stride_V_H, int stride_V_N, int stride_V_D,
    const float* coords_ptr, int stride_coords_Z, int stride_coords_S, int stride_coords_N,
    const float* spreads_ptr, int stride_spreads_H,

    float* L, int stride_L_Z, int stride_L_H, int stride_L_N,
    float* O, int stride_O_Z, int stride_O_H, int stride_O_N, int stride_O_D,

    float softmax_scale, float dropout, uint32_t rng_seed
    int batch, int N, int nheads, int d_k

    cudaStream_t stream
) {

	// define tile sizes for wmma (will make bigger and define diff sizes, since will be using wgmma instead)
	int TILE_SIZE = 16; // tile size
	int BUFF_SIZE = 2; // producers fetch the current iteration of Kj and VJ, as well as the next

	int num_warps = 16;
	int num_consumers = num_warps / 2;

	int BLOCK_I = TILE_SIZE*num_consumers;
	int BLOCK_J = TILE_SIZE*BUFF_SIZE

	// define block and grid dimensions
	dim3 block_size(32, num_warps, 1); // 16 warps per block, 8 producers, 8 consumers
	dim3 grid_size(
		(tot_N + BLOCK_I-1)/BLOCK_I, // each warp operates on 16 Qi at a time, each block has 8 consumer warps, so each block process 8*16 = 128 Qi
		1, // Kj, Vj looped through
		tot_Z*nheads // each block operates on a different Z and H combo
	);

	// configure the kernel to allow the maximum sram to maximize blocks/SM and more importantly warps/SM. 
	// h100 allows 228 kB, this is optimized for h100. will not test on other devices for now, but open to adjustments
	int device;	
	cudaGetDevice(&device);
	int maxSharedMemPerBlockOptin = 0;
	cudaDeviceGetAttribute(&maxSharedMemPerBlockOptin, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
	cudaFuncSetAttribute(attn_fwd_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, maxSharedMemPerBlockOptin);

	// define shared memory per block 
	int shared_mem = BLOCK_I*(sizeof(float)*(3 + 1 + 1) + sizeof(__half)*(d_k + d_k)) + BLOCK_J*(sizeof(float)*(3) + sizeof(__half)*(d_k + d_k)) + sizeof(uint8_t)*num_consumers;   
	// for each row, store x,y,z coords, Li, and mi in fp32, and the Qi and Oi in fp16
	// for each column, store x,y,z coords in fp32, and the Kj and Vj in fp16, multiplied by the buffer size (how many iterations of Kj and Vj are stored in shared mem by the producers)
	// also store a num_consumers=num_warps/2 8 bit bitmasks for communication between consumers and producers. each consumer/producer warp pair get BUFF_SIZE bits, but have extra room in case want to increase buffer size
	// Launch the kernel
	attn_fwd_kernel<<<grid_size, block_size, shared_mem, stream>>>(
		Q_ptr, stride_Q_Z, stride_Q_H, stride_Q_N, stride_Q_D,
		K_ptr, stride_K_Z, stride_K_H, stride_K_N, stride_K_D,
		V_ptr, stride_V_Z, stride_V_H, stride_V_N, stride_V_D,

		coords_ptr, stride_coords_Z, stride_coords_S, stride_coords_N,
		spreads_ptr, stride_spreads_H,

		L, stride_L_Z, stride_L_H, stride_L_N,
		O, stride_O_Z, stride_O_H, stride_O_N, stride_O_D,

		softmax_scale, dropout, rng_seed,
		batch, N, nheads, d_k,
		TILE_SIZE, BUFF_SIZE
	);
}