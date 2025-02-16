
/* 
author:         jaime cardenas
title:          attn_fwd_kernel.cu
descripiton:    cuda kernel to perform geometric attention forward pass. 
				adaptation of flash attention 3
*/

#include <cuda_runtime.h>
#include <cuda_fp16.h> // for half
#include <cstdint> // for unsigned 32 bit int

__device__ uint32_t hash_idxs(int idx1, int idx2) {

	// hashes to idxs to a single 32 bit unisgned number

	uint32_t hash_val = (idx1) ^ (idx2);
	uint32_t hash_val = (hash_val << 2) ^ ((idx1 >> 2) | (idx2 << 4));
	uint32_t hash_val = (hash_val << 4) ^ (hash_val >> 2); 

	return hash_val;
}

__device__ __half mask_val(__half val, bool condition){

	// masks vals to -inf if a condition is false, otherwise, returns the original val

	// 16 bit representation of -inf. -inf is 0xFC00 (1111100000000000). no built in -inf for half
	// -inf for masked, 0 for unmasked. will do val + bias, which results in -inf or val
	// if condition is true, shifts 5 bits to the left to get 16 bit representation of 0 (0x0000)
	// if false, no shift, so get -inf
	__half dropout_bias = 0xFC00 << (5*condition);

	__half new_val = __hadd(val, dropout_bias);

	return new_val;
}
__device__ __half dropout(int Z, int H, int I, int J, uint32_t rng_seed, float dropout_p, __half val) {

	// do a hash(Z,I) for sequence and hash(H,J) for offset. this decouples i and j 
	// to avoid symmetry issues, while offering better randomness by varying streams and offsets,
	// while still being deterministic

	// hash batch and i into single unsigned 32 bit number that determines the sequence
	uint32_t sequence = hash_idxs(Z, I);

	// hash H and J into single unsigned 32 bit number that determines the offset
	uint32_t offset = hash_idxs(H, J);

	// initialize the state
	curandStatePhilox4_32_10_t state;
	curand_init(rng_seed, sequence, offset, &state);

	// generate random number
	float rand_val = curand_uniform(&state);

	// masking values to -inf
	bool drop_val = rand_val < dropout_p;

	// -inf w/ prob dropout_p, val / 1-dropout w/ prob 1-dropout_p
	__half new_val = __hdiv(mask_val(val, drop_val), __float2half(1 - dropout_p));

	return new_val;
}

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

	// set max registers to low value for loading phase. producers keep small value, while consumers increase it in the computation phase
	asm volatile("setmaxnreg 32;");
	
	// calculate the unique local id for each thread
	int thread_id = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;
	int num_threads = blockDim.x * blockDim.y * blockDim.z; // number of threads in a block
	
	// compute warp id (within a block), then the lane id (thread id within a warp), then the number of warps
	int warp_id = thread_id / warpSize;
	int lane_id = thread_id % warpSize;
	int num_warps = num_threads / warpSize;

	// split into consumers and producers, only define one
	bool is_producer = warp_id >= (num_warps/2)

	// get the Qi offset for this block. each block computes, num_warps/2 * TILE_SIZE rows of Qi
	// producers dont count, thus the /2
	// each warp gets a starting offset. producers get a val, but not used
	int BLOCK_I = TILE_SIZE*num_warps/2;
	int BLOCK_J = TILE_SIZE*BUFF_SIZE;
	
	// this is the starting offset of the block
	int start_offs_I = blockIdx.x*BLOCK_I; 
	int offs_I = start_offs_I + warp_id*TILE_SIZE; // this is the warp specific offset

	// get batch head combo for this block
	int offs_ZH = blockIdx.z * blockDim.z;  
	int offs_Z = offs_ZH / nheads 
	int offs_H = offs_ZH % nheads 

	// init dynamic shared mem and split into seperate arrays. initialize as char for granular control
	extern __shared__ char shared_mem[];

	// explicitly set all values to zero before any computations
	// compute the number of elements in bytes
	int half_mem_size = (d_k*BLOCK_I*2 + d_k*BLOCK_J*2) * sizeof(__half);  // Qi, Oi, Kj, Vj
	int float_mem_size = (BLOCK_I*5 + 3*BLOCK_J) * sizeof(float);  // Li, mi, coords_Ixyz, coords_Jxyz, consumer/producer bitmasks
	int shared_mem_elements = half_mem_size + float_mem_size + sizeof(uint8_t)*num_warps/2;	

	// compute number of iterations
	int shared_mem_iters = (shared_mem_elements + num_threads - 1) / num_threads;
	
	// init shared mem to 0
	#pragma unroll 1 // disable loop unrolling so that threads reuse registers, implemented in every loop
	for (int mem_idx = 0; mem_idx < shared_mem_iters; ++mem_idx){
		int thread_mem_idx = mem_idx*num_threads + thread_id;
		bool mem_idx_valid = thread_mem_idx < shared_mem_elements;
		if (mem_idx_valid){ 
			shared_mem[thread_mem_idx] = 0;
		}
	}

	// first compute the arrays of type __half, aligns to next multiple of 2 address
	__half* half_shared_mem = (__half*)((reinterpret_cast<size_t>(shared_mem) + 1) & ~1);

	// to avoid bank conflicts, combine Qi and Oi into a single interleaved tensor, since these are fp16
	__half* QiOi = half_shared_mem;

	// similarly, combine Kj and Vj into single interleaved tensor
	__half* KjVj = QiOi + (2*d_k*BLOCK_I);

	// compute the starting offset for floats. aligns it to the next multiple of 4 address
	size_t half_offset = ((size_t)(KjVj + (2*d_k*BLOCK_J) - half_shared_mem) * sizeof(__half) + sizeof(float) - 1) & ~(sizeof(float) - 1);

	// split into arrays of floats
	float* full_shared_mem = (float*)(shared_mem + half_offset);
	float* Li = full_shared_mem;
	float* mi = Li + BLOCK_I; 
	float* coords_Ix = mi + BLOCK_I;
	float* coords_Iy = coords_Ix + BLOCK_I;
	float* coords_Iz = coords_Iy + BLOCK_I;
	float* coords_Jx = coords_Iz + BLOCK_I;
	float* coords_Jy = coords_Jx + BLOCK_J;
	float* coords_Jz = coords_Jy + BLOCK_J;
	float* spread = coords_Jz + BLOCK_J; // only one spread stored

	// still aligned to four bytes. allocate shared mem for bitmask used for consumer-producer communication
	size_t float_offset = (size_t)((spread + 1 - half_shared_mem) * sizeof(float));
	uint8_t* byte_shared_mem = (uint8_t)(shared_mem + float_offset)

	// sync threads when done initializing shared memory
	__syncthreads();

	// first load all I values, will be moved to registers, might change this to be loaded directly by consumers to bypass SMEM,
	// but need to write it out to understand where the optimizations might be
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	// load Qi
	// move Qi to right batch, head offset
	Qi_ptr += offs_Z*stride_Q_Z + offs_H*stride_Q_H;
	int Qi_iters = (BLOCK_I*d_k + num_threads - 1) / num_threads;
	#pragma unroll 1
	for (int i=0; i < Qi_iters; ++i) {
		int i_idx = start_offs_I*d_k + num_threads*i + thread_id;
		bool i_valid = i_idx < (tot_N*d_k);
		if (i_valid){
			// QiOi is interleaved tensor, Qi has even indexes. avoids bank conflicts
			QiOi[2*i_idx] = Qi_ptr[i_idx]; 
		}
	}

	// first move the coords to this batch head combo (also takes care of J offset)
	coords_ptr += offs_Z*stride_Q_Z + offs_H*stride_Q_H;
	// now move the pointer just for coords I, not coords J starts at position 0
	coords_I_ptr = coords_ptr + offs_I*stride_coords_N;
	// load coords_Ixyz. also set mi to -inf in this loop while we're at it
	int coordsI_iters = (BLOCK_I + num_threads - 1) / num_threads;
	#pragma unroll 1
	for (int i=0; i < coordsI_iters; ++i) {
		int i_idx = offs_I + num_threads*i + thread_id;
		bool i_valid = i_idx < (tot_N);
		if (i_valid){
			// these are all fp32, so no bank conflicts
			coords_Ix[i_idx] = coords_I_ptr[0*stride_coords_S + i_idx]; 
			coords_Iy[i_idx] = coords_I_ptr[1*stride_coords_S + i_idx]; 
			coords_Iz[i_idx] = coords_I_ptr[2*stride_coords_S + i_idx];
			mi[i_idx] = -INFINITY;
		}
	}

	// load first tile
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	// load coords_Jxyz
	// only load the first tile so the producers have work to do in the loop
	int coordsJ_iters = (TILE_SIZE + num_threads - 1) / num_threads;
	#pragma unroll 1
	for (int j=0; j < coordsJ_iters; ++j) {
		int j_idx = num_threads*j + thread_id; // starts at 0, unlike I
		bool j_valid = j_idx < (tot_N);
		if (j_valid){
			coords_Jx[j_idx] = coords_J_ptr[0*stride_coords_S + j_idx]; 
			coords_Jy[j_idx] = coords_J_ptr[1*stride_coords_S + j_idx]; 
			coords_Jz[j_idx] = coords_J_ptr[2*stride_coords_S + j_idx];
		}
	}

	// load first iter of Kj, Vj
	int KjVj_iters = (TILE_SIZE*d_k + num_threads - 1) / num_threads;
	#pragma unroll 1
	for (int j=0; j < KjVj_iters; ++j) {
		int j_idx = num_threads*j + thread_id;
		bool j_valid = j_idx < (tot_N*d_k);
		if (j_valid){
			// KjVj is interleaved to avoid bank conflicts, Kj is even idxs, Vj s odd
			KjVj[2*j_idx] = Kj_ptr[j_idx];
			KjVj[2*j_idx+1] = Vj_ptr[j_idx];
		}
	}

	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	// sync threads after done loading
	__syncthreads();

	// setup is finally done, now have the warps specialize
	if (is_producer){

		// these warps keep max registers as 64, so nothing to do there

		// loop through blocks of TILE_SIZE
			// wait until consumers need the next iteration of data.
			// if they have, load the next iter data
			// load Kj, Vj, coords_Jxyz


	} else { // is consumer

		// allocate registers that producers freed up 
		asm volatile("setmaxnreg 255;");

		// first move the constants from smem to rmem (Qi, Li_old, mi_old, coords_Ixyz, spread). note Li_old and mi_old are updated in the loop

		// loop through blocks of J

			// check if the producers have loaded the next iteration of data
			// if so, read Kj, Vj, coords_Jxyz
			// let the producers know this has been read and they can load the next iteration

			// FA3 logic

	}

	// epilogue (all threads do this)

	// all threads write to global memory

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