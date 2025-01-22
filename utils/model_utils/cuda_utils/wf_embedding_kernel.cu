
/* 
author:         jaime cardenas
title:          wf_embedding_kernel.cu
descripiton:    cuda kernel to embed 3d coordinates to target feature space using 
                green's function solution to the helmholtz equation, modeling each
                token as a point source in 3d space, and superposing the effects of 
                the point sources 
*/

#include <cuda_runtime.h>

// device kernel for wavefunction embedding
__global__ void wf_embedding_kernel(
    const float* coords_ptr, int stride_coords_Z, int stride_coords_N, int stride_coords_S,
    const float* wavenumbers_ptr, int stride_wavenumbers_K, 
    const bool* mask_ptr, int stride_mask_Z, int stride_mask_N,

    float* out_ptr, int stride_out_Z, int stride_out_N, int stride_out_D,
    float* cos_sums_ptr, int stride_cos_sums_Z, int stride_cos_sums_N, int stride_cos_sums_K,
    float* sin_sums_ptr, int stride_sin_sums_Z, int stride_sin_sums_N, int stride_sin_sums_K,

    int tot_Z, int tot_N, int d_model
) {
    // have 32 warps per block (1024 threads_per_block / 32 threads_per_warp)
    // since block_NI is 32, each warp works on a distinct NI (will reduce along NJ, as each NI is independant)
    // there are 32 threads per warp, which equals block_NJ, so each thread within a 
    // warp works to compute it's NJ's impact on the warp's NI
    // first 32 threads along x dimension are the first warp, each having the same NI but different NJ

    // compute global thread index
    int offs_NJ = blockIdx.x * blockDim.x + threadIdx.x;  // NJ index
    int offs_NI = blockIdx.y * blockDim.y + threadIdx.y;  // NI index
    int offs_Z = blockIdx.z * blockDim.z + threadIdx.z;  // batch index

    // calculate the unique local id for each thread
    int thread_id = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;
    
    // compute warp id (within a block), then the lane id (thread id within a warp)
    int warp_id = thread_id / warpSize;
    int lane_id = thread_id % warpSize;

    // shared memory for these values, as only the necessary warps fetch these from global memory in parallel
    extern __shared__ float wavenumbers[];

    extern __shared__ float coords_NI_x[];
    extern __shared__ float coords_NI_y[];
    extern __shared__ float coords_NI_z[];

    extern __shared__ float coords_NJ_x[];
    extern __shared__ float coords_NJ_y[];
    extern __shared__ float coords_NJ_z[];

    extern __shared__ bool mask_NI[];
    extern __shared__ bool mask_NJ[];

    // now, assign each warp a subset of wavenumber idxs, so each warp loads from global memory into shared memory in parallel
    // since there are 1024 threads, and for my case there are 256 wavenumbers, only 8/32 warps fetch this memory, 
    // i wont code the case where there are more wavenumbers than threads because i need max performance for my model, 
    // but if this kernel becomes popular i'll add it
    // while this is happening, the idle warps will load the NI coords and NJ coords there are only 32 NI and 32 NJ 
    // values, however, we have xyz coordinates, so need 64 warps total, note that the coords tensor is transposed 
    // before being passed to the kernel, so all x values are contiguous, and y, and z, so one warp gets 32 x vals, 
    // another the corresponding y, etc and an additional two get the masks
    // this brings the total to 16 warps being active while the other half are inactive

    // all available to start
    bool available_warps = thread_id > -1 

    int num_wavenumbers = d_model/2;
    int num_wavenumber_warps = num_wavenumbers / warpSize
    bool read_wavenumbers = warp_id < num_wavenumber_warps 
    available_warps = available_warps ^ read_wavenumbers // XOR to remove these warps from being available

    // partition warps to 
    bool read_NI_x = warp_id   

    // move the pointer to this batch
    float* coords_base_ptr = coords_ptr + (offs_Z*stride_coords_Z)

    // read from global and write to shared memory
    if (read_wavenumbers){
        wavenumbers[thread_id] = wavenumbers_ptr[thread_id*stride_wavenumbers_K];
    } elif (read_NI_x) {
        coords_NI_x[]
    } elif (read_NI_y) {

    } elif (read_NI_z) {

    } elif (read_NJ_x) {

    } elif (read_NJ_y) {

    } elif (read_NJ_z) {

    } elif (read_NI_mask) {

    } elif (read_NJ_masl) {

    }

    // mask out of bounds threads
    const bool thread_mask_NI = (offs_Z < tot_Z) && (offs_NI < tot_N); 
    const bool thread_mask_NJ = (offs_Z < tot_Z) && (offs_NJ < tot_N);

    // move the NI and NJ coords pointers
    // access zeroth element of the coords if thread should be masked.
    // need to be sure that consistently do this throughout the rest of the code, 
    // as invalid threads now have an arbitrary value. still best to do this, as it ensures no warp divergence
    
    // to avoid out-of-bounds memory accesses, if a thread is invalid, thread_mask==0 and it accesses the first element
    // masks from torch are 0 for valid positions, so invert it w/ !
    const bool mask_NI = (thread_mask_NI) && !(mask_ptr[thread_mask_NI*(offs_Z*stride_mask_Z + offs_NI*stride_mask_N)]);
    const float* coords_NI_ptr = coords_ptr + mask_NI * (offs_Z*stride_coords_Z + offs_NI*stride_coords_N);
    float3 coords_NI = make_float3(coords_NI_ptr) // assumes coalesced mem, which should be true

    const bool mask_NJ = (thread_mask_NJ) && !(mask_ptr[thread_mask_NJ*(offs_Z*stride_mask_Z + offs_NJ*stride_mask_N)]);
    const float* coords_NJ_ptr = coords_ptr + mask_NJ * (offs_Z*stride_coords_Z + offs_NJ*stride_coords_N);
    float3 coords_NJ = make_float3(coords_NJ_ptr)





    bool mask_IJ = (mask_NI && mask_NJ);


    // compute the distances
    float3 dists_raw = make_float3( coords_NI_ptr[0*stride_coords_S] - coords_NJ_ptr[0*stride_coords_S], 
                                    coords_NI_ptr[1*stride_coords_S] - coords_NJ_ptr[1*stride_coords_S], 
                                    coords_NI_ptr[2*stride_coords_S] - coords_NJ_ptr[2*stride_coords_S]
                                );
    float dists_sqrd = dists_raw.x*dists_raw.x + dists_raw.y*dists_raw.y + dists_raw.z*dists_raw.z; // x^2 + y^2 + z^2
    
    // mask 0 distances, do this before compute rsqrtf as sqrt(0)=0, to avoids division by zero in rsqrtf
    // in theory, this should already be taken care of with NI!=NJ in thread_mask, but just in case
    mask_IJ = mask_IJ && (dists_sqrd!=0);

    float dists = dists_sqrd * rsqrtf(dists_sqrd + (1-mask_IJ)); // fast approximation of sqrt
                                                                // add 1-mask_IJ so for valid threads, 1-1 = 0,
                                                                // but for invalid, 1-0 = 1, avoiding division
                                                                // by zero (invalid threads accessed coords[0] for both NI and NJ, so this is necessary)



    // sync the threads, all wavenumbers must be loaded
    __syncthreads();

    // initialize base pointers for output to avoid recomputing in the loop, only have to increment the feature index
    // note that only writing for NI
    float* block_out_ptr = out_ptr + (offs_Z*stride_out_Z) + (offs_NI*stride_out_N);
    float* block_cos_sums_ptr = cos_sums_ptr + (offs_Z*stride_cos_sums_Z) + (offs_NI*stride_cos_sums_N);
    float* block_sin_sums_ptr = sin_sums_ptr + (offs_Z*stride_sin_sums_Z) + (offs_NI*stride_sin_sums_N);

    // Loop over wavenumbers in shared memory
    for (int k = 0; k < num_wavenumbers; ++k) {

        // all threads access the same k from shared memory
        float wavenumber = wavenumbers[k];

        // compute the phase
        float phase = dists * wavenumber;

        // compute sine and cosine
        // multiply by mask to zero out invalid threads
        float cos = mask_IJ * __cosf(phase);  // Fast approximation of cosine
        float sin = mask_IJ * __sinf(phase);  // Fast approximation of sine

        // store sum of cosines and sins for backward pass
        // masked threads write to first k, but their value is zero so no effect on output
        atomicAdd(&block_cos_sums_ptr[k*stride_cos_sums_K*(mask_IJ)], cos);
        atomicAdd(&block_sin_sums_ptr[k*stride_sin_sums_K*(mask_IJ)], sin);

        // compute real and imaginary parts
        // divide by one for invalid to avoid nans
        // note that cos and sin are 0 for invalid threads already
        float real = cos / (dists + (1-mask_IJ));
        float imag = sin / (dists + (1-mask_IJ));

        // store results, note that real and imaginary parts are interleaved
        atomicAdd(&block_out_ptr[(2*k)*stride_out_D*(mask_IJ)], real);
        atomicAdd(&block_out_ptr[(2*k + 1)*stride_out_D*(mask_IJ)], imag);

    }   
}

// define function to sum the values of threads within a warp
// scales O(log_2(32)) = O(5), very fast
__device__ int warp_reduce_sum(int value) {
    unsigned mask = 0xFFFFFFFF; // mask for active threads in the warp
    for (int offset = 16; offset > 0; offset /= 2) {
        value += __shfl_down_sync(mask, value, offset);
    }
    return value; // result is available in all threads of the warp
}

// Host function to configure and launch the CUDA kernel
void wf_embedding_kernel_forward(
    const float* coords_ptr, int stride_coords_Z, int stride_coords_S, int stride_coords_N,
    const float* wavenumbers_ptr, int stride_wavenumbers_K, 
    const bool* mask_ptr, int stride_mask_Z, int stride_mask_N,

    float* out_ptr, int stride_out_Z, int stride_out_N, int stride_out_D,
    float* cos_sums_ptr, int stride_cos_sums_Z, int stride_cos_sums_N, int stride_cos_sums_K,
    float* sin_sums_ptr, int stride_sin_sums_Z, int stride_sin_sums_N, int stride_sin_sums_K,

    int tot_Z, int tot_N, int d_model, 
    cudaStream_t stream
) {
    // define block and grid dimensions
    dim3 block_size(32, 32, 1); // 32 threads for NI, 32 NJ, 1 for Z (1 for Z to ensure max parallelism along batch dim)
    dim3 grid_size(
        (tot_N + block_size.x - 1) / block_size.x,
        (tot_N + block_size.y - 1) / block_size.y,
        (tot_Z + block_size.z - 1) / block_size.z 
    );

    // wavenumbers will be in shared memory, which is of size d_model/2, and takes 4 bytes per float
    int shared_mem = 16*32*4;   // in my kernel 16 warps fetch from global mem, 
                                // each warp has 32 threads, each of which accesses 
                                // a single element, each of which is 4 bytes

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