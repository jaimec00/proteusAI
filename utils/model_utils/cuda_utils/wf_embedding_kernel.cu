
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

    int tot_Z, int tot_N, int d_model, 
    cudaStream_t stream
) {
    // compute global thread index
    int offs_NI = blockIdx.x * blockDim.x + threadIdx.x;  // NI index
    int offs_NJ = blockIdx.y * blockDim.y + threadIdx.y;  // NJ index
    int offs_Z = blockIdx.z * blockDim.z + threadIdx.z;  // batch index

    // mask out of bounds threads, or threads that have identical NI and NJ (skip diagonals)
    bool thread_mask = (offs_Z < tot_Z) && (offs_NI < tot_N) && (offs_NJ < tot_N) && (offs_NI!=offs_NJ);

    // to avoid out-of-bounds memory accesses, if a thread is invalid, thread_mask==0 and it accesses the first element
    bool mask_NI = mask_ptr[thread_mask*(offs_Z*stride_mask_Z + offs_NI*stride_mask_N)];
    bool mask_NJ = mask_ptr[thread_mask*(offs_Z*stride_mask_Z + offs_NJ*stride_mask_N)];
    bool mask_IJ = thread_mask && !(mask_NI || mask_NJ);    // masks from torch are 0 for valid positions, so invert it w/ !
                                                            // also note invalid threads are still invalid even though 
                                                            // accessed mask of first element

    // move the NI and NJ coords pointers
    // access zeroth element of the coords if thread should be masked.
    // need to be sure that consistently do this throughout the rest of the code, 
    // as invalid threads now have an arbitrary value. still best to do this, as it ensures no warp divergence
    float* coords_NI_ptr = coords_ptr + mask_IJ * (offs_Z*stride_coords_Z + offs_NI*stride_coords_N);
    float* coords_NJ_ptr = coords_ptr + mask_IJ * (offs_Z*stride_coords_Z + offs_NJ*stride_coords_N);
    
    // compute the distances
    float3 dists_raw = make_float3( coords_NI_ptr[0*stride_coords_S] - coords_NJ_ptr[0*stride_coords_S], 
                                    coords_NI_ptr[1*stride_coords_S] - coords_NJ_ptr[1*stride_coords_S], 
                                    coords_NI_ptr[2*stride_coords_S] - coords_NJ_ptr[2*stride_coords_S]
                                );
    float dists_sqrd = dot(dists_raw, dists_raw); // x^2 + y^2 + z^2
    
    // mask 0 distances, do this before compute rsqrtf as sqrt(0)=0, to avoids division by zero in rsqrtf
    // in theory, this should already be taken care of with NI!=NJ in thread_mask, but just in case
    bool mask_IJ = mask_IJ && (dists_sqrd!=0);

    float dists = dists_sqrd * rsqrtf(dists_sqrd + (1-mask_IJ)); // fast approximation of sqrt
                                                                // add 1-mask_IJ so for valid threads, this = 0,
                                                                // but for invalid, this = 1-0 = 1, avoiding division
                                                                // by zero (invalid threads accessed coords[0] for both NI and NJ, so this is necessary)

    // shared memory for wavenumbers, as all threads operate on the same wavenumbers
    extern __shared__ float wavenumbers[];

    // now, assign each thread a group of wavenumber idxs, so each thread loads from global memory into shared memory in parallel
    // all threads, regardless of their mask_IJ value, work to get the wavenumbers from global memory, unless there are more threads than wavenumbers
    
    // compute how many wavenumbers each thread must get
    int num_wavenumbers = d_model/2;
    int block_size =  blockDim.x * blockDim.y * blockDim.z;
    int wavenumbers_per_thread = (num_wavenumbers + block_size - 1) / block_size; // cdiv

    // calculate the unique idx for each thread
    int wavenumber_idx = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;
    
    // loop until all wavenumbers have been written to shared mem
    // should only be one iteration in most cases, but just incase there are more wavenumbers than threads
    for (int i=0; i<wavenumbers_per_thread; ++i){

        // mask out redundant threads
        bool wavenumber_mask = wavenumber_idx < num_wn;

        // redundant threads access the zeroth element of global wavenumbers and write to 0th element of shared
        int thread_wavenumber = wavenumber_mask*wavenumber_idx*stride_wavenumbers_K; 

        // read from global and write to shared
        wavenumbers[thread_wavenumber] = wavenumbers_ptr[thread_wavenumber];

        // increment the wavenumber idx for next iter
        wavenumber_idx += block_size;
    }

    // sync the threads, all wavenumbers must be loaded
    __syncthreads();

    // initialize base pointers for output to avoid recomputing in the loop, only have to increment the feature index
    // note that only writing for NI
    float* out_ptr = out_ptr + (offs_Z*stride_out_Z) + (offs_NI*stride_out_N);
    float* cos_sums_ptr = cos_sums_ptr + (offs_Z*stride_cos_sums_Z) + (offs_NI*stride_cos_sums_N);
    float* sin_sums_ptr = sin_sums_ptr + (offs_Z*stride_sin_sums_Z) + (offs_NI*stride_sin_sums_N);

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
        atomicAdd(&cos_sums_ptr[k*stride_cos_sums_K], cos);
        atomicAdd(&sin_sums_ptr[k*stride_sin_sums_K], sin);

        // compute real and imaginary parts
        // divide by one for invalid to avoid nans
        // note that cos and sin are 0 for invalid threads already
        float real = cos / (dists + (1-mask_IJ));
        float imag = sin / (dists + (1-mask_IJ));

        // store results, note that real and imaginary parts are interleaved
        atomicAdd(&out_ptr[(2*k)*stride_out_D], real);
        atomicAdd(&out_ptr[(2*k + 1)*stride_out_D], imag);

    }   
}

// Host function to configure and launch the CUDA kernel
void wf_embedding_kernel_forward(
    const float* coords_ptr, int stride_coords_Z, int stride_coords_N, int stride_coords_S,
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
    int shared_mem = 4*(d_model/2);

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