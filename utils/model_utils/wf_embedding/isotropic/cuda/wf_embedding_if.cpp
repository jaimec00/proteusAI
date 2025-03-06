
#include <torch/extension.h>  // for pytorch c++ extensions
#include <ATen/cuda/CUDAContext.h>  // for accessing cuda streams
#include <cuda_runtime.h>  // for cuda runtime functions
#include <cstdint>

// declare the cuda kernel implemented in wf_embedding_kernel.cu
void wf_embedding_kernel_forward(
    const float* coords, int stride_coords_Z, int stride_coords_S, int stride_coords_N,
    const float* wavenumbers, int stride_wavenumbers_K, 

    float* out, int stride_out_Z, int stride_out_N, int stride_out_D,
    float* cos_sums, int stride_cos_sums_Z, int stride_cos_sums_N, int stride_cos_sums_K,
    float* sin_sums, int stride_sin_sums_Z, int stride_sin_sums_N, int stride_sin_sums_K,

    int tot_Z, int tot_N, int d_model, int magnitude_type, float dropout_p, uint32_t rng_seed,
    cudaStream_t stream
);

void wf_embedding_forward(
    torch::Tensor coords, torch::Tensor wavenumbers, 
    torch::Tensor out, 
    torch::Tensor cos_sums, torch::Tensor sin_sums,
    int magnitude_type, float dropout_p, uint32_t rng_seed
) {


    // validate inputs
    TORCH_CHECK(coords.device().is_cuda(), "coords must be a CUDA tensor");
    TORCH_CHECK(wavenumbers.device().is_cuda(), "wavenumbers must be a CUDA tensor");
    TORCH_CHECK(out.device().is_cuda(), "out must be a CUDA tensor");
    TORCH_CHECK(cos_sums.device().is_cuda(), "cos_sums must be a CUDA tensor");
    TORCH_CHECK(sin_sums.device().is_cuda(), "sin_sums must be a CUDA tensor");

    TORCH_CHECK(coords.dtype() == torch::kFloat32, "coords must be of type float32");
    TORCH_CHECK(wavenumbers.dtype() == torch::kFloat32, "wavenumbers must be of type float32");
    TORCH_CHECK(out.dtype() == torch::kFloat32, "out must be of type float32");
    TORCH_CHECK(cos_sums.dtype() == torch::kFloat32, "out must be of type float32");
    TORCH_CHECK(sin_sums.dtype() == torch::kFloat32, "out must be of type float32");

    // get tensor sizes 
    int tot_Z = coords.size(0); // batch size
    int tot_N = coords.size(2); // sequence size (transposed by my autograd function to be Z x 3 x N for coalesced memory accesses)
    int d_model = wavenumbers.size(0)*2; // feature size (num_wn == d_model//2)

    TORCH_CHECK(out.size(0) == tot_Z, "out batch size mismatch");
    TORCH_CHECK(out.size(1) == tot_N, "out sequence size mismatch");
    TORCH_CHECK(out.size(2) == d_model, "out d_model size mismatch");

    // get raw pointers
    const float* coords_ptr = reinterpret_cast<const float*>(coords.data_ptr<float>());
    const float* wavenumbers_ptr = reinterpret_cast<const float*>(wavenumbers.data_ptr<float>());
    float* out_ptr = reinterpret_cast<float*>(out.data_ptr<float>());
    float* cos_sums_ptr = reinterpret_cast<float*>(cos_sums.data_ptr<float>());
    float* sin_sums_ptr = reinterpret_cast<float*>(sin_sums.data_ptr<float>());

    // launch the cuda kernel
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();  // get pytorch's current cuda stream
    // cast strides to doubel to prevent indexing errors
    wf_embedding_kernel_forward(    
        coords_ptr, 
        coords.stride(0), coords.stride(1), coords.stride(2),
        wavenumbers_ptr, 
        wavenumbers.stride(0),
        out_ptr, 
        out.stride(0), out.stride(1), out.stride(2),
        cos_sums_ptr, 
        cos_sums.stride(0), cos_sums.stride(1), cos_sums.stride(2),
        sin_sums_ptr, 
        sin_sums.stride(0), sin_sums.stride(1), sin_sums.stride(2),
        tot_Z, tot_N, d_model, magnitude_type, dropout_p, rng_seed,
        stream
    );

    // optional: synchronize the device
    // cudaDeviceSynchronize();
}

PYBIND11_MODULE(wf_embedding_kernel, m) {
    m.def("forward", &wf_embedding_forward, "Wavefunction Embedding Forward Method");
}