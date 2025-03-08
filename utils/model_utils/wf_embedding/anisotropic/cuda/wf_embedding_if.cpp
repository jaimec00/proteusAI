
#include <torch/extension.h>  // for pytorch c++ extensions
#include <ATen/cuda/CUDAContext.h>  // for accessing cuda streams
#include <cuda_runtime.h>  // for cuda runtime functions
#include <cstdint>

// declare the cuda kernel implemented in wf_embedding_kernel.cu
void wf_embedding_kernel_forward(
    const float* coordsA, int stride_coordsA_Z, int stride_coordsA_S, int stride_coordsA_N,
    const float* coordsB, int stride_coordsB_Z, int stride_coordsB_S, int stride_coordsB_N,
    const float* wavenumbers, int stride_wavenumbers_K, 

    float* out, int stride_out_Z, int stride_out_N, int stride_out_D,
    float* d_imag, int stride_d_imag_Z, int stride_d_imag_N, int stride_d_imag_K,
    float* d_real, int stride_d_real_Z, int stride_d_real_N, int stride_d_real_K,

    int tot_Z, int tot_N, int d_model, int magnitude_type, float dropout_p, uint32_t rng_seed,
    cudaStream_t stream
);

void wf_embedding_forward(
    torch::Tensor coordsA, torch::Tensor coordsB, torch::Tensor wavenumbers, 
    torch::Tensor out, 
    torch::Tensor d_imag, torch::Tensor d_real,
    int magnitude_type, float dropout_p, uint32_t rng_seed
) {


    // validate inputs
    TORCH_CHECK(coordsA.device().is_cuda(), "coordsA must be a CUDA tensor");
    TORCH_CHECK(coordsB.device().is_cuda(), "coordsB must be a CUDA tensor");
    TORCH_CHECK(wavenumbers.device().is_cuda(), "wavenumbers must be a CUDA tensor");
    TORCH_CHECK(out.device().is_cuda(), "out must be a CUDA tensor");
    TORCH_CHECK(d_imag.device().is_cuda(), "d_imag must be a CUDA tensor");
    TORCH_CHECK(d_real.device().is_cuda(), "d_real must be a CUDA tensor");

    TORCH_CHECK(coordsA.dtype() == torch::kFloat32, "coordsA must be of type float32");
    TORCH_CHECK(coordsB.dtype() == torch::kFloat32, "coordsB must be of type float32");
    TORCH_CHECK(wavenumbers.dtype() == torch::kFloat32, "wavenumbers must be of type float32");
    TORCH_CHECK(out.dtype() == torch::kFloat32, "out must be of type float32");
    TORCH_CHECK(d_imag.dtype() == torch::kFloat32, "out must be of type float32");
    TORCH_CHECK(d_real.dtype() == torch::kFloat32, "out must be of type float32");

    // get tensor sizes 
    int tot_Z = coordsA.size(0); // batch size
    int tot_N = coordsA.size(2); // sequence size (transposed by my autograd function to be Z x 3 x N for coalesced memory accesses)
    int d_model = wavenumbers.size(0)*2; // feature size (num_wn == d_model//2)

    TORCH_CHECK(out.size(0) == tot_Z, "out batch size mismatch");
    TORCH_CHECK(out.size(1) == tot_N, "out sequence size mismatch");
    TORCH_CHECK(out.size(2) == d_model, "out d_model size mismatch");

    // get raw pointers
    const float* coordsA_ptr = reinterpret_cast<const float*>(coordsA.data_ptr<float>());
    const float* coordsB_ptr = reinterpret_cast<const float*>(coordsB.data_ptr<float>());
    const float* wavenumbers_ptr = reinterpret_cast<const float*>(wavenumbers.data_ptr<float>());
    float* out_ptr = reinterpret_cast<float*>(out.data_ptr<float>());
    float* d_imag_ptr = reinterpret_cast<float*>(d_imag.data_ptr<float>());
    float* d_real_ptr = reinterpret_cast<float*>(d_real.data_ptr<float>());

    // launch the cuda kernel
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();  // get pytorch's current cuda stream
    // cast strides to doubel to prevent indexing errors
    wf_embedding_kernel_forward(    
        coordsA_ptr, 
        coordsA.stride(0), coordsA.stride(1), coordsA.stride(2),
        coordsB_ptr, 
        coordsB.stride(0), coordsB.stride(1), coordsB.stride(2),
        wavenumbers_ptr, 
        wavenumbers.stride(0),
        out_ptr, 
        out.stride(0), out.stride(1), out.stride(2),
        d_imag_ptr, 
        d_imag.stride(0), d_imag.stride(1), d_imag.stride(2),
        d_real_ptr, 
        d_real.stride(0), d_real.stride(1), d_real.stride(2),
        tot_Z, tot_N, d_model, magnitude_type, dropout_p, rng_seed,
        stream
    );

}

PYBIND11_MODULE(wf_embedding_kernel, m) {
    m.def("forward", &wf_embedding_forward, "Wavefunction Embedding Forward Method");
}