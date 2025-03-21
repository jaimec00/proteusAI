
#include <ATen/cuda/CUDAContext.h>  // for accessing cuda streams
#include <torch/extension.h>  // for pytorch c++ extensions
#include <cuda_runtime.h>  // for cuda runtime functions
#include <cstdint>


// declare the cuda kernel implemented in wf_embedding_kernel.cu
void wf_embedding_kernel_forward(
    float* coordsA, int stride_coordsA_Z, int stride_coordsA_S, int stride_coordsA_N,
    float* coordsB, int stride_coordsB_Z, int stride_coordsB_S, int stride_coordsB_N,
    int16_t* aa_labels, int stride_aa_labels_Z, int stride_aa_labels_N,
    const float* aa_magnitudes, int stride_aa_magnitudes_K, int stride_aa_magnitudes_A,
    const float* wavenumbers, int stride_wavenumbers_K, 

    float* out, int stride_out_Z, int stride_out_N, int stride_out_D,

    int tot_Z, int tot_N, int d_model, int tot_AA,
    float dropout_p, uint32_t rng_seed,
    cudaStream_t stream
);

void wf_embedding_forward(
    torch::Tensor coordsA, torch::Tensor coordsB, 
    torch::Tensor aa_labels, torch::Tensor aa_magnitudes, 
    torch::Tensor wavenumbers, 
    torch::Tensor out,
    float dropout_p, uint32_t rng_seed
) {


    // validate inputs
    TORCH_CHECK(coordsA.device().is_cuda(), "coordsA must be a CUDA tensor");
    TORCH_CHECK(coordsB.device().is_cuda(), "coordsB must be a CUDA tensor");
    TORCH_CHECK(aa_labels.device().is_cuda(), "aa_labels must be a CUDA tensor");
    TORCH_CHECK(aa_magnitudes.device().is_cuda(), "aa_magnitudes must be a CUDA tensor");
    TORCH_CHECK(wavenumbers.device().is_cuda(), "wavenumbers must be a CUDA tensor");
    TORCH_CHECK(out.device().is_cuda(), "out must be a CUDA tensor");

    TORCH_CHECK(coordsA.dtype() == torch::kFloat32, "coordsA must be of type float32");
    TORCH_CHECK(coordsB.dtype() == torch::kFloat32, "coordsB must be of type float32");
    TORCH_CHECK(aa_labels.dtype() == torch::kInt16, "aa_labels must be of type int16");
    TORCH_CHECK(aa_magnitudes.dtype() == torch::kFloat32, "aa_magnitudes must be of type float16");
    TORCH_CHECK(wavenumbers.dtype() == torch::kFloat32, "wavenumbers must be of type float32");
    TORCH_CHECK(out.dtype() == torch::kFloat32, "out must be of type float32");

    // get tensor sizes 
    int tot_Z = coordsA.size(0); // batch size
    int tot_N = coordsA.size(2); // sequence size (transposed by my autograd function to be Z x 3 x N for coalesced memory accesses)
    int d_model = wavenumbers.size(0)*2; // feature size (num_wn == d_model//2)
    int tot_AA = aa_magnitudes.size(1);

    TORCH_CHECK(out.size(0) == tot_Z, "out batch size mismatch");
    TORCH_CHECK(out.size(1) == tot_N, "out sequence size mismatch");
    TORCH_CHECK(out.size(2) == d_model, "out d_model size mismatch");

    // get raw pointers
    float* coordsA_ptr = reinterpret_cast<float*>(coordsA.data_ptr<float>());
    float* coordsB_ptr = reinterpret_cast<float*>(coordsB.data_ptr<float>());
    const float* wavenumbers_ptr = reinterpret_cast<const float*>(wavenumbers.data_ptr<float>());
    int16_t* aa_labels_ptr = aa_labels.data_ptr<int16_t>();
    const float* aa_magnitudes_ptr = reinterpret_cast<const float*>(aa_magnitudes.data_ptr<float>());
    float* out_ptr = reinterpret_cast<float*>(out.data_ptr<float>());

    // launch the cuda kernel
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();  // get pytorch's current cuda stream
    // cast strides to doubel to prevent indexing errors
    wf_embedding_kernel_forward(    
        coordsA_ptr, 
        coordsA.stride(0), coordsA.stride(1), coordsA.stride(2),
        coordsB_ptr, 
        coordsB.stride(0), coordsB.stride(1), coordsB.stride(2),
        aa_labels_ptr, 
        aa_labels.stride(0), aa_labels.stride(1),
        aa_magnitudes_ptr, 
        aa_magnitudes.stride(0), aa_magnitudes.stride(1),
        wavenumbers_ptr, 
        wavenumbers.stride(0),
        out_ptr, 
        out.stride(0), out.stride(1), out.stride(2),
        tot_Z, tot_N, d_model, tot_AA,
        dropout_p, rng_seed,
        stream
    );

}

PYBIND11_MODULE(wf_embedding_kernel, m) {
    m.def("forward", &wf_embedding_forward, "Wavefunction Embedding Forward Method");
}