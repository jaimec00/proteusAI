
#include <torch/extension.h>  // for pytorch c++ extensions
#include <ATen/cuda/CUDAContext.h>  // for accessing cuda streams
#include <cuda_runtime.h>  // for cuda runtime functions

// declare the cuda kernel implemented in wf_embedding_kernel.cu
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
);

void _attn_fwd(
    torch::Tensor Q,
    torch::Tensor K,
    torch::Tensor V,
    torch::Tensor coords, 
    torch::Tensor spreads,
    torch::Tensor rng_seed,

    torch::Tensor L,
    torch::Tensor O,

    int softmax_scale, int dropout,
    int batch, int N, int nheads, int d_k
    
) {

    // validate inputs
    TORCH_CHECK(Q.device().is_cuda(), "Q must be a CUDA tensor");
    TORCH_CHECK(K.device().is_cuda(), "K must be a CUDA tensor");
    TORCH_CHECK(V.device().is_cuda(), "V must be a CUDA tensor");
    TORCH_CHECK(coords.device().is_cuda(), "coords must be a CUDA tensor");
    TORCH_CHECK(spreads.device().is_cuda(), "spreads must be a CUDA tensor");
    TORCH_CHECK(rng_seed.device().is_cuda(), "rng_seed must be a CUDA tensor");
    TORCH_CHECK(L.device().is_cuda(), "L must be a CUDA tensor");
    TORCH_CHECK(out.device().is_cuda(), "out must be a CUDA tensor");

    TORCH_CHECK(Q.dtype() == torch::kFloat32, "Q must be a CUDA tensor");
    TORCH_CHECK(K.dtype() == torch::kFloat32, "K must be a CUDA tensor");
    TORCH_CHECK(V.dtype() == torch::kFloat32, "V must be a CUDA tensor");
    TORCH_CHECK(coords.dtype() == torch::kFloat32, "coords must be a CUDA tensor");
    TORCH_CHECK(spreads.dtype() == torch::kFloat32, "spreads must be a CUDA tensor");
    TORCH_CHECK(rng_seed.dtype() == torch::kFloat32, "rng_seed must be a CUDA tensor");
    TORCH_CHECK(L.dtype() == torch::kFloat32, "L must be a CUDA tensor");
    TORCH_CHECK(out.dtype() == torch::kFloat32, "out must be a CUDA tensor");

    // get tensor sizes 
    int tot_Z = coords.size(0); // batch size
    int tot_N = coords.size(2); // sequence size (transposed by my autograd function to be Z x 3 x N for coalesced memory accesses)
    int d_k = Q.size(1)
    int nheads*Q.size(3); // feature size (num_wn == d_model//2)

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
    attn_fwd_kernel(    
        Q_ptr, int stride_Q_Z, int stride_Q_H, int stride_Q_N, int stride_Q_D,
        K_ptr, int stride_K_Z, int stride_K_H, int stride_K_N, int stride_K_D,
        V_ptr, int stride_V_Z, int stride_V_H, int stride_V_N, int stride_V_D,
        coords_ptr, int stride_coords_Z, int stride_coords_S, int stride_coords_N,
        spreads_ptr, int stride_spreads_H,
        rng_seed_ptr, int stride_rng_seed_Z, int stride_rng_seed_H,

        L, int stride_L_Z, int stride_L_H, int stride_L_N,
        O, int stride_O_Z, int stride_O_H, int stride_O_N, int stride_O_D,

        softmax_scale, dropout,
        batch, N, nheads, d_k

        stream
    );

}

PYBIND11_MODULE(attn_fwd_kernel, m) {
    m.def("fwd", &_attn_fwd, "Geometric Attention Forward Method");
}