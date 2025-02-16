
#include <torch/extension.h>  // for pytorch c++ extensions
#include <ATen/cuda/CUDAContext.h>  // for accessing cuda streams
#include <cuda_runtime.h>  // for cuda runtime functions
#include <cstdint> // for unsigned 32 bit int

// declare the cuda kernel implemented in wf_embedding_kernel.cu
void attn_fwd(
    const float* Q_ptr, int stride_Q_Z, int stride_Q_H, int stride_Q_N, int stride_Q_D,
    const float* K_ptr, int stride_K_Z, int stride_K_H, int stride_K_N, int stride_K_D,
    const float* V_ptr, int stride_V_Z, int stride_V_H, int stride_V_N, int stride_V_D,
    const float* coords_ptr, int stride_coords_Z, int stride_coords_S, int stride_coords_N,
    const float* spreads_ptr, int stride_spreads_H,

    float* L, int stride_L_Z, int stride_L_H, int stride_L_N,
    float* O, int stride_O_Z, int stride_O_H, int stride_O_N, int stride_O_D,

    float softmax_scale, float dropout, uint32_t rng_seed,
    int batch, int N, int nheads, int d_k

    cudaStream_t stream
);

void _attn_fwd(
    torch::Tensor Q,
    torch::Tensor K,
    torch::Tensor V,
    torch::Tensor coords, 
    torch::Tensor spreads,

    torch::Tensor L,
    torch::Tensor O,

    int softmax_scale, int dropout, uint32_t rng_seed
    
) {

    // validate inputs
    TORCH_CHECK(Q.device().is_cuda(), "Q must be a CUDA tensor");
    TORCH_CHECK(K.device().is_cuda(), "K must be a CUDA tensor");
    TORCH_CHECK(V.device().is_cuda(), "V must be a CUDA tensor");
    TORCH_CHECK(coords.device().is_cuda(), "coords must be a CUDA tensor");
    TORCH_CHECK(spreads.device().is_cuda(), "spreads must be a CUDA tensor");
    TORCH_CHECK(L.device().is_cuda(), "L must be a CUDA tensor");
    TORCH_CHECK(O.device().is_cuda(), "O must be a CUDA tensor");

    TORCH_CHECK(Q.dtype() == torch::kFloat16, "Q must be a CUDA tensor");
    TORCH_CHECK(K.dtype() == torch::kFloat16, "K must be a CUDA tensor");
    TORCH_CHECK(V.dtype() == torch::kFloat16, "V must be a CUDA tensor");
    TORCH_CHECK(coords.dtype() == torch::kFloat32, "coords must be a CUDA tensor");
    TORCH_CHECK(spreads.dtype() == torch::kFloat32, "spreads must be a CUDA tensor");
    TORCH_CHECK(L.dtype() == torch::kFloat32, "L must be a CUDA tensor");
    TORCH_CHECK(O.dtype() == torch::kFloat16, "O must be a CUDA tensor");

    // get tensor sizes 
    int tot_Z = Q.size(0); // batch size
    int nheads = Q.size(1); // heads
    int tot_N = Q.size(2); // seq size
    int d_k = Q.size(3) // d_k
    int d_model = nheads*d_k; // feature size (d_model == d_k*nheads)

    TORCH_CHECK(O.size(0) == tot_Z, "O batch size mismatch");
    TORCH_CHECK(O.size(1) == tot_N, "O sequence size mismatch");
    TORCH_CHECK(O.size(2) == d_model, "O d_model size mismatch");

    // get raw pointers
    const __half* Q_ptr = reinterpret_cast<const __half*>(Q.data_ptr<__half>());
    const __half* K_ptr = reinterpret_cast<const __half*>(K.data_ptr<__half>());
    const __half* V_ptr = reinterpret_cast<const __half*>(V.data_ptr<__half>());
    const float* coords_ptr = reinterpret_cast<const float*>(coords.data_ptr<float>());
    const float* spreads_ptr = reinterpret_cast<const float*>(spreads.data_ptr<float>());
    float* L_ptr = reinterpret_cast<float*>(L.data_ptr<float>());
    __half* O_ptr = reinterpret_cast<__half*>(O.data_ptr<__half>());

    // launch the cuda kernel
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();  // get pytorch's current cuda stream
    // cast strides to doubel to prevent indexing errors
    attn_fwd_kernel(    
        Q_ptr, int stride_Q_Z, int stride_Q_H, int stride_Q_N, int stride_Q_D,
        K_ptr, int stride_K_Z, int stride_K_H, int stride_K_N, int stride_K_D,
        V_ptr, int stride_V_Z, int stride_V_H, int stride_V_N, int stride_V_D,
        coords_ptr, int stride_coords_Z, int stride_coords_S, int stride_coords_N,
        spreads_ptr, int stride_spreads_H,

        L_ptr, int stride_L_Z, int stride_L_H, int stride_L_N,
        O_ptr, int stride_O_Z, int stride_O_H, int stride_O_N, int stride_O_D,

        softmax_scale, dropout, rng_seed,
        batch, N, nheads, d_k

        stream
    );

}

PYBIND11_MODULE(attn_fwd_kernel, m) {
    m.def("fwd", &_attn_fwd, "Geometric Attention Forward Method");
}