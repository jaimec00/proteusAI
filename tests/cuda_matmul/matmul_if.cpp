
#include <torch/extension.h>  // for pytorch c++ extensions
#include <ATen/cuda/CUDAContext.h>  // for accessing cuda streams
#include <cuda_runtime.h>  // for cuda runtime functions
#include <cstdint> // for unsigned 32 bit int


#include <cute/tensor.hpp>

using cute::half_t;  // CUTE's FP16 type

// declare the cuda kernel implemented in wf_embedding_kernel.cu
void main(
    const half_t* A_ptr,
    const half_t* B_ptr,
    const half_t* C_ptr,
    half_t* D_ptr,

    cudaStream_t stream
);

void matmul_kernel(
    torch::Tensor A,
    torch::Tensor B,
    torch::Tensor C,
    torch::Tensor D,
    
) {

    // validate inputs
    TORCH_CHECK(A.device().is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B.device().is_cuda(), "B must be a CUDA tensor");
    TORCH_CHECK(C.device().is_cuda(), "C must be a CUDA tensor");
    TORCH_CHECK(D.device().is_cuda(), "D must be a CUDA tensor");

    TORCH_CHECK(A.dtype() == torch::kFloat16, "A must be a CUDA tensor");
    TORCH_CHECK(B.dtype() == torch::kFloat16, "B must be a CUDA tensor");
    TORCH_CHECK(C.dtype() == torch::kFloat16, "C must be a CUDA tensor");
    TORCH_CHECK(D.dtype() == torch::kFloat16, "D must be a CUDA tensor");


    const half_t* A_ptr = reinterpret_cast<const half_t*>(A.data_ptr<half_t>());
    const half_t* B_ptr = reinterpret_cast<const half_t*>(B.data_ptr<half_t>());
    const half_t* C_ptr = reinterpret_cast<const half_t*>(C.data_ptr<half_t>());
    half_t* D_ptr = reinterpret_cast<half_t*>(D.data_ptr<half_t>());

    // launch the cuda kernel
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();  // get pytorch's current cuda stream
    // cast strides to doubel to prevent indexing errors
    mamtul_kernel(    
        A_ptr, B_ptr, C_ptr, D_ptr,

        stream
    );

}

PYBIND11_MODULE(matmul_kernel, m) {
    m.def("fwd", &_matmul_kernel, "Matmul Test");
}