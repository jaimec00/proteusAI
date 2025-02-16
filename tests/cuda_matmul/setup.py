from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="matmul_kernel",
    ext_modules=[
        CUDAExtension(
            name="matmul_kernel",
            sources=["matmul_if.cpp", "matmul_kernel.cu"]
        )
    ],
    cmdclass={"build_ext": BuildExtension}
)