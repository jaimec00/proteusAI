from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="wf_embedding_kernel",
    ext_modules=[
        CUDAExtension(
            name="wf_embedding_kernel",
            sources=["wf_embedding_if.cpp", "wf_embedding_kernel.cu"]
        )
    ],
    cmdclass={"build_ext": BuildExtension}
)