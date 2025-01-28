from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="attn_fwd_kernel",
    ext_modules=[
        CUDAExtension(
            name="attn_fwd_kernel",
            sources=["attn_fwd_if.cpp", "attn_fwd_kernel.cu"]
        )
    ],
    cmdclass={"build_ext": BuildExtension}
)