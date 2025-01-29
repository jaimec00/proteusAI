from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
#import sysconfig
#from torch.utils.cpp_extension import include_paths

#python_include_path = sysconfig.get_paths()["include"]
#all_includes = [python_include_path]+include_paths()

setup(
    name="wf_embedding_kernel",
    ext_modules=[
        CUDAExtension(
            name="wf_embedding_kernel",
            sources=["wf_embedding_if.cpp", "wf_embedding_kernel.cu"]
#            include_dirs=all_includes

	)
    ],
    cmdclass={"build_ext": BuildExtension}
)
