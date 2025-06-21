#!/bin/bash

# put this file in $CONDA_PREFIX/etc/conda/activate.d dir so that it executes when you activate the conda env. 
# make sure you modify it first to have the correct paths. will make a setup.py script to do this automatically 
# once production ready.

# setup env for proteus ai
export PYTHONPATH="/storage/cms/wangyy_lab/hjc2538/proteusAI"

# cuda config
export CUDA_HOME=$CONDA_PREFIX
export TORCH_NVCC_EXECUTABLE=$CUDA_HOME/bin/nvcc
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# for triton compilation
# pytorch was built w/ g++, and triton dynamically compiles the geo attn kernels, so ensure it uses g++
# but only for cpp files, set gcc for c files
export CXX=$(which g++)
export CC=$(which gcc)
export NVCC_WRAPPER_DEFAULT_COMPILER=$(which g++)
export CUDAHOSTCXX=$(which g++)

# define triton cache dir so dont run out of space in home
export TRITON_HOME="/scratch/hjc2538/projects/proteusAI"
export TRITON_CACHE_DIR="$TRITON_HOME/.triton/cache"

