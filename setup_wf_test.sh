#!/bin/bash

export PYTHONPATH="/storage/cms/wangyy_lab/hjc2538/proteusAI"
# module load cuda/12.5
spack load ninja
# module load nvhpc-hpcx-cuda12/24.11
conda activate protAI_env

# force g++
export CXX=$(which g++)
export CC=$(which g++)
export NVCC_WRAPPER_DEFAULT_COMPILER=$(which g++)
export CUDAHOSTCXX=$(which g++)

export CUDA_HOME=$CONDA_PREFIX
export TORCH_NVCC_EXECUTABLE=$CUDA_HOME/bin/nvcc
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
