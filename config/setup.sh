#!/bin/bash

# get the config dir
CONFIG_DIR=$(cd $(dirname ${BASH_SOURCE}) && pwd)
PROTAI_DIR=$(dirname $CONFIG_DIR)

# check if conda is available, if not then setup conda
source "$CONFIG_DIR/setup_conda.sh"

# if it does, create the environment
PROTAI_ENV="$CONFIG_DIR/protAI_env.yml"
conda env create -f $PROTAI_ENV

# activate the environment
conda activate protAI_env

# check where the env is located
ACT_SCRIPT="$CONDA_PREFIX/etc/conda/activate.d/conda_activation_script.sh"
cat << EOF > $ACT_SCRIPT

# setup env for proteus ai
export PYTHONPATH="\$PYTHONPATH;$PROTAI_DIR"

# cuda config
export CUDA_HOME=\$CONDA_PREFIX
export TORCH_NVCC_EXECUTABLE=\$CUDA_HOME/bin/nvcc
export PATH=\$CUDA_HOME/bin:\$PATH
export LD_LIBRARY_PATH=\$CUDA_HOME/lib64:\$LD_LIBRARY_PATH

# for triton compilation
# pytorch was built w/ g++, and triton dynamically compiles the geo attn kernels, so ensure it uses g++
# but only for cpp files, set gcc for c files
export CXX=\$(which g++)
export CC=\$(which gcc)
export NVCC_WRAPPER_DEFAULT_COMPILER=\$(which g++)
export CUDAHOSTCXX=\$(which g++)

# define triton cache dir so dont run out of space in home
export TRITON_HOME="/scratch/hjc2538/projects/proteusAI" # leaving this for now
export TRITON_CACHE_DIR="\$TRITON_HOME/.triton/cache"

EOF 

# deactivate and activate so the activation script works
conda deactivate
conda activate protAI_env
