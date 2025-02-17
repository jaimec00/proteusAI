#!/bin/bash
#SBATCH --job-name=test_geo_attn
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=geo.out
#SBATCH --error=geo.err

export PYTHONPATH="/storage/cms/wangyy_lab/hjc2538/proteusAI"

source ~/.bashrc
module load cuda/12.5
module load nvhpc-hpcx-cuda12/24.11
conda activate protAI_env
nvidia-smi

# define triton cache dir so dont run out of space in home
export TRITON_HOME="/scratch/hjc2538/projects/proteusAI"
export TRITON_CACHE_DIR="/scratch/hjc2538/projects/proteusAI/.triton/cache"

# pytorch was built w/ g++, and triton dynamically compiles cuda kernels, so ensure it uses g++
# but only for cpp files, set gcc for c files
export CXX=$(which g++)
export CC=$(which gcc)
export NVCC_WRAPPER_DEFAULT_COMPILER=$(which g++)
export CUDAHOSTCXX=$(which g++)

python -u tests/geo_attn.py
