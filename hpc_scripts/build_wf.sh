#!/bin/bash
#SBATCH --job-name=build_wf
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --nodes=1-1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=build_wf.out
#SBATCH --error=build_wf.err


export PYTHONPATH="/storage/cms/wangyy_lab/hjc2538/proteusAI"
module load cuda/12.5
spack load ninja
module load nvhpc-hpcx-cuda12/24.11
conda activate protAI_env

nvidia-smi

# force g++
export CXX=$(which g++)
export CC=$(which g++)
export NVCC_WRAPPER_DEFAULT_COMPILER=$(which g++)
export CUDAHOSTCXX=$(which g++)

cd utils/model_utils/wf_embedding/cuda
python3 setup.py build_ext --inplace --verbose

