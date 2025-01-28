#!/bin/bash
#SBATCH --job-name=protAI_train
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --nodes=1-1
#SBATCH --cpus-per-task=1
#SBATCH --threads-per-core=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=00:30:00
 #SBATCH --output=protAI_train.out
#SBATCH --error=protAI_train.err


export PYTHONPATH="/storage/cms/wangyy_lab/hjc2538/proteusAI"

source ~/.bashrc
# source /usr/share/Modules/init/bash
module load cuda/12.5
source "$PYTHONPATH/protAI_env"
nvidia-smi

# define triton cache dir so dont run out of space in home
export TRITON_HOME="/scratch/hjc2538/proteusAI"
export TRITON_CACHE_DIR="/scratch/hjc2538/proteusAI/.triton/cache"

python -u learn_seqs.py --config config/config.yml