#!/bin/bash

#BSUB -n 1
#BSUB -W 10:00
#BSUB -R "rusage[mem=128GB]"
#BSUB -q gpu
##BSUB -R "select[a100]"
#BSUB -R "select[l40 || a100]"
##BSUB -R "select[a100 || h100]"
##BSUB -R "select[a10 || a30]"
##BSUB -R "select[l40 || a100 || h100]"
#BSUB -gpu "num=1:mode=shared:mps=yes"
##BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -o out.%J
#BSUB -e err.%J
#BSUB -J protAI_train

source ~/.bashrc
source /usr/share/Modules/init/bash
module load cuda/12.1
conda activate protAI_env

# define triton cache dir so dont run out of space in home
export TRITON_HOME="/share/wangyy/hjc2538/proteusAI"
export TRITON_CACHE_DIR="/share/wangyy/hjc2538/proteusAI/.triton/cache"

# print autotuning configs, to make sure it isnt autotuning each time, and to see for reference
export TRITON_PRINT_AUTOTUNING="1"
# export ATTN_AUTOTUNE="1"

export CUDA_VISIBLE_DEVICES=0
export CUDA_LAUNCH_BLOCKING=1
python -u learn_seqs.py --config config/config.yml
