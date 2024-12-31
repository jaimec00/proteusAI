#!/bin/bash

#BSUB -n 1
#BSUB -W 1:00
#BSUB -R "rusage[mem=32GB]"
#BSUB -q gpu
#BSUB -R "select[h100]"
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

export CUDA_VISIBLE_DEVICES=0
export CUDA_LAUNCH_BLOCKING=1
export PYTHONPATH="/home/hjc2538/ondemand/data/sys/myjobs/projects/proteusAI"
python -u tests/gauss_attn.py
