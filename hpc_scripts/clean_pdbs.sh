#!/bin/bash

#BSUB -n 3
#BSUB -W 5:00
#BSUB -R "rusage[mem=64GB]"
#BSUB -q gpu
#BSUB -R "span[ptile=3]"
#BSUB -R "select[l40 || a10 || a30 || a100 || h100]"
#BSUB -gpu "num=3:mode=exclusive_process"
#BSUB -o out.%J
#BSUB -e err.%J
#BSUB -J protAI_cleanpdbs

source ~/.bashrc
source /usr/share/Modules/init/bash
module load cuda/12.1
conda activate protAI_env
nvidia-smi

# define triton cache dir so dont run out of space in home
export TRITON_HOME="/share/wangyy/hjc2538/proteusAI"
export TRITON_CACHE_DIR="/share/wangyy/hjc2538/proteusAI/.triton/cache"

# so can import from base dir
export PYTHONPATH="/home/hjc2538/ondemand/data/sys/myjobs/projects/proteusAI"

python -u utils/data_utils.py
