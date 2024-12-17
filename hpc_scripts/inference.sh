#!/bin/bash

#BSUB -n 1
#BSUB -W 02:00
#BSUB -R "rusage[mem=32GB]"
#BSUB -q gpu
##BSUB -R "select[h100]"
##BSUB -R "select[a100 || h100]"
#BSUB -R "select[a10 || a30]"
##BSUB -R "select[l40]"
##BSUB -gpu "num=1:mode=shared:mps=yes"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -o out.%J
#BSUB -e err.%J
#BSUB -J protAI_inference

source ~/.bashrc
source /usr/share/Modules/init/bash
module load cuda/12.1
conda activate protAI_env
nvidia-smi

#export CUDA_VISIBLE_DEVICES=0
export CUDA_LAUNCH_BLOCKING=1
python -u test_functions.py
