#!/bin/bash

#BSUB -n 1
#BSUB -W 72:00
#BSUB -R "rusage[mem=64GB]"
#BSUB -q gpu
#BSUB -R "select[a10 || a30]"
##BSUB -gpu "num=1:mode=shared:mps=yes"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -o out.%J
#BSUB -e err.%J
#BSUB -J protAI_clean_pdbs

source ~/.bashrc
source /usr/share/Modules/init/bash
module load cuda/12.1
conda activate protAI_env
nvidia-smi

#export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python -u get_pmpnn_pdbs.py
