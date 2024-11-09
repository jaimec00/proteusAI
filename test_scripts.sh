#!/bin/bash

#BSUB -n 1
#BSUB -W 00:30
#BSUB -R "rusage[mem=16GB]"
#BSUB -q gpu
##BSUB -R "select[a10 || a30]"
#BSUB -gpu "num=1:mode=shared:mps=yes"
##BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -o out.%J
#BSUB -e err.%J
#BSUB -J protAI_tests

source ~/.bashrc
source /usr/share/Modules/init/bash
module load cuda/12.1
conda activate protAI_env
nvidia-smi

conda activate protAI_env

export CUDA_VISIBLE_DEVICES=0

python -u get_pmpnn_pdbs.py
