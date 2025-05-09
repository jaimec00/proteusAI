#!/bin/bash

#BSUB -n 1
#BSUB -W 5:00:00
##BSUB -R "rusage[mem=16GB]"
##BSUB -R span[hosts=1]
##BSUB -R select[stc]
##BSUB -x
##BSUB -q single_chassis
#BSUB -o out.%J
#BSUB -e err.%J
#BSUB -J protAI_cleanpdbs

source ~/.bashrc
conda activate protAI_env
python -u utils/train_utils/data_utils.py
