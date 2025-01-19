#!/bin/bash

#BSUB -n 1
#BSUB -W 5:00
##BSUB -R "rusage[mem=16GB]"
##BSUB -R span[hosts=1]
##BSUB -R select[stc]
##BSUB -x
##BSUB -q single_chassis
#BSUB -o out.%J
#BSUB -e err.%J
#BSUB -J protAI_cleanpdbs

source ~/.bashrc
source /usr/share/Modules/init/bash
module load cuda/12.1
conda activate protAI_env

# so can import from base dir
export PYTHONPATH="/home/hjc2538/ondemand/data/sys/myjobs/projects/proteusAI"

python -u utils/data_utils.py
