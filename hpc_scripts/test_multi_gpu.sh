#!/bin/bash

## request 1 node, two processes per node, and two gpus per node

#BSUB -n 2
#BSUB -R "span[ptile=2]"
#BSUB -gpu "num=2:mode=shared:mps=yes"
##BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -q gpu

#BSUB -W 00:10
#BSUB -R "rusage[mem=16GB]"

##BSUB -R "select[h100]"
##BSUB -R "select[a100 || h100]"
#BSUB -R "select[a10 || a30]"
##BSUB -R "select[l40 || a100 || h100]"

#BSUB -o out.%J
#BSUB -e err.%J
#BSUB -J protAI_train

source ~/.bashrc
source /usr/share/Modules/init/bash
module load cuda/12.1
conda activate protAI_env
nvidia-smi

export PYTHONPATH="/home/hjc2538/ondemand/data/sys/myjobs/projects/proteusAI/"

torchrun --nproc_per_node=2 other/test_functions.py
