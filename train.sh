#!/bin/bash

#BSUB -n 1
#BSUB -W 48:00
#BSUB -R "rusage[mem=32GB]"
#BSUB -q gpu
##BSUB -R "select[a100]"
#BSUB -gpu "num=1:mode=shared:mps=yes"
##BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -o out.%J
#BSUB -e err.%J
#BSUB -J protAI_train_3lay_lr0.0001_sched-plateu0.75_d0.05_n2048_b32_lbl-smooth0.25_goodcrossfeat_nolayernorm

source ~/.bashrc
source /usr/share/Modules/init/bash
module load cuda/12.1
conda activate protAI_env
nvidia-smi

# export CUDA_VISIBLE_DEVICES=0  # Makes only GPU 0 visible to your job

python -u learn_seqs.py --pt_path /gpfs_backup/wangyy_data/protAI/pdbs/pt --out_path "output/3lay_lr0.0001_sched-plateau0.75_d0.01_n2048_b32_lbl-smooth0.25_goodcrossfeat_nolayernorm"
