#!/bin/bash

#BSUB -n 1
#BSUB -W 12:00
#BSUB -R "rusage[mem=64GB]"
#BSUB -q gpu
#BSUB -R "select[a10 || a30]"
##BSUB -gpu "num=1:mode=shared:mps=yes"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -o out.%J
#BSUB -e err.%J
#BSUB -J protAI_train_new_model_full_lr2.5e-5_d0.2

source ~/.bashrc
source /usr/share/Modules/init/bash
module load cuda/12.1
conda activate protAI_env
nvidia-smi

export CUDA_VISIBLE_DEVICES=0

out_dir="/gpfs_backup/wangyy_data/protAI/models/tests/new_model_full_lr2.5e-5_d0.2"

python -u learn_seqs.py --input_atoms 0 --d_model 512 --num_heads 8 --hidden_linear_dim 1024 \
                        --train_val 0.75 --val_test 0.5 \
                        --epochs 20 --batch_size 32 \
                        \
                        --learning_step 0.00005 --dropout 0.1 --label_smoothing 0.1 \
                        --num_inputs 4096 \
                        \
                        --out_path "$out_dir" \
                        --loss_plot "loss_vs_epoch.png" \
                        --seq_plot "seq_sim_vs_epoch.png" \
                        --weights_path "model_parameters.pth" \
                        --write_dot False \
                        \
                        --pt_path "/gpfs_backup/wangyy_data/protAI/cath_nr40/filtered/pt" \
                        --use_wf True
