#!/bin/bash
#SBATCH --job-name=protAI_train
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=72:00:00
#SBATCH --output=train.out
#SBATCH --error=train.err

source ~/.bashrc
conda activate protAI_env
nvidia-smi
python -u learn_seqs.py --config config/train.yml
