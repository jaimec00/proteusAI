#!/bin/bash
#SBATCH --job-name=pmpnn_train
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=72:00:00
#SBATCH --output=pmpnntrain.out
#SBATCH --error=pmpnntrain.err

source ~/.bashrc
conda activate pmpnn
nvidia-smi
python -u /scratch/hjc2538/software/ProteinMPNN/training/training.py --mixed_precision False --path_for_training_data /scratch/hjc2538/projects/proteusAI/data/multi_chain/raw --num_neighbors 30 --backbone_noise 0.00
