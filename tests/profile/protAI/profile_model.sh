#!/bin/bash

set -e

export PYTHONPATH="/home/ubuntu/proteusAI-tx/proteusAI"
# source /home/ubuntu/miniconda3/bin/activate
# conda init --all

conda activate protAI_env
out="tests/profile/protAI/protAI_out"
/usr/local/NVIDIA-Nsight-Compute-2025.1/ncu -o $out python tests/protAI.py
chown ubuntu "${out}.ncu-rep"
