#!/bin/bash

set -e 
export PYTHONPATH="/home/ubuntu/proteusAI-tx/proteusAI"
source /home/ubuntu/miniconda3/bin/activate
conda init --all

conda activate protAI_env
out="tests/profile/attn/attn_out2"
export ATTN_AUTOTUNE="0"
/usr/local/NVIDIA-Nsight-Compute-2025.1/ncu -o $out python tests/geo_attn.py
chown ubuntu "${out}.ncu-rep"
