#!/bin/bash


export PYTHONPATH="/home/ubuntu/proteusAI-testing/proteusAI"
out="tests/profile/wf/wf_out"
/usr/local/NVIDIA-Nsight-Compute/ncu -o $out /home/ubuntu/anaconda3/envs/protAI_env/bin/python tests/wf_embed_aniso_learnAA.py
chown ubuntu "${out}.ncu-rep"
