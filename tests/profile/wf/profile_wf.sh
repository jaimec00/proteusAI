#!/bin/bash

# export PYTHONPATH="/home/ubuntu/proteusAI/proteusAI"
out="tests/profile/wf/wf_out"
# /usr/lib/nsight-compute/ncu
ncu -o $out python tests/wf_embed_aniso_learnAA.py
# chown ubuntu "${out}.ncu-rep"
# chown ubuntu "${out}.ncu-rep"
