#!/bin/bash

export PYTHONPATH="/home/ubuntu/proteusAI/proteusAI"
out="tests/profile/wf/wf_out"
/usr/lib/nsight-compute/ncu -o $out python tests/wf_embed.py
chown ubuntu "${out}.ncu-rep"
