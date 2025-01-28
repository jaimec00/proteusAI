#!/bin/bash

export PYTHONPATH="/home/ubuntu/proteusAI/proteusAI"
out="tests/profile/protAI/protAI_out6"
/usr/lib/nsight-compute/ncu -o $out python tests/protAI.py
chown ubuntu "${out}.ncu-rep"
