#!/bin/bash

export PYTHONPATH="/home/ubuntu/proteusAI/proteusAI"
out="tests/profile/attn/attn_out"
/usr/lib/nsight-compute/ncu -o $out python tests/geo_attn.py
chown ubuntu "${out}.ncu-rep"
