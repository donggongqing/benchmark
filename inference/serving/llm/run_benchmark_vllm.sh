#!/bin/bash
MODEL_NAME="$1"
DEVICE="$2"


python benchmark_onlyvllm.py \
    --model $MODEL_NAME \
    --device $DEVICE \
    --dry  \  #不需要部署服务加--dry
    --input 128,256 \
    --output 128 \
    --concurrent 256