#!/bin/bash
MODEL_NAME="$1"
DEVICE="$2"


python benchmark_llm.py \
    --model $MODEL_NAME \
    --device $DEVICE \
    --config $DEVICE/config.yaml \
    --proxy true \
    --dry true \
    --input 128,256,1024,2048,4096 \
    --output 128 \
    --concurrent 256,512,1024