#!/bin/bash
MODEL_NAME="$1"
MODEL_ALIAS="$2"

python convert_data.py \
    --model $MODEL_NAME \
    --model-alias $MODEL_ALIAS \
    --data-type FP16 \
    --driver 'NVIDIA-Linux-x86_64' \
    --driver-version '535.161.08' \
    --backend cuda \
    --backend-version 2.8.0 \
    --engine cuda \
    --engine-version 2.8.0 \
    --serving vllm \
    --serving-version 0.7.3 \
    --gpu 'A100' \
    --gpu-num 1 \
    --tp 1 \
    --base-dir result_outputs \
    --source 'in-house_benchmark'