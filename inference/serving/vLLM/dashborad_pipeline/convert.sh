#!/bin/bash
# convert.sh vLLM benchmark 转换脚本
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <input_csv_path> <model_alias>"
    exit 1
fi

INPUT_CSV="$1"
MODEL_ALIAS="$2"

python3 convert_vLLM_bench.py \
    --input "$INPUT_CSV" \
    --model-alias "$MODEL_ALIAS" \
    --tp 4 \
    --dp 1 \
    --pp 1 \
    --ep 1 \
    --data-type bf16 \
    --gpu "S5000" \
    --gpu-num 4 \
    --driver "msua_driver" \
    --driver-version "3.3.1-server" \
    --backend musa \
    --backend-version 4.3.1 \
    --engine musa \
    --engine-version 4.3.1 \
    --serving vllm \
    --serving-version 0.9.3 \
    --source "vllm"
