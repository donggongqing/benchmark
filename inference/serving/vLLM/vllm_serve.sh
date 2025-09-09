#!/bin/bash
# Parameter parsing
if [ $# -ne 1 ]; then
    echo "Please provide one parameter as MODEL_DIR"
    exit 1
fi
MODEL_DIR="$1"

# Validate required parameters
if [ -z "${MODEL_DIR}" ]; then
    echo "The --model_dir parameter must be specified"
    exit 1
fi

# Extract model name
MODEL_NAME=$(basename "${MODEL_DIR}" | tr '[:upper:]' '[:lower:]')

# Start the vLLM service
vllm serve "${MODEL_DIR}" \
    --trust-remote-code \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.9 \
    --dtype bfloat16 \
    --device cuda