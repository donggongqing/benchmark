#!/bin/bash

# Parameter parsing
if [ $# -ne 2 ]; then
    echo "Usage: $0 <MODEL_DIR> <TP>"
    echo "TP must be an integer 1-8"
    exit 1
fi

MODEL_DIR="$1"
TP="$2"

# Validate model dir
if [ -z "${MODEL_DIR}" ]; then
    echo "The <MODEL_DIR> parameter must be specified"
    exit 1
fi

# Validate TP (must be integer 1-8)
if ! [[ "${TP}" =~ ^[1-8]$ ]]; then
    echo "Error: TP must be an integer between 1 and 8"
    exit 1
fi

# Extract model name
MODEL_NAME=$(basename "${MODEL_DIR}" | tr '[:upper:]' '[:lower:]')

# Start the vLLM service
vllm serve "${MODEL_DIR}" \
    --trust-remote-code \
    --tensor-parallel-size ${TP} \
    --gpu-memory-utilization 0.9 \
    --served-model-name "${MODEL_NAME}" 
