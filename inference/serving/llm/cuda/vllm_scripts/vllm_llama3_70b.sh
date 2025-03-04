#!/bin/bash
export HF_ENDPOINT=https://hf-mirror.com
export VLLM_WORKER_MULTIPROC_METHOD=spawn

python -O -u -m vllm.entrypoints.openai.api_server \
    --host=127.0.0.1 \
    --port=8000 \
    --model=/models/Meta-Llama-3-70B-Instruct \
    --max-model-len=16384 \
    --tokenizer=hf-internal-testing/llama-tokenizer \
    --api-key=openai \
    --trust-remote-code \
    --tensor-parallel-size=8 \
    --gpu-memory-utilization=0.95 \
    --dtype float16 \
    --served-model-name meta-llama-3-70b \
    --device="cuda"