#!/bin/bash
export HF_ENDPOINT=https://hf-mirror.com
export VLLM_WORKER_MULTIPROC_METHOD=spawn
python -O -u -m vllm.entrypoints.openai.api_server \
    --host=127.0.0.1 \
    --port=8000 \
    --model=/data/models/qwen/qwq-32b \
    --max-model-len 16384 \
    --max-num-batched-tokens 65536 \
    --tokenizer /data/models/qwen/qwq-32b \
    --api-key=openai \
    --trust-remote-code \
    --tensor-parallel-size=1 \
    --gpu-memory-utilization=0.95 \
    --dtype float16 \
    --served-model-name qwq-32b \
    --device="cuda"