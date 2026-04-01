#!/bin/bash
export HF_ENDPOINT=https://hf-mirror.com
export VLLM_WORKER_MULTIPROC_METHOD=spawn
python -O -u -m vllm.entrypoints.openai.api_server \
    --host=127.0.0.1 \
    --port=8000 \
    --model=/data/models/deepseek-ai/deepSeek-r1-distill-llama-70b \
    --max-model-len 16384 \
    --max-num-batched-tokens 65536 \
    --tokenizer /data/models/deepseek-ai/deepSeek-r1-distill-llama-70b \
    --api-key=openai \
    --trust-remote-code \
    --tensor-parallel-size=1 \
    --gpu-memory-utilization=0.95 \
    --dtype float16 \
    --served-model-name deepSeek-r1-distill-llama-70b \
    --device="cuda"