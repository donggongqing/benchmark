#!/bin/bash
export HF_ENDPOINT=https://hf-mirror.com

python -O -u -m vllm.entrypoints.openai.api_server \
    --host=127.0.0.1 \
    --port=8000 \
    --model=/data/mtt/model_convert/Qwen2-72B-Instruct-fp16-tp8-convert \
    --max-model-len=8192 \
    --tokenizer=hf-internal-testing/llama-tokenizer \
    --api-key=openai \
    --trust-remote-code \
    --tensor-parallel-size=8 \
    --block-size=64 \
    -pp=1 \
    --gpu-memory-utilization=0.95 \
    --device="musa"