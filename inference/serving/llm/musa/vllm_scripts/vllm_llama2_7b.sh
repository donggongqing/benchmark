#!/bin/bash
export HF_ENDPOINT=https://hf-mirror.com

python -O -u -m vllm.entrypoints.openai.api_server \
    --host=127.0.0.1 \
    --port=8000 \
    --model=/data/mtt/model_convert/llama-2-7b-hf-fp16-tp1-convert \
    --max-model-len=2048 \
    --tokenizer=hf-internal-testing/llama-tokenizer \
    --api-key=openai \
    --trust-remote-code \
    --tensor-parallel-size=1 \
    --block-size=64 \
    -pp=1 \
    --gpu-memory-utilization=0.95 \
    --device="musa"