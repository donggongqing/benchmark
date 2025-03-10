#!/bin/bash
MODEL_NAME="$1"
test -n "$MODEL_NAME"
export OPENAI_API_KEY=openai
export OPENAI_API_BASE="http://0.0.0.0:4000/v1"
export HF_ENDPOINT=https://hf-mirror.com

python llmperf/token_benchmark_ray.py \
--model $MODEL_NAME \
--mean-input-tokens 1024 \
--stddev-input-tokens 24 \
--mean-output-tokens 64 \
--stddev-output-tokens 4 \
--max-num-completed-requests 10 \
--timeout 600 \
--num-concurrent-requests 2 \
--results-dir "result_outputs" \
--llm-api openai