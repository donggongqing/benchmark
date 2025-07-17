#!/bin/bash
MODEL_NAME="$1"
SAVE_DIR="$2"
INPUT_TOKENS="$3"
OUTPUT_TOKENS="$4"
CONCURRENT="$5"

test -n "$MODEL_NAME"
export OPENAI_API_KEY=openai
export OPENAI_API_BASE="http://0.0.0.0:8000/v1"
export HF_ENDPOINT=https://hf-mirror.com

python llmperf/token_benchmark_ray.py \
--model $MODEL_NAME \
--mean-input-tokens $INPUT_TOKENS \
--stddev-input-tokens 16 \
--mean-output-tokens $OUTPUT_TOKENS \
--stddev-output-tokens 16 \
--max-num-completed-requests 1024 \
--timeout 3600 \
--num-concurrent-requests $CONCURRENT \
--results-dir "result_outputs/$SAVE_DIR" \
--llm-api openai 