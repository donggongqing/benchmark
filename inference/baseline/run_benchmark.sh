#!/bin/bash
set -euo pipefail
MODEL_PATH="${1:-/data/model/deepseek-r1-distill-llama-70b}"
TP_NUM="${2:-8}"
WORKLOAD_YAML="${3:-configs/workloads/optimal_sets.yaml}"
REMOTE_HOST="${4:-}"
REMOTE_PORT="${5:-30000}"
echo "============================================================"
echo "Starting SGLang Benchmark toolkit"
if [ -n "$REMOTE_HOST" ]; then echo "Mode: REMOTE/SEPARATED (Host: $REMOTE_HOST:$REMOTE_PORT)"; else echo "Mode: LOCAL/INTEGRATED"; fi
echo "============================================================"
cd "$(dirname "$0")" || true
if [ -n "$REMOTE_HOST" ]; then
  python run_eval.py --model "$MODEL_PATH" --tp "$TP_NUM" --workload "$WORKLOAD_YAML" --host "$REMOTE_HOST" --port "$REMOTE_PORT"
else
  python run_eval.py --model "$MODEL_PATH" --tp "$TP_NUM" --workload "$WORKLOAD_YAML"
fi
echo "? Benchmark Completed"
