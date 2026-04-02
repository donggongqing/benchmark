#!/bin/bash
set -euo pipefail

# --- User Configurations ---
# You can edit server parameters like model path inside your YAML files directly in the `server_config:` block.
WORKLOAD_YAML="configs/workloads/optimal_sets.yaml"
# ---------------------------

echo "Starting SGLang Benchmark toolkit"
echo "============================================================"
cd "$(dirname "$0")" || true
python run_eval.py --workload "$WORKLOAD_YAML"
echo "✅ Benchmark Completed"
