#!/bin/bash
set -euo pipefail

if [ $# -lt 1 ] || [ $# -gt 2 ]; then
	echo "Usage: $0 <input_csv> [output_csv]"
	exit 1
fi

INPUT_CSV="$1"
OUTPUT_CSV="${2:-}"

# Allow overriding defaults via environment variables when needed.
MODEL_ALIAS="${MODEL_ALIAS:-SGL_BENCH}"
TP="${TP:-8}"
DP="${DP:-}"
PP="${PP:-}"
EP="${EP:-}"
DATA_TYPE="${DATA_TYPE:-FP16}"
GPU_MODEL="${GPU_MODEL:-H100}"
GPU_NUM="${GPU_NUM:-8}"
DRIVER="${DRIVER:-NVIDIA-Linux-x86_64}"
DRIVER_VERSION="${DRIVER_VERSION:-550.54.14}"
BACKEND="${BACKEND:-sglang}"
BACKEND_VERSION="${BACKEND_VERSION:-0.1.0}"
ENGINE="${ENGINE:-cuda}"
ENGINE_VERSION="${ENGINE_VERSION:-12.4}"
SERVING="${SERVING:-sglang}"
SERVING_VERSION="${SERVING_VERSION:-0.1.0}"
SOURCE="${SOURCE:-sglang_perf}"
TIMESTAMP_OVERRIDE="${TIMESTAMP_OVERRIDE:-}"

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
CONVERTER="${SCRIPT_DIR}/convert_sgl.py"

if [ ! -f "${CONVERTER}" ]; then
	echo "Error: converter script not found at ${CONVERTER}"
	exit 1
fi

CMD=(
	python3 "${CONVERTER}" \
		--model-alias "${MODEL_ALIAS}" \
		--input "${INPUT_CSV}" \
		--tp "${TP}" \
		--dp "${DP}" \
		--pp "${PP}" \
		--ep "${EP}" \
		--data-type "${DATA_TYPE}" \
		--gpu "${GPU_MODEL}" \
		--gpu-num "${GPU_NUM}" \
		--driver "${DRIVER}" \
		--driver-version "${DRIVER_VERSION}" \
		--backend "${BACKEND}" \
		--backend-version "${BACKEND_VERSION}" \
		--engine "${ENGINE}" \
		--engine-version "${ENGINE_VERSION}" \
		--serving "${SERVING}" \
		--serving-version "${SERVING_VERSION}" \
		--source "${SOURCE}"
)

if [ -n "${TIMESTAMP_OVERRIDE}" ]; then
	CMD+=(--timestamp "${TIMESTAMP_OVERRIDE}")
fi

if [ -n "${OUTPUT_CSV}" ]; then
	CMD+=(--output "${OUTPUT_CSV}")
fi

exec "${CMD[@]}"
