#!/bin/bash
set -euo pipefail

# Allows overriding from environment without editing the script
MODEL_NAME=${MODEL_NAME:-"sglang_model"}
TP_NUM=${TP_NUM:-8}

# Input/output length pairs for the random dataset
IO_PAIRS=(
    "256 256"
    "512 512"
    "1024 1024"
    "2048 1024"
    "3072 1024"
    "4096 1024"
)

# Concurrency and num-prompts pairs (1-to-1 mapping)
CONCURRENCY_AND_PROMPTS=(
    "1 2"
    "2 4"
    "4 8"
    "8 16"
    "16 32"
    "32 64"
    "64 128"
    "128 256"
    "256 512"
)

DATASET_NAME=${DATASET_NAME:-"random-ids"}
WARMUP_REQUESTS=10
BACKEND="sglang"
BASE_DIR="output_result"
DATA_TIME=$(date +%Y%m%d_%H%M%S)
OUTPUT_FILE="sgl_bench_${MODEL_NAME}_tp${TP_NUM}_${DATA_TIME}_results.csv"
FULL_OUTPUT_PATH="${BASE_DIR}/${OUTPUT_FILE}"

mkdir -p "${BASE_DIR}"
: > "${FULL_OUTPUT_PATH}"

HEADER="input_len,output_len,max_concurrency,num_prompts,Successful_requests,"
HEADER+="Benchmark_duration_s,Total_input_tokens,Total_generated_tokens,Request_throughput_req_s,"
HEADER+="Output_token_throughput_tok_s,Total_Token_throughput_tok_s,Mean_TTFT_ms,Median_TTFT_ms,"
HEADER+="P95_TTFT_ms,P99_TTFT_ms,Mean_TPOT_ms,Median_TPOT_ms,P95_TPOT_ms,P99_TPOT_ms,"
HEADER+="Mean_ITL_ms,Median_ITL_ms,P95_ITL_ms,P99_ITL_ms,Mean_E2EL_ms,Median_E2EL_ms,"
HEADER+="P95_E2EL_ms,P99_E2EL_ms"
echo "${HEADER}" >> "${FULL_OUTPUT_PATH}"

echo "📊 Benchmark started for backend: ${BACKEND}"
echo "📝 Recording results as model: ${MODEL_NAME} (for tracking only)"
echo "📂 Saving CSV results to: ${FULL_OUTPUT_PATH}"
echo

for PAIR in "${IO_PAIRS[@]}"; do
    read -r INPUT_LEN OUTPUT_LEN <<< "${PAIR}"

    for CP in "${CONCURRENCY_AND_PROMPTS[@]}"; do
        read -r MAX_CONCURRENCY NUM_PROMPTS <<< "${CP}"
        echo "===== Running benchmark: input_len=${INPUT_LEN}, output_len=${OUTPUT_LEN}, max_concurrency=${MAX_CONCURRENCY}, num_prompts=${NUM_PROMPTS} ====="

        CMD_OUTPUT=$(python3 -m sglang.bench_serving \
            --backend "${BACKEND}" \
            --dataset-name "${DATASET_NAME}" \
            --random-input-len "${INPUT_LEN}" \
            --random-output-len "${OUTPUT_LEN}" \
            --warmup-requests "${WARMUP_REQUESTS}" \
            --output-details \
            --flush-cache \
            --num-prompts "${NUM_PROMPTS}" \
            --max-concurrency "${MAX_CONCURRENCY}" \
            2>&1)

        BENCH_SECTION=$(echo "${CMD_OUTPUT}" | awk '/^============ Serving Benchmark Result ============/ {flag=1; next} /^==================================================$/ {flag=0} flag')

        if [ -z "${BENCH_SECTION}" ]; then
            echo "❌ Failed to parse benchmark output:"
            echo "${CMD_OUTPUT}"
            exit 1
        fi

        CSV_LINE=$(echo "${BENCH_SECTION}" | awk '
            function clean(s) { gsub(/^[ \t]+|[ \t]+$/, "", s); return s }
            BEGIN {
                sr=""; bd=""; ti=""; tg=""; rt=""; ot=""; tt="";
                mttft=""; medttft=""; p95ttft=""; p99ttft="";
                mtpot=""; medtpot=""; p95tpot=""; p99tpot="";
                mitl=""; meditl=""; p95itl=""; p99itl="";
                me2el=""; mede2el=""; p95e2el=""; p99e2el="";
            }
            /Successful requests:/ { sr = clean($NF) }
            /Benchmark duration/ { bd = clean($NF) }
            /Total input tokens:/ {
                if (index($0, "text") == 0 && index($0, "vision") == 0) {
                    ti = clean($NF)
                }
            }
            /Total generated tokens:/ {
                if (index($0, "retokenized") == 0) {
                    tg = clean($NF)
                }
            }
            /Request throughput/ { rt = clean($NF) }
            /Output token throughput/ { ot = clean($NF) }
            tolower($0) ~ /total token throughput/ { tt = clean($NF) }
            /Mean TTFT/ { mttft = clean($NF) }
            /Median TTFT/ { medttft = clean($NF) }
            /P95 TTFT/ { p95ttft = clean($NF) }
            /P99 TTFT/ { p99ttft = clean($NF) }
            /Mean TPOT/ { mtpot = clean($NF) }
            /Median TPOT/ { medtpot = clean($NF) }
            /P95 TPOT/ { p95tpot = clean($NF) }
            /P99 TPOT/ { p99tpot = clean($NF) }
            /Mean ITL/ { mitl = clean($NF) }
            /Median ITL/ { meditl = clean($NF) }
            /P95 ITL/ { p95itl = clean($NF) }
            /P99 ITL/ { p99itl = clean($NF) }
            /Mean E2E Latency/ { me2el = clean($NF) }
            /Median E2E Latency/ { mede2el = clean($NF) }
            /P95 E2E Latency/ { p95e2el = clean($NF) }
            /P99 E2E Latency/ { p99e2el = clean($NF) }
            END {
                printf "%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n",
                    "'$INPUT_LEN'","'$OUTPUT_LEN'","'$MAX_CONCURRENCY'","'$NUM_PROMPTS'",
                    sr, bd, ti, tg, rt, ot, tt,
                    mttft, medttft, p95ttft, p99ttft,
                    mtpot, medtpot, p95tpot, p99tpot,
                    mitl, meditl, p95itl, p99itl,
                    me2el, mede2el, p95e2el, p99e2el;
            }
        ')

        echo "${CSV_LINE}" >> "${FULL_OUTPUT_PATH}"
        MEAN_E2EL=$(echo "${CSV_LINE}" | cut -d',' -f25)
        echo "✅ Completed: input_len=${INPUT_LEN}, output_len=${OUTPUT_LEN}, max_concurrency=${MAX_CONCURRENCY}, num_prompts=${NUM_PROMPTS}, Mean_E2EL_ms=${MEAN_E2EL}"
        echo
    done

done

echo "🎉 All benchmarks finished!"
echo "📊 Final CSV: ${FULL_OUTPUT_PATH}"
