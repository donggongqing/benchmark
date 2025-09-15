#!/bin/bash
set -euo pipefail

MODEL_PATH="/data/model/deepseek-r1-distill-llama-70b/"

# Input/output length pairs
IO_PAIRS=(
    "16 16"
    "32 32"
)

# Concurrency and num-prompts (1-to-1 mapping)
CONCURRENCY_AND_PROMPTS=(
    "1 4"
    "2 4"
)

DATASET_NAME="random"

MODEL_NAME=$(basename "${MODEL_PATH%/}" | tr '[:upper:]' '[:lower:]')
OUTPUT_FILE="vllm_bench_${MODEL_NAME}_results.csv"
BASE_DIR="output_result"
RESULT_DIR="vllm_results"

# Ensure output directories exist
mkdir -p "$BASE_DIR"
mkdir -p "$RESULT_DIR"

FULL_OUTPUT_PATH="$BASE_DIR/$OUTPUT_FILE"
echo "" > "$FULL_OUTPUT_PATH"  # Clear file if exists

# CSV header (with P95 fields included)
HEADER="input_len,output_len,max_concurrency,num_prompts,Successful_requests,"\
"Benchmark_duration_s,Total_input_tokens,Total_generated_tokens,Request_throughput_req_s,"\
"Output_token_throughput_tok_s,Total_Token_throughput_tok_s,Mean_TTFT_ms,Median_TTFT_ms,"\
"P95_TTFT_ms,P99_TTFT_ms,Mean_TPOT_ms,Median_TPOT_ms,P95_TPOT_ms,P99_TPOT_ms,"\
"Mean_ITL_ms,Median_ITL_ms,P95_ITL_ms,P99_ITL_ms,Mean_E2EL_ms,Median_E2EL_ms,"\
"P95_E2EL_ms,P99_E2EL_ms"
echo "$HEADER" >> "$FULL_OUTPUT_PATH"

echo "📊 Benchmark started for model: $MODEL_NAME"
echo "📂 Saving CSV results to: $FULL_OUTPUT_PATH"
echo

# Loop through IO_PAIRS
for PAIR in "${IO_PAIRS[@]}"; do
    read W O <<< "$PAIR"

    # Loop through concurrency and corresponding num-prompts
    for CP in "${CONCURRENCY_AND_PROMPTS[@]}"; do
        read C N <<< "$CP"
        echo "===== Running benchmark: input_len=$W, output_len=$O, max_concurrency=$C, num_prompts=$N ====="

        RESULT=$(vllm bench serve \
            --model "$MODEL_PATH" \
            --served-model-name "$MODEL_NAME" \
            --dataset-name "$DATASET_NAME" \
            --random-input-len "$W" \
            --random-output-len "$O" \
            --num-prompts "$N" \
            --ignore-eos \
            --save-result \
            --percentile-metrics 'ttft,tpot,itl,e2el' \
            --metric-percentiles "95,99" \
            --result-dir "$RESULT_DIR" \
            --max-concurrency "$C" \
            2>&1 | awk '/============ Serving Benchmark Result ============/{flag=1; next} /==================================================/{flag=0} flag')

        CSV_LINE=$(echo "$RESULT" | awk '
            function clean(s){gsub(/^[ \t]+|[ \t]+$/,"",s); return s}
            /Successful requests:/ {sr=clean($NF)}
            /Benchmark duration/ {bd=clean($NF)}
            /Total input tokens/ {ti=clean($NF)}
            /Total generated tokens/ {tg=clean($NF)}
            /Request throughput/ {rt=clean($NF)}
            /Output token throughput/ {ot=clean($NF)}
            /Total Token throughput/ {tt=clean($NF)}
            /Mean TTFT/ {mttft=clean($NF)}
            /Median TTFT/ {medttft=clean($NF)}
            /P95 TTFT/ {p95ttft=clean($NF)}
            /P99 TTFT/ {p99ttft=clean($NF)}
            /Mean TPOT/ {mtpot=clean($NF)}
            /Median TPOT/ {medtpot=clean($NF)}
            /P95 TPOT/ {p95tpot=clean($NF)}
            /P99 TPOT/ {p99tpot=clean($NF)}
            /Mean ITL/ {mitl=clean($NF)}
            /Median ITL/ {meditl=clean($NF)}
            /P95 ITL/ {p95itl=clean($NF)}
            /P99 ITL/ {p99itl=clean($NF)}
            /Mean E2EL/ {me2el=clean($NF)}
            /Median E2EL/ {mede2el=clean($NF)}
            /P95 E2EL/ {p95e2el=clean($NF)}
            /P99 E2EL/ {p99e2el=clean($NF)}
            END {
                printf "%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n",
                "'$W'","'$O'","'$C'","'$N'",
                sr, bd, ti, tg, rt, ot, tt,
                mttft, medttft, p95ttft, p99ttft,
                mtpot, medtpot, p95tpot, p99tpot,
                mitl, meditl, p95itl, p99itl,
                me2el, mede2el, p95e2el, p99e2el
            }'
        )

        echo "$CSV_LINE" >> "$FULL_OUTPUT_PATH"

        MEAN_E2EL=$(echo "$CSV_LINE" | cut -d',' -f25)
        echo "✅ Completed: input_len=$W, output_len=$O, max_concurrency=$C, num_prompts=$N, Mean_E2EL_ms=$MEAN_E2EL"
        echo
    done
done

echo "🎉 All benchmarks finished!"
echo "📊 Final CSV: $FULL_OUTPUT_PATH"