#!/bin/bash
MODEL_PATH="/data/models/deepseek-ai/deepseek-r1-distill-qwen-1.5b"
# Input length and output length lists
INPUT_LIST=(128 256 512 1024 2048 4096)
OUTPUT_LIST=(4)  # Can be extended with different output lengths
# Concurrency and num-prompts lists
CONCURRENCY_LIST=(1 4 8 16 32 64 128)
NUM_PROMPTS_LIST=(4)  # Can be extended



DATASET_NAME="random"
RESULT_DIR="vllm_results"
BASE_DIR=output_result
MODEL_NAME=$(basename "${MODEL_PATH%/}" | tr '[:upper:]' '[:lower:]')
OUTPUT_FILE="vllm_bench_${MODEL_NAME}_results.csv"
# Create output directory if it doesn't exist
mkdir -p $BASE_DIR 2>/dev/null
# Construct full output path
FULL_OUTPUT_PATH="$BASE_DIR/$OUTPUT_FILE"
echo "" > $FULL_OUTPUT_PATH  # Empty the file

# CSV header
HEADER="input_len,output_len,max_concurrency,num_prompts,Successful_requests,Benchmark_duration_s,Total_input_tokens,Total_generated_tokens,Request_throughput_req_s,Output_token_throughput_tok_s,Total_Token_throughput_tok_s,Mean_TTFT_ms,Median_TTFT_ms,P99_TTFT_ms,Mean_TPOT_ms,Median_TPOT_ms,P99_TPOT_ms,Mean_ITL_ms,Median_ITL_ms,P99_ITL_ms,Mean_E2EL_ms,Median_E2EL_ms,P99_E2EL_ms"
echo $HEADER >> $FULL_OUTPUT_PATH

# Four-layer loop to generate all combinations
for W in "${INPUT_LIST[@]}"; do
    for O in "${OUTPUT_LIST[@]}"; do
        for C in "${CONCURRENCY_LIST[@]}"; do
            for N in "${NUM_PROMPTS_LIST[@]}"; do
                echo "===== Running benchmark: input_len=$W, output_len=$O, max_concurrency=$C, num_prompts=$N ====="

                # Run benchmark and capture Serving Benchmark Result
                RESULT=$(vllm bench serve \
                    --model $MODEL_PATH \
                    --served-model-name $MODEL_NAME \
                    --dataset-name $DATASET_NAME \
                    --random-input-len $W \
                    --random-output-len $O \
                    --num-prompts $N \
                    --ignore-eos \
                    --percentile-metrics 'ttft,tpot,itl,e2el' \
                    --result-dir $RESULT_DIR \
                    --max-concurrency $C \
                    2>&1 | awk '/============ Serving Benchmark Result ============/{flag=1; next} /==================================================/{flag=0} flag')

                # Extract numeric values and generate CSV line
                CSV_LINE=$(echo "$RESULT" | awk -F':' '
                    function clean(s){gsub(/^[ \t]+|[ \t]+$/,"",s); return s}
                    /Successful requests/ {sr=clean($2)}
                    /Benchmark duration/ {bd=clean($2)}
                    /Total input tokens/ {ti=clean($2)}
                    /Total generated tokens/ {tg=clean($2)}
                    /Request throughput/ {rt=clean($2)}
                    /Output token throughput/ {ot=clean($2)}
                    /Total Token throughput/ {tt=clean($2)}
                    /Mean TTFT/ {mttft=clean($2)}
                    /Median TTFT/ {medttft=clean($2)}
                    /P99 TTFT/ {p99ttft=clean($2)}
                    /Mean TPOT/ {mtpot=clean($2)}
                    /Median TPOT/ {medtpot=clean($2)}
                    /P99 TPOT/ {p99tpot=clean($2)}
                    /Mean ITL/ {mitl=clean($2)}
                    /Median ITL/ {meditl=clean($2)}
                    /P99 ITL/ {p99itl=clean($2)}
                    /Mean E2EL/ {me2el=clean($2)}
                    /Median E2EL/ {mede2el=clean($2)}
                    /P99 E2EL/ {p99e2el=clean($2)}
                    END {
                        printf "%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n",
                        "'$W'","'$O'","'$C'","'$N'",
                        sr, bd, ti, tg, rt, ot, tt,
                        mttft, medttft, p99ttft,
                        mtpot, medtpot, p99tpot,
                        mitl, meditl, p99itl,
                        me2el, mede2el, p99e2el
                    }')

                # Write to CSV
                echo "$CSV_LINE" >> $FULL_OUTPUT_PATH

                # Print log with perfectly aligned columns
                MEAN_E2EL=$(echo $CSV_LINE | cut -d',' -f21)
                echo ">>> Completed: input_len=$W, output_len=$O, max_concurrency=$C, num_prompts=$N, Mean_E2EL_ms=$MEAN_E2EL"
                echo
            done
        done
    done
done

echo "All benchmarks finished. Results saved to $FULL_OUTPUT_PATH."