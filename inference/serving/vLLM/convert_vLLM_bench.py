import argparse
import pandas as pd
import os
from datetime import datetime, timezone


TARGET_COLS = [
    "model","gpu","gpuNum","tp","dp","pp","ep","batch",
    "mean_input_tokens","mean_output_tokens","num_concurrent_requests",
    "results_inter_token_latency_s_mean","results_ttft_s_mean","results_tpot_s_mean",
    "results_end_to_end_latency_s_mean",
    "results_request_output_throughput_token_per_s_mean",
    "results_mean_output_throughput_token_per_s",
    "results_num_completed_requests",
    "dataType","driver","driverVersion","backend","backendVersion",
    "engine","engineVersion","serving","servingVersion","timestamp","source"
]

def safe_div(a, b):
    try:
        if b is None or (isinstance(b, (int, float)) and b == 0):
            return float("nan")
        return a / b
    except Exception:
        return float("nan")

def transform_row(row, args):

    out = {}

    # Fields primarily populated by args
    out["model"] = args.model_alias
    out["gpu"] = args.gpu or ""
    out["gpuNum"] = args.gpu_num if args.gpu_num is not None else ""
    out["tp"] = args.tp if args.tp is not None else ""
    out["dp"] = args.dp if args.dp is not None else ""
    out["pp"] = args.pp if args.pp is not None else ""
    out["ep"] = args.ep if args.ep is not None else ""


    # Calculate mean tokens (based on Successful_requests)
    succ = row.get("Successful_requests", None)
    total_in = row.get("Total_input_tokens", None)
    total_out = row.get("Total_generated_tokens", None)
    max_conc = row.get("max_concurrency", None)

    # Check the numerical calculations in the transform_row function
    out["mean_input_tokens"] = safe_div(total_in, succ) if (pd.notna(total_in) 
        and pd.notna(succ) and succ > 0) else float("nan")
    out["mean_output_tokens"] = safe_div(total_out, succ) if pd.notna(total_out) and pd.notna(succ) else float("nan")

    # Direct mapping of concurrent requests
    out["batch"] = int(max_conc) if pd.notna(max_conc) else ""
    out["num_concurrent_requests"] = int(max_conc) if pd.notna(max_conc) else ""


    itl_ms = row.get("Mean_ITL_ms", None)
    ttft_ms = row.get("Mean_TTFT_ms", None)
    tpot_ms = row.get("Mean_TPOT_ms", None)
    e2e_ms = row.get("Mean_E2EL_ms", None)
    #  ms -> s
    out["results_inter_token_latency_s_mean"] = (itl_ms / 1000.0) if pd.notna(itl_ms) else float("nan")
    out["results_ttft_s_mean"] = (ttft_ms / 1000.0) if pd.notna(ttft_ms) else float("nan")
    out["results_tpot_s_mean"] = (tpot_ms / 1000.0) if pd.notna(tpot_ms) else float("nan")
    out["results_end_to_end_latency_s_mean"] = (e2e_ms / 1000.0) if pd.notna(e2e_ms) else float("nan")

    # Throughput: Overall output throughput (tok/s) and average per request (based on concurrency)
    out_put_tok_s = row.get("Output_token_throughput_tok_s", None)
    if pd.notna(out_put_tok_s):
        out["results_mean_output_throughput_token_per_s"] = out_put_tok_s
        # Average output throughput per request (NaN if concurrency is 0)
        if pd.notna(max_conc) and max_conc != 0:
            out["results_request_output_throughput_token_per_s_mean"] = out_put_tok_s / max_conc
        else:
            out["results_request_output_throughput_token_per_s_mean"] = float("nan")
    else:
        out["results_mean_output_throughput_token_per_s"] = float("nan")
        out["results_request_output_throughput_token_per_s_mean"] = float("nan")

    out["results_num_completed_requests"] = int(succ) if pd.notna(succ) else ""
    out["dataType"] = args.data_type
    out["driver"] = args.driver
    out["driverVersion"] = args.driver_version
    out["backend"] = args.backend
    out["backendVersion"] = args.backend_version
    out["engine"] = args.engine
    out["engineVersion"] = args.engine_version or ""
    out["serving"] = args.serving or ""
    out["servingVersion"] = args.serving_version or ""

    # timestamp: Prioritize using the values specified in args, otherwise use the current time or source file name
    out["timestamp"] = args.timestamp or datetime.now(datetime.timezone.utc).isoformat()
    out["source"] = args.source
    return out

def parse_args():
    parser = argparse.ArgumentParser(description='convert bench result')
    
    parser.add_argument('--model', required=True, help='Model name (e.g., deepseek-r1)')
    parser.add_argument('--model-alias', required=True, help='Uppercase model alias')
    parser.add_argument('--tp', required=True, help='Tensor parallelism (default: 1)', default=1)
    parser.add_argument('--dp', help='Data parallelism (default: 1)', default='')
    parser.add_argument('--pp', help='Pipeline parallelism (default: 1)', default='')
    parser.add_argument('--ep', help='Expert parallelism (default: 1)', default='')
    parser.add_argument('--data-type', required=True, help='Precision type', default='FP16')
    parser.add_argument('--gpu', required=True, help='GPU model (e.g., A100)')
    parser.add_argument('--gpu-num', type=int, required=True, help='Number of GPUs (default: 1)', default=1)
    parser.add_argument('--driver', required=True, help='Driver model (e.g., NVIDIA-Linux-x86_64)')
    parser.add_argument('--driver-version', required=True, help='Driver version (e.g., 535.161.08)')
    parser.add_argument('--backend', required=True, help='Backend model (e.g., cuda)')
    parser.add_argument('--backend-version', required=True, help='Backend version (e.g., 2.8.0)')
    parser.add_argument('--engine', required=True, help='Engine model (e.g., cuda)')
    parser.add_argument('--engine-version', required=True, help='Engine version (e.g., 2.8.0)')
    parser.add_argument('--serving', required=True, help='Serving model (e.g., vllm)')
    parser.add_argument('--serving-version', required=True, help='Serving version (e.g., 0.7.3)')
    parser.add_argument('--source', required=True, help='Source (e.g., vllm)')
    parser.add_argument('--timestamp', 
        default=datetime.now(timezone.utc).isoformat(),
        help='Timestamp in ISO format')
    # Future feature expansion
    parser.add_argument('--input', help='output_result/vllm_bench_<model>_results.csv')
    parser.add_argument('--output', help='convert_output/vllm_bench_<model>_converted.csv')

    return parser.parse_args()


def main():
    args = parse_args()
    
    # Automatically generate the input path
    if not args.input:
        args.input = os.path.join('output_result', f'vllm_bench_{args.model}_results.csv')
    
    # Automatically create the output directory and generate the path
    if not args.output:
        output_dir = 'convert_output'
        os.makedirs(output_dir, exist_ok=True)
        output_file = f'vllm_bench_{args.model}_gpu{args.gpu}_tp{args.tp}_converted.csv'
        args.output = os.path.join(output_dir, output_file)

    # Add file existence check
    if not os.path.exists(args.input):
        print(f"Error: Input file does not exist - {args.input}")
        return
    
    df = pd.read_csv(args.input)

    # Check if necessary columns exist 
    required_cols = ["max_concurrency", "Successful_requests", "Total_input_tokens", "Total_generated_tokens",
                     "Output_token_throughput_tok_s", "Mean_ITL_ms", "Mean_TTFT_ms", "Mean_TPOT_ms", "Mean_E2EL_ms"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        print(f"[ERROR] input CSV missing columns: {missing}")
        print("Example of known input columns:")
        print(", ".join(required_cols))
        return

    out_rows = []
    for _, row in df.iterrows():
        # Add empty line filtering in the main() function
        if pd.notna(row['max_concurrency']) and pd.notna(row['Successful_requests']):
            out_rows.append(transform_row(row, args, src_name=os.path.basename(args.input)))

    out_df = pd.DataFrame(out_rows, columns=TARGET_COLS)

    # Check if the output file already exists to avoid accidental overwriting
    if os.path.exists(args.output):
        print(f"Warning: Output file already exists - {args.output}")
        response = input("Overwrite? (y/n): ").strip().lower()
        if response != 'y':
            print("Operation canceled")
            return

    # save CSV
    out_df.to_csv(args.output, index=False)
    print(f"Conversion complete: {args.output} (total {len(out_df)} rows)")

if __name__ == "__main__":
    main()
