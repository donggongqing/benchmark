import os
import csv
import json
import argparse
from datetime import datetime
from pathlib import Path

from src.collect_env import collect_environment
from src.backend_sgl import SGLangEngine
from src.orchestrator import run_workload_matrix

CSV_HEADERS = [
    "input_len", "output_len", "max_concurrency", "num_prompts", "Successful_requests",
    "Benchmark_duration_s", "Total_input_tokens", "Total_generated_tokens", "Request_throughput_req_s",
    "Output_token_throughput_tok_s", "Total_Token_throughput_tok_s", "Mean_TTFT_ms", "Median_TTFT_ms",
    "P95_TTFT_ms", "P99_TTFT_ms", "Mean_TPOT_ms", "Median_TPOT_ms", "P95_TPOT_ms", "P99_TPOT_ms",
    "Mean_ITL_ms", "Median_ITL_ms", "P95_ITL_ms", "P99_ITL_ms", "Mean_E2EL_ms", "Median_E2EL_ms",
    "P95_E2EL_ms", "P99_E2EL_ms"
]

def main():
    parser = argparse.ArgumentParser(description="Baseline LLM Benchmark")
    parser.add_argument("--model", type=str, required=True, help="Path to the model directory")
    parser.add_argument("--tp", type=int, default=8, help="Tensor parallelism degree")
    parser.add_argument("--workload", type=str, required=True, help="Path to YAML workload config")
    
    # HYBRID MODE ARGS
    parser.add_argument("--host", type=str, default=None, help="Remote server host (enables remote mode)")
    parser.add_argument("--port", type=int, default=30000, help="Remote server port (default 30000)")
    
    args = parser.parse_args()

    # 1. Create timestamped result directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_name = Path(args.model).name.lower()
    
    mode_str = "remote" if args.host else "local"
    result_dir = Path(f"results/{timestamp}_{model_name}_tp{args.tp}_{mode_str}")
    result_dir.mkdir(parents=True, exist_ok=True)

    # 2. Collect Environment telemetry
    print("🔍 Collecting hardware/software environment...")
    env_info = collect_environment()
    with open(result_dir / "env_info.json", "w") as f:
        json.dump(env_info, f, indent=4)

    # 3. Setup CSV Writer
    csv_path = result_dir / f"sgl_bench_{model_name}_tp{args.tp}_{timestamp}_results.csv"
    with open(csv_path, mode='w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(CSV_HEADERS)

        # 4. Hybrid Engine Setup
        if args.host:
            print(f"\n🌐 [HYBRID MODE] Separated: Evaluator running against Remote Server at {args.host}:{args.port}")
            engine = None
            target_host = args.host
            target_port = args.port
        else:
            print("\n💻 [HYBRID MODE] Integrated: Evaluator will start/stop Local Server automatically.")
            engine = SGLangEngine(model_path=args.model, tp_size=args.tp)
            target_host = "127.0.0.1"
            target_port = 30000 # Default SGLang port
        
        try:
            print("\n=======================================================")
            if engine:
                engine.start_server()

            # 5. Run the Matrix
            if os.path.exists(args.workload):
                run_workload_matrix(engine, writer, args.workload, target_host, target_port)
            else:
                print(f"⚠️ Error: Could not find '{args.workload}'.")
                
        finally:
            print("\n=======================================================")
            if engine:
                engine.stop_server()
            print(f"📊 Final CSV: {csv_path}")

if __name__ == "__main__":
    main()
