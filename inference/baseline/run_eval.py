import os
import socket
import csv
import json
import argparse
import yaml
from datetime import datetime
from pathlib import Path

from src.collect_env import collect_environment
from src.backend_sgl import SGLangEngine
from src.orchestrator import run_workload_matrix
from src.report_generator import generate_final_report
from src.validator import validate_results

CSV_HEADERS = [
    "input_len", "output_len", "max_concurrency", "request_rate", "num_prompts", "Successful_requests",
    "Benchmark_duration_s", "Total_input_tokens", "Total_generated_tokens", "Request_throughput_req_s",
    "Output_token_throughput_tok_s", "Total_Token_throughput_tok_s", "Mean_TTFT_ms", "Median_TTFT_ms",
    "P95_TTFT_ms", "P99_TTFT_ms", "Mean_TPOT_ms", "Median_TPOT_ms", "P95_TPOT_ms", "P99_TPOT_ms",
    "Mean_ITL_ms", "Median_ITL_ms", "P95_ITL_ms", "P99_ITL_ms", "Mean_E2EL_ms", "Median_E2EL_ms",
    "P95_E2EL_ms", "P99_E2EL_ms"
]

def main():
    parser = argparse.ArgumentParser(description="Baseline LLM Benchmark")
    parser.add_argument("--model", type=str, default=None, help="Path to the model directory (overrides yaml)")
    parser.add_argument("--tp", type=int, default=None, help="Tensor parallelism degree (overrides yaml)")
    parser.add_argument("--workload", type=str, required=True, help="Path to YAML workload config")
    
    # HYBRID MODE ARGS
    parser.add_argument("--host", type=str, default=None, help="Remote server host (enables remote mode)")
    parser.add_argument("--port", type=int, default=None, help="Remote server port (overrides yaml)")
    
    args, extra_args = parser.parse_known_args()

    # Load YAML workload config
    with open(args.workload, "r") as f:
        workload_config = yaml.safe_load(f)
    
    server_config = workload_config.get("server_config", {})
    
    # Merge CLI and YAML (CLI takes precedence)
    args.model = args.model or server_config.get("model_path")
    if not args.model:
        parser.error("--model must be provided either via CLI or workload yaml server_config.model_path")
        
    args.tp = args.tp if args.tp else server_config.get("tp", 8)
    args.host = args.host or server_config.get("host")
    args.port = args.port if args.port is not None else server_config.get("port", 30000)
    
    yaml_extra_args = server_config.get("extra_args", [])
    if isinstance(yaml_extra_args, str):
        yaml_extra_args = yaml_extra_args.split()
    extra_args = yaml_extra_args + extra_args

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
        target_host = args.host if args.host else "127.0.0.1"
        target_port = args.port if args.port else 30000

        def is_port_in_use(host, port):
            check_host = "127.0.0.1" if host == "0.0.0.0" else host
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(1.0)
                return s.connect_ex((check_host, port)) == 0

        engine = None
        
        # If user explicitly provided a host/port, we check if it's up.
        # If it is up, we use it without starting a local server.
        # If it is NOT up (or if they didn't provide one), we start the local server.
        
        is_running = is_port_in_use(target_host, target_port)
        
        if is_running:
            print(f"\n🌐 [HYBRID MODE] Separated: Evaluator found running Server at {target_host}:{target_port}")
        else:
            print(f"\n💻 [HYBRID MODE] Integrated: No server found at {target_host}:{target_port}. Starting Local Server automatically.")
            
            host_args = []
            if "--host" not in extra_args:
                host_args.extend(["--host", target_host])
            if "--port" not in extra_args:
                host_args.extend(["--port", str(target_port)])
                
            logs_dir = result_dir / "logs"
            logs_dir.mkdir(parents=True, exist_ok=True)
            engine = SGLangEngine(model_path=args.model, tp_size=args.tp, extra_args=extra_args + host_args, log_dir=str(logs_dir))
        
        try:
            print("\n=======================================================")
            if engine:
                engine.start_server()

            def update_report():
                csvfile.flush()
                os.fsync(csvfile.fileno())
                generate_final_report(result_dir, csv_path, env_info, args, engine)

            # 5. Run the Matrix
            if os.path.exists(args.workload):
                run_workload_matrix(engine, writer, args.workload, target_host, target_port, result_dir, on_step_complete=update_report)
            else:
                print(f"⚠️ Error: Could not find '{args.workload}'.")
                
        finally:
            print("\n=======================================================")
            if engine:
                engine.stop_server()
            print(f"📊 Intermediate Raw CSV: {csv_path}")

    # 6. Generate final Excel-ready report
    print("\nGenerating final comprehensive Excel-ready CSV report...")
    final_csv = generate_final_report(result_dir, csv_path, env_info, args, engine)

    # 7. Validate results against theoretical hardware bounds
    hw_info = env_info.get("hardware", {})
    vendor_details = hw_info.get("details", {})
    gpus = vendor_details.get("gpus", [])
    gpu_name = gpus[0].split(",")[0].strip() if gpus and gpus[0] else ""

    if final_csv and gpu_name:
        print("\n🔎 Validating results against theoretical hardware bounds...")
        report, report_path = validate_results(
            csv_path=str(final_csv),
            model_path=args.model,
            gpu_name=gpu_name,
            tp=args.tp,
            output_dir=str(result_dir),
        )

        total_warnings = len(report.get("row_warnings", [])) + len(report.get("trend_warnings", []))
        if total_warnings > 0:
            print(f"\n⚠️  Validation found {total_warnings} warning(s):")
            for w in report.get("row_warnings", []):
                row_tag = f"[Row {w['row_index']}] " if w.get("row_index") else ""
                print(f"   {row_tag}{w['severity'].upper()}: {w['message']}")
            for w in report.get("trend_warnings", []):
                print(f"   TREND {w['severity'].upper()}: {w['message']}")
            print(f"   Full report: {report_path}")
        else:
            print("✅ Validation passed — no anomalies detected.")
    else:
        if not gpu_name:
            print("\n⚠️  Skipping validation: GPU name not detected in environment info.")

if __name__ == "__main__":
    main()
