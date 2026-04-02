from tqdm import tqdm
import subprocess
import re
import csv
from pathlib import Path

METRICS = [
    "Successful requests", "Benchmark duration", "Total input tokens",
    "Total generated tokens", "Request throughput", "Output token throughput",
    "Total Token throughput", "Mean TTFT", "Median TTFT", "P95 TTFT", "P99 TTFT",
    "Mean TPOT", "Median TPOT", "P95 TPOT", "P99 TPOT",
    "Mean ITL", "Median ITL", "P95 ITL", "P99 ITL",
    "Mean E2E Latency", "Median E2E Latency", "P95 E2E Latency", "P99 E2E Latency"
]

def run_sglang_benchmark(input_len: int, output_len: int, max_concurrency: int, request_rate: float, num_prompts: int, host: str, port: int, logs_dir: Path):
    """
    Executes sglang.bench_serving client against an already running server.
    """
    output_file = logs_dir / f"sglang_{input_len}_{output_len}_c{max_concurrency or 0}_q{request_rate or 0}_{num_prompts}.jsonl"
    
    cmd = [
        "python", "-m", "sglang.bench_serving",
        "--backend", "sglang",
        "--dataset-name", "random-ids",
        "--random-input-len", str(input_len),
        "--random-output-len", str(output_len),
        "--warmup-requests", "2",
        "--output-details",
        "--output-file", str(output_file),
        "--flush-cache",
        "--num-prompts", str(num_prompts),
        "--host", host,
        "--port", str(port)
    ]

    if max_concurrency:
        cmd.extend(["--max-concurrency", str(max_concurrency)])
        mode_msg = f"{max_concurrency} max cc"
    elif request_rate:
        cmd.extend(["--request-rate", str(request_rate)])
        mode_msg = f"{request_rate} qps"
    else:
        mode_msg = "Unknown mode"

    tqdm.write(f"    ▶ Benchmarking {input_len}/{output_len} @ {mode_msg} (Prompts: {num_prompts}) on {host}:{port}")

    # Catch OOMs here if the client fails
    output = subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT)
    return output

def parse_and_write_results(csv_writer, output_text, input_len, output_len, max_conc, req_rate, prompts):
    data = {}
    
    for line in output_text.splitlines():
        line = line.strip()
        for metric in METRICS:
            if line.lower().startswith(metric.lower()):
                # Skip sub-breakdowns printed by sglang
                if metric == "Total input tokens" and ("text" in line.lower() or "vision" in line.lower()):
                    continue
                if metric == "Total generated tokens" and "retokenized" in line.lower():
                    continue

                m = re.search(r"([-+]?\d*\.\d+|\d+)\s*$", line)
                if m:
                    data[metric] = m.group(1)
                break

    # Build CSV row using exact sequence
    row = [input_len, output_len, max_conc or "N/A", req_rate or "N/A", prompts] + [data.get(m, "") for m in METRICS]
    csv_writer.writerow(row)
    tqdm.write(f"      ✅ Recorded row. Throughput: {data.get('Request throughput', 'N/A')} req/s")
