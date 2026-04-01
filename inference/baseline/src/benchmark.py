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

def run_sglang_benchmark(input_len: int, output_len: int, max_concurrency: int, num_prompts: int, host: str, port: int):
    """
    Executes sglang.bench_serving client against an already running server.
    """
    cmd = [
        "python", "-m", "sglang.bench_serving",
        "--backend", "sglang",
        "--dataset-name", "random-ids",
        "--random-input-len", str(input_len),
        "--random-output-len", str(output_len),
        "--warmup-requests", "2",
        "--output-details",
        "--flush-cache",
        "--num-prompts", str(num_prompts),
        "--max-concurrency", str(max_concurrency),
        "--host", host,
        "--port", str(port)
    ]

    print(f"    ▶ Benchmarking {input_len}/{output_len} @ {max_concurrency} max cc (Prompts: {num_prompts}) on {host}:{port}")
    
    # Catch OOMs here if the client fails
    output = subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT)
    return output

def parse_and_write_results(csv_writer, output_text, input_len, output_len, max_conc, prompts):
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
    row = [input_len, output_len, max_conc, prompts] + [data.get(m, "") for m in METRICS]
    csv_writer.writerow(row)
    print(f"      ✅ Recorded row. Throughput: {data.get('Request throughput', 'N/A')} req/s")
