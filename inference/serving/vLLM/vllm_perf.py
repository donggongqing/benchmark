#!/usr/bin/env python3
# This script is compatible with the configuration in the config.json file and supports setting the output path for test results.
# Usage: python vllm_perf_test.py --config <config.json> --output_dir </data/...>

import subprocess
import csv
from pathlib import Path
from typing import List, Dict, Any
import json
import argparse
import re
from itertools import product

# ----------------- Default Configuration (Customizable) -----------------
# Modify to the model path to be tested, which is used for 'bench serve' command
DEFAULT_MODEL_PATH = "/data/model/deepseek-r1-distill-llama-70b"   

# Modify the input length, output length, concurrency, and number of requests to be tested
IO_PAIRS = [(4096, 1024), (2048, 1024), (1024, 1024), (512, 512), (256, 256)]
CONC_PROMPTS = [(1, 4), (2, 4), (4, 8), (8, 16), (16, 32), (32, 64), (64, 128), (128, 256)]


# Each element is in the format: [input_len, output_len, max_concurrency, num_prompts]
DEFAULT_TEST_CASES = [[i, o, c, n] for (i, o), (c, n) in product(IO_PAIRS, CONC_PROMPTS)]

DATASET_NAME = "random"

BASE_DIR = Path("output_result")

CSV_HEADER = [
    "input_len", "output_len", "max_concurrency", "num_prompts", "Successful_requests",
    "Benchmark_duration_s", "Total_input_tokens", "Total_generated_tokens", "Request_throughput_req_s",
    "Output_token_throughput_tok_s", "Total_Token_throughput_tok_s", "Mean_TTFT_ms", "Median_TTFT_ms",
    "P95_TTFT_ms", "P99_TTFT_ms", "Mean_TPOT_ms", "Median_TPOT_ms", "P95_TPOT_ms", "P99_TPOT_ms",
    "Mean_ITL_ms", "Median_ITL_ms", "P95_ITL_ms", "P99_ITL_ms", "Mean_E2EL_ms", "Median_E2EL_ms",
    "P95_E2EL_ms", "P99_E2EL_ms"
]

# ----------------- Function Definitions -----------------

def parse_args():
    parser = argparse.ArgumentParser(description="Run vLLM performance benchmarks")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to JSON configuration file. If not provided, defaults will be used."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save benchmark CSV results. Defaults to 'output_result'."
    )
    return parser.parse_args()


def load_config(config_file: str) -> List[Dict[str, Any]]:
    """
    读取 JSON 配置文件，如果文件不存在或为 None 则使用默认配置
    返回列表，每个元素包含：
    {
        "model_path": str,
        "perf_test_cases": List[List[int, int, int, int]]
    }
    """
    if config_file and Path(config_file).exists():
        with open(config_file, "r") as f:
            config = json.load(f)
        print(f"✅ Loaded config from {config_file}")
        return config
    else:
        print(f"⚠️ Config file not found or not provided, using default model and test cases")
        return [{"model_path": DEFAULT_MODEL_PATH, "perf_test_cases": DEFAULT_TEST_CASES}]


def run_benchmark(model_path, model_name, input_len, output_len, max_concurrency, num_prompts):
    base_cmd = [
        "vllm", "bench", "serve",
        "--model", model_path,
        "--dataset-name", "random",
        "--random-input-len", str(input_len),
        "--random-output-len", str(output_len),
        "--num-prompts", str(num_prompts),
        "--ignore-eos",
        "--save-result",
        "--percentile-metrics", "ttft,tpot,itl,e2el",
        "--metric-percentiles", "95,99",
        "--result-dir", "vllm_results",
        "--max-concurrency", str(max_concurrency)
    ]

    # 带 --served-model-name和不带都尝试，失败后再不带
    cmd = base_cmd + ["--served-model-name", model_name]
    try:
        print("▶ Running with served-model-name bench...")
        output = subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT)
        return output
    except subprocess.CalledProcessError as e:
        try:
            # 不带 --served-model-name
            print("▶ Running without served-model-name bench...")
            output = subprocess.check_output(base_cmd, text=True, stderr=subprocess.STDOUT)
            return output
        except subprocess.CalledProcessError as e2:
            raise RuntimeError(
                f"Failed to start vllm bench serve:\n{e2.output}"
            ) from e2


def extract_and_save(output: str, input_len: int, output_len: int,
                     max_concurrency: int, num_prompts: int, writer: csv.writer):
    """
    从 vllm 输出log中提取指标并写入 CSV
    """
    data = {}

    # 定义指标列表
    metrics = [
        "Successful requests", "Benchmark duration", "Total input tokens",
        "Total generated tokens", "Request throughput", "Output token throughput",
        "Total Token throughput", "Mean TTFT", "Median TTFT", "P95 TTFT", "P99 TTFT",
        "Mean TPOT", "Median TPOT", "P95 TPOT", "P99 TPOT",
        "Mean ITL", "Median ITL", "P95 ITL", "P99 ITL",
        "Mean E2EL", "Median E2EL", "P95 E2EL", "P99 E2EL"
    ]

    for line in output.splitlines():
        line = line.strip()
        for metric in metrics:
            if line.startswith(metric):
                # 匹配行末最后一个数字（整数或浮点数）
                m = re.search(r"([-+]?\d*\.\d+|\d+)\s*$", line)
                if m:
                    data[metric] = m.group(1)
                break

    # 生成 CSV 行
    csv_row = [
        input_len, output_len, max_concurrency, num_prompts
    ] + [data.get(metric, "") for metric in metrics]

    writer.writerow(csv_row)
    print(f"✅ Completed: input_len={input_len}, output_len={output_len}, "
          f"max_concurrency={max_concurrency}, num_prompts={num_prompts}, "
          f"Benchmark duration_ms={data.get('Benchmark duration', '')}\n")



def main():
    args = parse_args()
    base_dir = Path(args.output_dir) if args.output_dir else BASE_DIR
    base_dir.mkdir(parents=True, exist_ok=True)

    configs = load_config(args.config)

    for cfg in configs:
        model_path = cfg["model_path"]
        model_name = Path(model_path).name.lower()
        output_file = base_dir / f"vllm_bench_{model_name}_results.csv"

        print(f"\n📊 Benchmark started for model: {model_name}")
        print(f"📂 Saving CSV results to: {output_file}\n")

        with open(output_file, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(CSV_HEADER)

            for case in cfg["perf_test_cases"]:
                input_len, output_len, max_concurrency, num_prompts = case
                print(f"===== Running benchmark: input_len={input_len}, output_len={output_len}, "
                      f"max_concurrency={max_concurrency}, num_prompts={num_prompts} =====")
                output = run_benchmark(model_path, model_name, input_len, output_len, max_concurrency, num_prompts)
                extract_and_save(output, input_len, output_len, max_concurrency, num_prompts, writer)

        print(f"🎉 Finished benchmarks for model: {model_name}")
        print(f"📊 CSV saved: {output_file}")


if __name__ == "__main__":
    main()
