# CLAUDE.md — Benchmark Project Guide

## Project Overview

This is an LLM inference benchmarking monorepo covering GPU/graphics profiling, offline inference, serving benchmarks, and training benchmarks. The primary active subsystem is the **Baseline Benchmark** framework under `inference/baseline/`.

---

## Baseline Benchmark (`inference/baseline/`)

### Purpose

Automated LLM inference performance baseline testing. Runs a configurable matrix of workloads (I/O token lengths × concurrency levels) against inference engines, collects latency/throughput metrics, and generates Excel-ready CSV reports.

### Architecture

```
inference/baseline/
├── run_eval.py              # Entry point — CLI arg parsing, hybrid mode detection, orchestration
├── run_benchmark.sh         # Shell wrapper, calls run_eval.py with default workload YAML
├── configs/
│   ├── workloads/           # YAML test matrices (io_pairs, concurrency, request_rate)
│   │   ├── optimal_sets.yaml      # Golden benchmark matrix (Short/Normal/Prefill/Decode)
│   │   ├── warmup_sets.yaml
│   │   └── warmup_qps_sets.yaml
│   ├── backends/            # (Planned) Backend-specific args (vllm, sglang, trtllm)
│   ├── env/                 # (Planned) Hardware topology configs (tp, dp, pp, ep, pd_mode)
│   └── models/              # (Planned) Model path/dtype/chat-template configs
├── src/
│   ├── engine.py            # BaseEngine ABC — start_server(), stop_server(), restart_server()
│   ├── backend_sgl.py       # SGLangEngine — launches sglang.launch_server via subprocess
│   ├── benchmark.py         # Runs sglang.bench_serving, parses stdout metrics into CSV rows
│   ├── orchestrator.py      # Iterates workload matrix, handles OOM recovery, tracks timing
│   ├── collect_env.py       # Detects GPU (nvidia-smi), CPU, framework versions, backend toolkits
│   └── report_generator.py  # Merges raw CSV + env_info + args into final Excel-ready report
└── results/                 # Output dir (timestamped subdirs per run)
```

### Key Concepts

- **Hybrid Mode**: `run_eval.py` probes the target `host:port` via socket.
  - **Integrated**: No server found → auto-launches local SGLang server, auto-stops after tests.
  - **Separated**: Server already running → acts as client only, does not manage server lifecycle.

- **Workload YAML** (`configs/workloads/*.yaml`): Defines `server_config` (model_path, tp, host, port, extra_args) and `sets` (named groups of `io_pairs` × `concurrency_and_prompts` or `request_rate_and_prompts`).

- **OOM Recovery**: `orchestrator.py` wraps each test in try/except. On failure, logs `FAILED_OOM`, restarts the engine (Integrated mode) or waits 30s (Separated mode), then skips remaining higher concurrencies for that I/O pair.

- **Metrics Collected**: TTFT, TPOT, ITL, E2E Latency (mean/median/p95/p99), request throughput, token throughput (input/output/total).

### Running

```bash
# Default (uses optimal_sets.yaml)
cd inference/baseline
./run_benchmark.sh

# Custom workload
python run_eval.py --workload configs/workloads/optimal_sets.yaml

# Override model/tp/host
python run_eval.py --workload configs/workloads/optimal_sets.yaml \
    --model /path/to/model --tp 4 --host 10.0.0.1 --port 30001
```

CLI args (`--model`, `--tp`, `--host`, `--port`) override YAML `server_config` values.

### Output Structure

Each run creates `results/<timestamp>_<model>_tp<N>_<mode>/` containing:
- `sgl_bench_*.csv` — Raw per-test metrics
- `final_report_*.csv` — Enriched report with GPU/CPU/driver/framework info, ready for Excel
- `env_info.json` — Hardware/software snapshot
- `elapsed_time.json` — Per-test and per-group timing
- `logs/` — Per-test sglang benchmark JSONL output files

### Dependencies

```
sglang
pyyaml
tqdm
# torch omitted — supports custom accelerator builds (MUSA, ROCm, CANN)
```

### Workload Matrix Categories

The golden matrix (`optimal_sets.yaml`) covers:
- **Short**: Small I/O (64/64), high concurrency stress test
- **Normal** (Light→Massive): Increasing context lengths (256→131072 input), realistic generation
- **Prefill_Only** (Light→Massive): Output=1, isolates prefill/TTFT performance
- **Decode_Only** (Light→Medium): Input=1, isolates decode/TPOT performance

### Code Conventions

- Python 3, no type annotations enforced
- `tqdm.write()` for output inside progress bars (not `print()`)
- CSV headers use underscore naming; final report headers use human-readable names
- Engine abstraction via ABC (`BaseEngine`); only `SGLangEngine` implemented currently
- YAML `extra_args` can be string (space-split) or list

### Current Limitations / TODOs

- Only SGLang backend implemented; vLLM and TRT-LLM planned via engine abstraction
- `configs/backends/`, `configs/env/`, `configs/models/` directories exist but are empty (planned)
- PD disaggregation (separate prefill/decode nodes) not yet implemented
- `get_backend_info()` in `collect_env.py` for nvcc/mcc/hipcc parsing still marked TODO
- No automated visualization or CI regression detection yet
