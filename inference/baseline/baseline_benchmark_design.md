# LLM Inference Baseline Benchmark — Design Document

## 1. Overview

A Python-driven LLM inference benchmark framework that replaces monolithic Bash scripts with modular orchestration, providing fault tolerance, hybrid deployment modes, and cross-platform accelerator support. The framework automates workload matrix execution, metric collection, and Excel-ready report generation.

### Current Implementation Status

| Component | Status | Notes |
|---|---|---|
| SGLang backend | ✅ Implemented | Full lifecycle: launch, benchmark, restart, stop |
| Hybrid mode (Integrated/Separated) | ✅ Implemented | Socket-based auto-detection |
| Workload matrix orchestration | ✅ Implemented | YAML-driven, OOM recovery, timing |
| Environment collection | ✅ Implemented | NVIDIA GPU, CPU, framework versions |
| Final report generation | ✅ Implemented | Excel-ready CSV with computed metrics |
| Server log redirection | ✅ Implemented | Server stdout/stderr → `logs/sglang_server.log` |
| Result validation | ✅ Implemented | Theoretical bounds checking in `src/validator/` |
| vLLM backend | ❌ Planned | Engine ABC ready for extension |
| TRT-LLM backend | ❌ Planned | Engine ABC ready for extension |
| PD disaggregation | ❌ Planned | Dual-engine orchestration needed |
| Multi-accelerator env (ROCm/MUSA/CANN) | 🔶 Partial | `collect_env.py` has stubs, not fully tested |
| Configs: backends/env/models dirs | ❌ Empty | Scaffolded, schemas not defined |

## 2. Supported Matrix

- **Hardware Targets**: NVIDIA (CUDA) ✅, AMD (ROCm) 🔶, Moore Threads (MUSA) 🔶, Huawei (CANN) 🔶.
- **Inference Frameworks**: SGLang ✅, vLLM (planned), TRT-LLM (planned).
- **Serving Methods**: Aggregate Serving ✅, Prefill-Decode (PD) Disaggregation (planned).
- **Benchmark Methods**: Normal (Full generation) ✅, Prefill-Only (TTFT isolation) ✅, Decode-Only (TPOT isolation) ✅.

## 3. Architecture & Directory Structure

```
inference/baseline/
├── run_eval.py              # Entry point — CLI parsing, hybrid mode detection, orchestration
├── run_benchmark.sh         # Shell wrapper with default workload YAML
├── configs/
│   ├── workloads/           # YAML test matrices (active)
│   │   ├── optimal_sets.yaml      # Golden benchmark matrix
│   │   ├── warmup_sets.yaml       # Lightweight warmup workloads
│   │   └── warmup_qps_sets.yaml   # QPS-based warmup workloads
│   ├── backends/            # (Planned) Backend-specific launch args
│   ├── env/                 # (Planned) Hardware topology configs
│   └── models/              # (Planned) Model path/dtype/chat-template configs
├── src/
│   ├── engine.py            # BaseEngine ABC — start_server(), stop_server(), restart_server()
│   ├── backend_sgl.py       # SGLangEngine — subprocess launch, log redirection, OOM restart
│   ├── benchmark.py         # Executes sglang.bench_serving, parses stdout metrics
│   ├── orchestrator.py      # Workload matrix iteration, OOM recovery, progress bar, timing
│   ├── collect_env.py       # GPU/CPU/driver/framework version detection
│   ├── report_generator.py  # Merges raw CSV + env_info → final Excel-ready CSV
│   └── validator/           # Result validation against theoretical bounds
│       ├── __init__.py      # Public API: validate_results(), ResultValidator
│       ├── gpu_specs.json   # Hardware spec lookup (NVIDIA, AMD, Moore Threads)
│       ├── hardware.py      # GPU spec matching + model param extraction
│       └── result_validator.py  # Per-row bounds checks + cross-row trend analysis
├── results/                 # Output: timestamped run directories
│   └── <timestamp>_<model>_tp<N>_<mode>/
│       ├── sgl_bench_*.csv        # Raw per-test metrics
│       ├── final_report_*.csv     # Enriched report for Excel
│       ├── env_info.json          # Hardware/software snapshot
│       ├── elapsed_time.json      # Per-test and per-group timing
│       └── logs/
│           ├── sglang_server.log  # Server stdout/stderr (Integrated mode)
│           └── sglang_*.jsonl     # Per-test benchmark detail logs
├── baseline_benchmark_design.md   # This document
├── implementation_guide.md        # Phase-by-phase implementation plan
├── TODO.md                        # Feature tracking
└── requirements.txt               # sglang, pyyaml, tqdm
```

## 4. Key Modules & Responsibilities

### 4.1. Entry Point — `run_eval.py`

Handles CLI argument parsing with YAML config merging (CLI takes precedence). Implements **hybrid mode detection** via socket probe on `host:port`:

- **Integrated mode**: Port not in use → creates `SGLangEngine`, auto-launches server with log redirection to `results/.../logs/sglang_server.log`, auto-stops on completion.
- **Separated mode**: Port already in use → skips engine creation, acts as benchmark client only, does not manage server lifecycle.

Orchestrates the full pipeline: env collection → CSV setup → engine start → workload matrix → final report.

### 4.2. Configuration Design — YAML Workloads

Workload files define `server_config` and `sets`. Each set contains `io_pairs` crossed with `concurrency_and_prompts` and/or `request_rate_and_prompts`.

**Current `optimal_sets.yaml` structure:**

```yaml
name: golden_benchmark_matrix
server_config:
  model_path: /data/models/Model/
  tp: 8
  host: 0.0.0.0
  port: 30001
  extra_args: --trust-remote-code --disable-radix-cache ...

sets:
  - name: Short                    # 64/64, concurrency 16→512
  - name: Normal_Light             # 256-512 input, full generation
  - name: Normal_Medium            # 1K-4K input
  - name: Normal_Large             # 8K-16K input
  - name: Normal_Extra_Large       # 32K-65K input, 8K-16K with 4K output
  - name: Normal_Massive           # 128K input
  - name: Prefill_Light→Massive    # output=1, isolates TTFT
  - name: Decode_Light→Medium      # input=1, isolates TPOT
```

`extra_args` accepts either a string (space-split) or a list. CLI args `--model`, `--tp`, `--host`, `--port` override YAML values. Any unknown CLI args are appended to `extra_args`.

### 4.3. Engine Abstraction — `engine.py` + `backend_sgl.py`

`BaseEngine` (ABC) defines the interface:
- `start_server()` — Launch inference backend subprocess
- `stop_server()` — Terminate server process (SIGKILL on Linux, taskkill on Windows)
- `restart_server()` — stop + start (used during OOM recovery)

`SGLangEngine` implementation:
- Launches `python -m sglang.launch_server` with model path, TP, and extra args
- Redirects server stdout/stderr to `logs/sglang_server.log` (append mode, survives restarts)
- Captures the full command string in `self.server_cmd` for report `Cmd` field
- Uses 30-second naive wait for startup; checks `process.poll()` for crash detection
- Cross-platform stop: `taskkill /F /T` on Windows, `SIGKILL` on Linux

### 4.4. Benchmark Client — `benchmark.py`

Executes `python -m sglang.bench_serving` against a running server:
- Supports two load modes: `--max-concurrency` (fixed concurrency) or `--request-rate` (QPS)
- Uses `--dataset-name random-ids` with configurable `--random-input-len` / `--random-output-len`
- Saves per-request detail logs as `.jsonl` in `logs/`
- Parses 23 metrics from stdout via regex: successful requests, duration, throughput (request/token), TTFT/TPOT/ITL/E2EL (mean/median/p95/p99)

### 4.5. Orchestrator — `orchestrator.py`

Iterates the full workload matrix with fault tolerance:
- **Progress**: `tqdm` bar with total test count, live postfix showing current test name/status
- **OOM Recovery**: On `subprocess.CalledProcessError`, logs `FAILED_OOM` row, restarts engine (Integrated) or waits 30s (Separated), then **breaks** the inner concurrency loop — skips remaining higher concurrencies for that I/O pair
- **Timing**: Records per-test, per-group, and total elapsed time in `elapsed_time.json`, updated incrementally after each group
- **Report hook**: Calls `on_step_complete()` after each test to flush CSV and regenerate final report

### 4.6. Environment Collector — `collect_env.py`

Detects and captures:
- **GPU**: `nvidia-smi` → GPU name, VRAM, driver version. Stubs for `rocm-smi`, `musa-smi`, `npu-smi` (not yet implemented).
- **CPU**: Parses `/proc/cpuinfo` on Linux, falls back to `platform.machine()`.
- **Backend toolkit**: `nvcc --version` → CUDA version. Stubs for `mcc` (MUSA), `hipcc` (ROCm).
- **Framework versions**: `pip show sglang`, `pip show vllm`.

### 4.7. Report Generator — `report_generator.py`

Transforms raw CSV into an Excel-ready final report:
- Reads raw `sgl_bench_*.csv`, merges with `env_info`, CLI args, and derived metrics
- **Computed fields**: `input_throughput` (total_input_tokens / duration), `tps` (1000 / mean_tpot), `prefill_throughput` (input_tp / prefill_gpu_count), `decode_throughput` (output_tp / decode_gpu_count)
- **Auto-detected fields**: `data_type` from model `config.json` (`torch_dtype`), `pp`/`dp`/`ep` parsed from server command string via regex
- **Captured fields**: `EnvVar` (selected env vars like `CUDA_VISIBLE_DEVICES`), `Cmd` (full server launch command)
- Handles legacy CSV format with missing `request_rate` column (backward compatibility)

## 5. Data Flow

```
                    ┌─────────────────────────────────────────────┐
                    │              run_eval.py                     │
                    │  CLI args + YAML merge → hybrid mode detect  │
                    └──────────┬──────────────────────┬───────────┘
                               │                      │
                    ┌──────────▼──────────┐  ┌────────▼────────────┐
                    │   collect_env.py    │  │   SGLangEngine       │
                    │   → env_info.json   │  │   start/stop/restart │
                    └─────────────────────┘  │   → server.log       │
                                             └────────┬────────────┘
                                                      │
                    ┌─────────────────────────────────▼───────────┐
                    │             orchestrator.py                  │
                    │  for each set → for each io_pair → conc:    │
                    │    benchmark.py → parse → csv_writer         │
                    │    on_step_complete → report_generator       │
                    │    OOM? → restart engine, skip higher conc   │
                    │  → elapsed_time.json                        │
                    └─────────────────────────────────────────────┘
                                        │
                    ┌───────────────────▼─────────────────────────┐
                    │           report_generator.py                │
                    │  raw CSV + env_info + args → final_report    │
                    └─────────────────────────────────────────────┘
```

## 6. Value Proposition vs. Monolithic Bash

1. **Dynamic Resiliency**: OOM/crash per-test isolation — logs failure, restarts engine, skips escalating concurrencies, continues remaining tests.
2. **Hybrid Deployment**: Single script handles both "deploy + bench" and "connect to existing server" via socket probe, zero config change.
3. **Live Reporting**: Final report CSV is regenerated after every test, always up-to-date even if the run is interrupted.
4. **Hardware Agnostic**: Engine ABC + `collect_env.py` stubs enable extension to ROCm/MUSA/CANN without touching orchestration logic.
5. **Reproducibility**: Every run captures `env_info.json`, server command, environment variables, and timestamped results.

---

## 7. Future Design — Planned Extensions

### 7.1. Multi-Backend Support (Priority: High)

The `BaseEngine` ABC is ready. New backends require implementing `start_server()` and `stop_server()`:

**vLLM (`backend_vllm.py`)**:
- Launch via `python -m vllm.entrypoints.openai.api_server`
- `benchmark.py` needs a backend selector — currently hardcodes `--backend sglang`
- Consider abstracting the benchmark client too: `BaseBenchmarkClient` with `SGLangBenchClient` and `VLLMBenchClient`
- vLLM uses different metric output format; `parse_and_write_results()` needs backend-aware parsing

**TRT-LLM (`backend_trtllm.py`)**:
- May involve MPI launch or Triton server wrapper
- Engine compilation step before serving — `start_server()` should handle build + serve
- Consider a `prepare()` method on `BaseEngine` for one-time setup (engine build, model conversion)

**Design recommendation**: Add a `--backend` CLI arg to `run_eval.py`. Use a factory:
```python
BACKENDS = {"sglang": SGLangEngine, "vllm": VLLMEngine, "trtllm": TRTLLMEngine}
engine_cls = BACKENDS[args.backend]
```

### 7.2. Config Directory Schemas (Priority: High)

The empty `configs/backends/`, `configs/env/`, `configs/models/` directories need defined schemas:

**`configs/backends/<backend>.yaml`** — Backend-specific default launch args:
```yaml
# backends/sglang.yaml
engine: sglang
launch_module: sglang.launch_server
bench_module: sglang.bench_serving
default_args:
  --trust-remote-code: true
  --disable-radix-cache: true
  --mem-fraction-static: 0.8
```

**`configs/models/<model>.yaml`** — Model identity and parameters:
```yaml
# models/qwen3-9b.yaml
model_path: /data/models/Qwen3.5-9B/
data_type: bf16
chat_template: qwen3
extra_args: --reasoning-parser qwen3 --tool-call-parser qwen3_coder
```

**`configs/env/<topology>.yaml`** — Hardware and parallelism topology:
```yaml
# env/single_node_8gpu.yaml
topology: aggregate
tp: 8
dp: 1
pp: 1
gpu_ids: "0,1,2,3,4,5,6,7"
```

**Config merge order**: `env/*.yaml` → `models/*.yaml` → `backends/*.yaml` → `workloads/*.yaml` → CLI args (highest precedence).

### 7.3. PD Disaggregation (Priority: Medium)

Requires orchestrating two separate engine instances (prefill node + decode node) that communicate via KV-cache transfer:

```yaml
# env/pd_disagg.yaml
topology: pd_disaggregation
prefill_node:
  host: "192.168.1.10"
  tp: 8
  extra_args: "--kv-transfer-send"
decode_node:
  host: "192.168.1.11"
  tp: 8
  extra_args: "--kv-transfer-recv"
```

**Design considerations**:
- `BaseEngine` may need a `MultiNodeEngine` subclass that manages multiple subprocesses or issues remote SSH commands
- Benchmark client targets the decode node endpoint; prefill node is transparent
- Report needs `prefill_num` / `decode_num` GPU counts and per-role throughput metrics
- The orchestrator's OOM recovery logic must handle restarting both nodes atomically

### 7.4. Server Readiness Detection (Priority: Medium)

Current implementation uses a naive 30-second `time.sleep()`. Improvement options:
- **HTTP health poll**: `GET /health` or `GET /v1/models` endpoint with exponential backoff
- **Log-based detection**: Tail `sglang_server.log` for startup-complete markers (e.g., "Uvicorn running on")
- **Configurable timeout**: `server_config.startup_timeout_s` in YAML, with a reasonable default (120s for large models)

### 7.5. Enhanced Environment Collection (Priority: Medium)

- **Multi-vendor GPU detection**: Add `rocm-smi`, `musa-smi`, `npu-smi` parsers alongside `nvidia-smi`
- **Backend toolkit versions**: Complete `get_backend_info()` — `nvcc`, `mcc`, `hipcc` are stubbed but need error handling and version extraction testing
- **Windows CPU detection**: Current `cat /proc/cpuinfo` fails on Windows — add `wmic cpu get name` or `platform.processor()` fallback
- **Container awareness**: Detect Docker/Kubernetes environment, capture `HOSTNAME`, cgroup limits

### 7.6. Advanced Workload Patterns (Priority: Low)

Beyond fixed concurrency and fixed QPS:
- **Poisson / Gamma arrival**: `request_rate_distribution: poisson` in YAML sets
- **Burst traffic**: Define burst windows with peak/trough QPS
- **Dataset replay**: Support ShareGPT / LMSYS conversation traces with real token length distributions
- **Warmup separation**: Dedicated warmup phase before measured phase (currently uses `--warmup-requests 2` inline)

### 7.7. Visualization & CI Integration (Priority: Low)

- **Auto-plotting** (`src/plot_results.py`): Generate latency-vs-throughput curves, concurrency scaling charts from `final_report_*.csv`
- **Regression detection**: Compare current run against a baseline golden result, flag throughput drops > X% or latency spikes
- **CI pipeline**: GitHub Actions / Jenkins integration — run on merge, publish results as artifacts, post summary to PR

### 7.8. Benchmark Client Abstraction (Priority: Medium)

Currently `benchmark.py` is tightly coupled to `sglang.bench_serving`. For multi-backend support:

```python
class BaseBenchmarkClient(ABC):
    @abstractmethod
    def run(self, input_len, output_len, concurrency, request_rate, num_prompts, host, port, logs_dir) -> str:
        """Execute benchmark and return raw stdout text."""

    @abstractmethod
    def parse_metrics(self, output_text) -> dict:
        """Parse stdout into a metrics dictionary."""

class SGLangBenchClient(BaseBenchmarkClient): ...
class VLLMBenchClient(BaseBenchmarkClient): ...
```

This decouples the orchestrator from any specific benchmark tool's output format.

### 7.9. Result Validation (Priority: High) — ✅ Implemented

Automated sanity-checking of benchmark results against theoretical hardware+model performance bounds. Catches measurement errors, misconfigurations, and anomalous data before results are shared.

**Implementation**: `src/validator/`

```
src/validator/
├── __init__.py           # Public API: validate_results(), ResultValidator, loaders
├── gpu_specs.json        # Hardware spec lookup (~20 GPUs: NVIDIA, AMD, Moore Threads)
├── hardware.py           # GPU spec matching + model param extraction from config.json
└── result_validator.py   # Validation logic: per-row bounds + cross-row trend checks
```

**Theoretical Bounds**:

| Check | Formula | Detects |
|---|---|---|
| TPOT floor | `model_bytes / (mem_bandwidth × tp)` → ms | TPOT impossibly low (measurement bug) |
| TTFT floor | `2 × activated_params × input_tokens / (TFLOPS × tp)` → ms | TTFT below compute limit (concurrency=1 only) |
| Throughput ceiling | `mem_bandwidth × tp / model_bytes` → tok/s | Output throughput exceeding HW capacity |
| Concurrency scaling | Compare throughput across concurrency levels per IO pair | Throughput regression (>15% drop) at higher concurrency |

**Integration Point** (designed, not yet wired):

```python
# In run_eval.py, after generate_final_report():
from src.validator import validate_results
report = validate_results(final_csv_path, model_path, gpu_name, tp)
# Writes validation_report.json alongside final CSV
```

**Output**: `validation_report.json` containing per-row warnings (metric, severity, measured vs theoretical values) and trend warnings. Severities: `error`, `warning`, `info`.

**Adding GPU Specs**: Edit `gpu_specs.json` — each entry needs `vram_gb`, `mem_bandwidth_tb_s`, `bf16_tflops`, `fp8_tflops`, `architecture`. GPU name matching is fuzzy (handles nvidia-smi naming variations).
