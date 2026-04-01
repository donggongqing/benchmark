# LLM Inference Flexible Benchmark Framework Design

## 1. Overview
This document outlines a flexible, modular LLM inference benchmark framework. It is designed to replace monolithic Bash scripts with a robust Python-driven orchestration framework, providing flexibility, fault tolerance, and cross-platform support.

## 2. Supported Matrix
- **Hardware Targets**: NVIDIA (CUDA), AMD (ROCm), Moore Threads (MUSA), Huawei (CANN).
- **Inference Frameworks**: vLLM, SGLang, TRT-LLM.
- **Serving Methods**: Aggregate Serving, Prefill-Decode (PD) Disaggregation.
- **Benchmark Methods**: Normal (Full generation), Prefill-Only (Time to First Token), Decode-Only (Time per Output Token).

## 3. Architecture & Directory Structure

`	ext
inference/baseline/
├── configs/
│   ├── env/             # Topology & Hardware config (tp, dp, pp, ep, pd_mode)
│   ├── models/          # Model paths, dtype, chat-templates
│   ├── backends/        # Specific backend args (vllm, sglang, trtllm)
│   └── workloads/       # I/O pairs and concurrency matrices (YAML)
├── src/
│   ├── collect_env.py   # Handles hw/sw info collection
│   ├── engine.py        # Abstract class for Launching/Terminating inference servers
│   ├── backend_vllm.py  # vLLM implementation of engine.py 
│   ├── backend_sgl.py   # SGLang implementation of engine.py
│   ├── benchmark.py     # Python wrapper to call bench serve or bench_serving
│   └── orchestrator.py  # Main loop: config parsing, error handling, retries
├── exps/
│   ├── 20260401_run1/
│   │   ├── env_info.json
│   │   ├── configs_snapshot.yaml
│   │   ├── server_A.log 
│   │   └── summary.csv
└── run_eval.py          # Framework entry point
`

## 4. Key Modules & Responsibilities

### 4.1. Configuration Design (YAML-based)
Instead of hardcoded logic, YAML configurations are used to dynamically inject benchmark parameters.

**workloads/optimal_sets.yaml** (Example):
`yaml
name: "golden_benchmark_matrix"
sets:
  - name: "Short_IO"
    io_pairs: [[64, 64]]
    concurrency_and_prompts:
      - [16, 64]
      - [128, 512]
  - name: "Decode_Only"
    io_pairs: [[1, 2048], [1, 4096]]
    concurrency_and_prompts:
      - [1, 2]
      - [16, 32]
      - [64, 128]
  - name: "High_Context_PD_Disagg"
    io_pairs: [[8192, 4096], [16384, 4096]]
    concurrency_and_prompts:
      - [1, 2]
      - [16, 32]
`

**env/pd_disagg.yaml** (Example):
`yaml
topology: "pd_disaggregation"
prefill_node:
  host: "192.168.1.10"
  tp: 8
  pp: 1
  extra_args: "--kv-transfer-send"
decode_node:
  host: "192.168.1.11"
  tp: 8
  pp: 1
  extra_args: "--kv-transfer-recv"
`

### 4.2. Environment Collector (collect_env.py)
Automatically detects and captures hardware/software telemetry based on OS/Platform.
- **Hardware Check**: Detects running environments (
vidia-smi, 
ocm-smi, musa-smi, 
pu-smi). Captures GPU names, driver versions, and VRAM.
- **Software Check**: Parses package versions (vllm, sglang) and dependency versions (e.g., NVCC/MUSA toolkit).
- **Artifact**: Exports env_info.json alongside benchmark runs to ensure reproducibility.

### 4.3. The Orchestrator & Fault Tolerance (orchestrator.py)
Handles test execution, metric collection, and OOM/crash recovery. Ensures one failed configuration doesn't crash the entire pipeline.
- Integrates 	ry/except around sub-process benchmark client calls.
- In case of failure (OOM, timeout), captures the failure state (e.g., logging FAILED_OOM), and gracefully restarts the server via the Engine Adapter module.

### 4.4. Engine Adapters (engine.py, backend_vllm.py, backend_sgl.py)
Abstracts starting/stopping the background inference server processes.
- Exposes common APIs like start_server(), stop_server(), 
estart_server().
- Can handle complex start-ups like PD Disaggregation (starting multiple inter-connected processes or issuing remote SSH commands).

## 5. Value Proposition vs. Monolithic Bash
1. **Dynamic Resiliency**: Python orchestrator handles OOM cleanly by gracefully skipping failing tests and restarting the engine, avoiding full termination.
2. **PD Disagg Native**: Decoupling the server launch logic easily allows spinning up distributed nodes before firing client requests.
3. **Hardware Agnostic**: Automatic tooling adaptation guarantees portability between different AI accelerators.
