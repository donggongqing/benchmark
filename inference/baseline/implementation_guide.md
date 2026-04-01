# Step-by-Step Implementation Guide

## Phase 1: Environment & Orchestration Skeleton (Current Focus)
1. **Directory Setup**: Scaffold the folder structures for configs/ and src/.
2. **Environment Collector (collect_env.py)**: Implement hardware/software detection.
    - Start with NVIDIA (
vidia-smi parser).
    - Add SGLang version check.
3. **Abstract Engine (engine.py)**: Define the standard interface for starting and stopping a backend server.
4. **SGLang Engine (ackend_sgl.py)**: Implement the NVIDIA+SGLang launch logic.

## Phase 2: The Benchmark Client
1. **Benchmark Executor (enchmark.py)**: Write Python wrappers that execute python -m sglang.bench_serving.
2. **Output Parser**: Capture the results (TTFT, TPOT, Throughput, etc.) and write them to a master CSV file.

## Phase 3: The Orchestrator
1. **Main Loop (orchestrator.py)**: Parse the workload YAML configurations (I/O pairs, concurrency).
2. **Error Handling**: Add 	ry/except blocks. If an OOM exception or a timeout is raised by the enchmark.py, restart the Engine cleanly and proceed to the next config.

## Phase 4: Advanced Features (Later)
1. Add vLLM Engine.
2. Add AMD (ROCm), MT (MUSA), Huawei (CANN) system parsers.
3. Add Prefill-Decode Disaggregation network topology setups.
