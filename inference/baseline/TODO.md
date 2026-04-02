# Benchmark Framework TODOs

## Features required for Full Benchmark Report

The final report CSV requires certain system and workload parameters that are currently not supported or reliably tracked by our evaluator.

### Missing Data Points to Implement:
- [x] **`prefill_num`**: Number of cards dedicated to prefill (used in disaggregated architectures).
- [x] **`decode_num`**: Number of cards dedicated to decode (used in disaggregated architectures).
- [x] **`prefill_throughput`**: Need metrics extracted on a per-card basis for prefill architectures.
- [x] **`decode_throughput`**: Need metrics extracted on a per-card basis for decode architectures.
- [x] **`pp` (Pipeline Parallelism)**: Evaluator should accept and track pipeline parallelism degree parameter.
- [x] **`dp` (Data Parallelism)**: Evaluator should accept and track data parallelism degree parameter.
- [x] **`ep` (Expert Parallelism)**: Evaluator should accept MoE parallel parameters.
- [x] **`data_type`**: Evaluator should track the quantization or loaded type of the model (e.g. `fp16`, `bf16`, `fp8`).
- [ ] **`backend`**: Need detailed parsing of `cuda` (nvcc version), `musa`, or `rocm` compiler targets for accurate env tracking.
- [x] **`cpu`**: Require deeper hardware telemetry for exact CPU Models (e.g., parsing `/proc/cpuinfo` instead of just using architecture).
- [x] **`EnvVar`**: Capture a snapshot of injected or active environment variables affecting the run (e.g., `VLLM_ATTENTION_BACKEND`, `CUDA_VISIBLE_DEVICES`).
- [x] **`Cmd`**: Need to seamlessly capture the underlying server startup string/command used to bring up the engine.

### Implementation Guide
1. [x] **Extend `run_eval.py` arguments**: Add CLI arguments for `--pp`, `--dp`, `--ep`, `--data-type`, `--prefill-num`, and `--decode-num`.
2. [x] **Update `SGLangEngine` & Server Startup**: Return the underlying executed command for `Cmd` fields, and add a parameter hook to capture local OS `EnvVars` in use.
3. [ ] **Extend `collect_env.py`**:
   - [x] `get_cpu_info()`: Add linux parser for `cat /proc/cpuinfo | grep 'model name'`.
   - [ ] `get_backend_info()`: Subprocess query `nvcc --version` (for CUDA) or `mcc --version` (for MUSA).
4. [x] **Update `src/report_generator.py`**: Retrieve these placeholders from `args` and `env_info` dictionaries once they are being populated correctly.
## Next Stage Features & Architecture Improvements

### 1. Prefill-Decode (PD) Disaggregation
- [ ] **Dual-Engine Orchestration**: Support deploying and managing separate engine processes for independent Prefill and Decode nodes.
- [ ] **KV-Cache Transfer Metrics**: Track network overhead, bandwidth, and transfer latency for matching KV Cache tensors between PD physical/logical boundaries.
- [ ] **Disaggregated Configurations**: Expand `optimal_sets.yaml` schemas to allow defining independent TP/DP dimensions and hardware allocations specifically for Prefill vs. Decode workers.

### 2. Multi-Framework Generalization
- [ ] **Base Engine Abstraction**: Refactor `SGLangEngine` into a generalized `BaseLLMEngine` class.
- [ ] **vLLM Integration**: Implement `vLLMEngine` to seamlessly swap out the backend for 1-to-1 performance baseline comparisons.
- [ ] **TensorRT-LLM Integration**: Add execution hooks for TRT-LLM deployment (handling MPI/Triton server wrappers) underneath the same evaluation scripts.

### 3. Advanced Workload Synthesis (Real-World Traffic)
- [ ] **Dynamic Traffic Distributions**: Move beyond static fixed-concurrency by introducing Poisson, Gamma, or Burst traffic arrival patterns to simulate complex production environments.
- [ ] **Dataset Replay**: Introduce support for replaying real-world conversation traces (e.g., ShareGPT, LMSYS Chatbot Arena) ensuring input/output tokens and request timings precisely mimic production.

### 4. Automated Visualization & CI/CD Integration
- [ ] **Auto-Plotting**: Build a `src/plot_results.py` tool to automatically generate industry-standard plots (e.g., Latency vs. Throughput curves for TTFT/TPOT) directly off the final CSV reports.
- [ ] **Performance Regression Detection**: Add CI pipeline checks to compare current runs against historical "golden matrices", automatically setting off alerts on throughput degradation or latency spikes.
