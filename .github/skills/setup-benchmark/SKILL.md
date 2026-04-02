---
name: setup-benchmark
description: "Set up and configure LLM inference baseline benchmarks. USE WHEN: user provides a model path and wants to run benchmarks, needs to create workload YAML configs, wants optimal SGLang/vLLM serve commands, needs to check if a server port is available, wants to configure I/O token pairs and concurrency levels for benchmark runs. Handles: model path configuration, serve command lookup from SGLang cookbook, port availability checking via OpenAI API, workload YAML generation, run script updates."
argument-hint: "Provide the model path and optionally I/O pairs, concurrency, port, tp size"
---

# Setup Benchmark

Configure and prepare LLM inference baseline benchmarks — from model path to ready-to-run state.

## When to Use

- User wants to benchmark a new model and provides a model path
- User needs a workload config with specific I/O token combinations
- User wants the best SGLang serve command for their model + GPU combination
- User needs to check if a serving port is already occupied
- User wants to update `run_benchmark.sh` to point to a new config

## Procedure

### Step 1: Gather Requirements

Collect from the user (ask if not provided):

1. **Model path** (required): Local path to the model weights directory (e.g., `/data/models/Qwen3-8B/`)
2. **I/O pairs** (optional): Input/output token length combinations. Default: use the golden matrix from [optimal_sets.yaml](../../../inference/baseline/configs/workloads/optimal_sets.yaml)
3. **Concurrency levels** (optional): Concurrency × prompt count pairs. Default: `[[1, 4], [16, 64], [64, 256], [128, 512]]`
4. **TP size** (optional): Tensor parallelism degree. Default: infer from model size and GPU
5. **Host / Port** (optional): Server bind address. Default: `0.0.0.0:30000`
6. **Backend** (optional): `sglang` (default) or `vllm`
7. **Multiple models** (optional): User may provide multiple model paths to create batch configs. Handle each model as a separate config file in one pass.

### Step 2: Look Up Best Serve Command from SGLang Cookbook

Identify the model family from the model path name (e.g., `Qwen3.5`, `DeepSeek-R1`, `Llama3.1`).

Fetch the matching cookbook page from [SGLang Cookbook](https://cookbook.sglang.io/). Use the cookbook URL mapping in [cookbook-models.md](./references/cookbook-models.md).

From the cookbook page, extract:
- **Recommended `sglang serve` command** with optimal flags for the model + GPU combination
- **TP requirements** per GPU type and quantization (BF16/FP8/FP4)
- **Model-specific args**: reasoning parser, tool-call parser, speculative decoding, attention backend, etc.
- **Environment variables**: e.g., `SGLANG_USE_CUDA_IPC_TRANSPORT=1`

If no cookbook entry exists, use sensible defaults:
```
sglang serve --model-path <path> --tp <tp> --trust-remote-code --mem-fraction-static 0.8 --host 0.0.0.0 --port 30000
```

**When creating multiple configs**, repeat Steps 2–4 for each model. Each model may have different cookbook flags, TP requirements, and port assignments.

### Step 3: Check Port Availability

Check if the target port already has a running server by calling the OpenAI-compatible API:

```python
# Check via OpenAI API (preferred — confirms model serving, not just port open)
import requests
try:
    resp = requests.get(f"http://{host}:{port}/v1/models", timeout=3)
    if resp.status_code == 200:
        models = resp.json()
        # Port is in use — report the running model
        print(f"Server already running on {host}:{port}")
        print(f"Models: {models}")
        # → Use Separated mode (no engine launch needed)
    else:
        # Port open but not an OpenAI-compatible server
        print(f"Port {port} responds but not an OpenAI API — pick a different port")
except requests.ConnectionError:
    # Port free — use Integrated mode (will auto-launch server)
    print(f"Port {port} is free — will auto-launch server")
```

**Decision logic:**
- **Port free** → Generate config with `server_config` including the full serve command as `extra_args`. The benchmark framework will auto-launch the server (Integrated mode).
- **Port occupied with correct model** → Generate config without `extra_args` or mark as Separated mode. Inform user the server is already running and benchmarks will connect to it directly.
- **Port occupied with wrong model or non-API service** → Suggest a different port (try 30002, 30003, etc.) and re-check.

### Step 4: Generate Workload YAML Config(s)

Create a new YAML file under `inference/baseline/configs/workloads/` named after the model (e.g., `qwen3_8b_benchmark.yaml`).

**For multiple models**, create one YAML per model. Use distinct filenames and assign non-conflicting ports:
- Model 1: `qwen3_8b_benchmark.yaml` → port 30000
- Model 2: `deepseek_r1_benchmark.yaml` → port 30001
- Model 3: `llama3_70b_benchmark.yaml` → port 30002

Use the template structure from [workload-template.yaml](./assets/workload-template.yaml), filling in:
- `name`: descriptive name like `"qwen3_8b_benchmark"`
- `server_config.model_path`: the user's model path
- `server_config.tp`: from cookbook or user input
- `server_config.host` / `server_config.port`: checked port
- `server_config.extra_args`: optimal flags from cookbook lookup
- `sets`: I/O pairs and concurrency combinations

**If user provides specific I/O pairs**, create a focused config with just those pairs:
```yaml
sets:
  - name: Custom
    io_pairs:
      - [<user_input_len>, <user_output_len>]
    concurrency_and_prompts:
      - [1, 4]
      - [16, 64]
      - [64, 256]
```

**If user wants full baseline**, use the golden matrix categories from optimal_sets.yaml:
- Short, Normal (Light→Massive), Prefill_Only (Light→Massive), Decode_Only (Light→Medium)

### Step 5: Update Run Script

Update `inference/baseline/run_benchmark.sh` to point `WORKLOAD_YAML` to the newly created config file:

```bash
WORKLOAD_YAML="configs/workloads/<new_config_name>.yaml"
```

**For multiple configs**, set the first model as the default in `run_benchmark.sh` and list all available configs in the summary so the user can run them individually:
```bash
python run_eval.py --workload configs/workloads/<config_1>.yaml
python run_eval.py --workload configs/workloads/<config_2>.yaml
```

### Step 6: Notify User — Ready to Run

Print a clear summary:

```
✅ Benchmark configuration ready!

📋 Config:     inference/baseline/configs/workloads/<config>.yaml
🚀 Serve cmd:  sglang serve --model-path ... --tp <N> ...
🔌 Endpoint:   <host>:<port> (Integrated / Separated mode)
📊 Workload:   <N> test sets, <M> total test combinations

To run:
  cd inference/baseline
  ./run_benchmark.sh

Or directly:
  python run_eval.py --workload configs/workloads/<config>.yaml
```

If the server is NOT running (Integrated mode), remind the user that the framework will auto-launch and auto-stop the server.

If the server IS already running (Separated mode), remind the user the framework will only run benchmarks and will NOT stop their server.

**For multiple configs**, show a combined summary listing all configs:
```
✅ Benchmark configurations ready! (3 models)

  1. configs/workloads/qwen3_8b_benchmark.yaml       → 0.0.0.0:30000 (Integrated)
  2. configs/workloads/deepseek_r1_benchmark.yaml     → 0.0.0.0:30001 (Integrated)
  3. configs/workloads/llama3_70b_benchmark.yaml      → 0.0.0.0:30002 (Separated)

Run all sequentially:
  cd inference/baseline
  python run_eval.py --workload configs/workloads/qwen3_8b_benchmark.yaml
  python run_eval.py --workload configs/workloads/deepseek_r1_benchmark.yaml
  python run_eval.py --workload configs/workloads/llama3_70b_benchmark.yaml
```

## Important Notes

- The `extra_args` field in YAML can be a string (space-separated) or a list. Prefer string format for readability since serve commands from the cookbook are already formatted that way.
- CLI args `--model`, `--tp`, `--host`, `--port` override YAML `server_config` values at runtime.
- The cookbook uses `sglang serve` (new CLI) but the current framework uses `python -m sglang.launch_server` (legacy). Both work — the framework's `backend_sgl.py` uses the legacy form. Put cookbook-specific flags in `extra_args` (they're compatible).
- Always include `--trust-remote-code` for custom model architectures.
- Always include `--mem-fraction-static 0.8` unless the cookbook recommends otherwise.
