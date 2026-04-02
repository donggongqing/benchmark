# SGLang Cookbook — Model URL Mapping

Reference for looking up best serve commands per model family.

## Base URL

`https://cookbook.sglang.io/`

## Autoregressive Models

| Model Family | Cookbook URL | Key Serve Flags |
|---|---|---|
| **Qwen3.5** | `https://cookbook.sglang.io/autoregressive/Qwen/Qwen3.5` | `--reasoning-parser qwen3 --tool-call-parser qwen3_coder --speculative-algo NEXTN --enable-flashinfer-allreduce-fusion` |
| **Qwen3** | `https://cookbook.sglang.io/autoregressive/Qwen/Qwen3` | `--reasoning-parser qwen3 --tool-call-parser qwen3_coder` |
| **Qwen3-Next** | `https://cookbook.sglang.io/autoregressive/Qwen/Qwen3-Next` | |
| **Qwen3-VL** | `https://cookbook.sglang.io/autoregressive/Qwen/Qwen3-VL` | |
| **Qwen3-Coder** | `https://cookbook.sglang.io/autoregressive/Qwen/Qwen3-Coder` | |
| **Qwen2.5-VL** | `https://cookbook.sglang.io/autoregressive/Qwen/Qwen2.5-VL` | |
| **DeepSeek-V3.2** | `https://cookbook.sglang.io/autoregressive/DeepSeek/DeepSeek-V3_2` | |
| **DeepSeek-V3.1** | `https://cookbook.sglang.io/autoregressive/DeepSeek/DeepSeek-V3_1` | |
| **DeepSeek-V3** | `https://cookbook.sglang.io/autoregressive/DeepSeek/DeepSeek-V3` | |
| **DeepSeek-R1** | `https://cookbook.sglang.io/autoregressive/DeepSeek/DeepSeek-R1` | |
| **Llama4** | `https://cookbook.sglang.io/autoregressive/Llama/Llama4` | |
| **Llama3.3-70B** | `https://cookbook.sglang.io/autoregressive/Llama/Llama3.3-70B` | |
| **Llama3.1** | `https://cookbook.sglang.io/autoregressive/Llama/Llama3.1` | |
| **GLM-5** | `https://cookbook.sglang.io/autoregressive/GLM/GLM-5` | |
| **GLM-4.7** | `https://cookbook.sglang.io/autoregressive/GLM/GLM-4.7` | |
| **GLM-4.5** | `https://cookbook.sglang.io/autoregressive/GLM/GLM-4.5` | |
| **Kimi-K2.5** | `https://cookbook.sglang.io/autoregressive/Moonshotai/Kimi-K2.5` | |
| **Kimi-K2** | `https://cookbook.sglang.io/autoregressive/Moonshotai/Kimi-K2` | |
| **MiniMax-M2.5** | `https://cookbook.sglang.io/autoregressive/MiniMax/MiniMax-M2.5` | |
| **MiniMax-M2** | `https://cookbook.sglang.io/autoregressive/MiniMax/MiniMax-M2` | |
| **Mistral Small 4** | `https://cookbook.sglang.io/autoregressive/Mistral/Mistral-Small-4` | |
| **Ernie4.5** | `https://cookbook.sglang.io/autoregressive/Ernie/Ernie4.5` | |
| **InternVL3.5** | `https://cookbook.sglang.io/autoregressive/InternVL/InternVL3_5` | |
| **gpt-oss** | `https://cookbook.sglang.io/autoregressive/OpenAI/GPT-OSS` | |

## How to Match Model Path → Cookbook URL

1. Extract the model directory name from the path (e.g., `/data/models/Qwen3.5-9B/` → `Qwen3.5-9B`)
2. Match against known model family patterns (case-insensitive):
   - `qwen3.5` → Qwen3.5
   - `qwen3` (without .5) → Qwen3
   - `deepseek-v3` → DeepSeek-V3
   - `deepseek-r1` → DeepSeek-R1
   - `llama-3.1`, `llama3.1` → Llama3.1
   - `llama-4`, `llama4` → Llama4
   - `glm-4`, `glm4`, `chatglm` → GLM-4.x
   - `internvl` → InternVL
   - `kimi` → Kimi
   - `mistral` → Mistral
   - `ernie` → Ernie
3. Fetch the matched cookbook URL using `fetch_webpage`
4. Parse the "Model Deployment" section for serve commands, TP tables, and model-specific flags

## What to Extract from Cookbook Pages

Each cookbook page follows a standard structure. Key sections:

### Section 3: Model Deployment
- **3.1 Basic Configuration**: Contains an interactive command generator. The page text includes the full `sglang serve` command with all recommended flags.
- **3.2 Configuration Tips**: Lists TP requirements per GPU type (H100/H200/B200/B300/MI300X/MI325X/MI355X) and quantization level (BF16/FP8/FP4). Also model-specific notes like reasoning parsers, speculative decoding, mamba radix cache, etc.

### Key Flags to Extract
- `--tp <N>` — tensor parallelism
- `--reasoning-parser <name>` — for reasoning/thinking models
- `--tool-call-parser <name>` — for tool-calling models
- `--speculative-algo <algo>` — speculative decoding (EAGLE, NEXTN)
- `--speculative-num-steps`, `--speculative-eagle-topk`, `--speculative-num-draft-tokens`
- `--enable-flashinfer-allreduce-fusion` — for NVIDIA GPUs
- `--attention-backend triton` — for AMD GPUs
- `--mem-fraction-static <float>` — GPU memory allocation
- `--mamba-scheduler-strategy <strategy>` — for hybrid architectures
- Environment variables like `SGLANG_USE_CUDA_IPC_TRANSPORT=1`
