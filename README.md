# benchmark
服务与离线基准测试套件
## LLM 基准测试结构（vLLM 与 SGLang）

## vLLM 服务与基准测试

本项目提供完整、可脚本化的 vLLM 服务部署与可复现的性能测试流程。主要内容位于 `inference/serving/vLLM/`。

### 前置条件
- 按官方指南安装 vLLM：https://vllm.readthedocs.io/en/latest/getting_started/installation.html
- 准备本地模型文件（HuggingFace 或本地权重）
- 根据显存与硬件调整精度和并行方式

### 目录总览
- `inference/serving/vLLM/` —  vLLM 服务与基准测试相关脚本与配置。
- `inference/serving/sgl/` —  SGLang 服务与基准测试相关脚本与配置。

### 启动vLLM推理服务
```
1) 快速启动脚本：
   - bash inference/serving/vLLM/vllm_serve.sh <MODEL_DIR> <TP>
   - <MODEL_DIR> 为模型目录，<TP> 为张量并行度，取值 [1~8]。

2) 运行基准测试
   - 编辑 inference/serving/vLLM/vllm_perf.sh：
     - 设置 MODEL_PATH 为模型目录，可选配置 TP_NUM、PP_NUM。
     - 配置 IO_PAIRS（输入/输出长度）与 CONCURRENCY_AND_PROMPTS（并发/请求数）。
     - 参考：inference/serving/vLLM/README.md
   - 执行：
     - bash inference/serving/vLLM/vllm_perf.sh
   - 输出：
     - CSV 位于 inference/serving/vLLM/output_result/，命名如 vllm_bench_<model>_tp<tp>pp<pp>_<YYYYMMDD_HHMMSS>_results.csv。
     - 指标包含请求吞吐、输出 Token 吞吐、TTFT/TPOT/ITL/E2E 延迟
```
详细步骤参见：[inference/serving/vLLM/README.md](inference/serving/vLLM/README.md)

## SGLang 服务与基准测试
### 前置条件
- 安装 SGLang 及依赖具体参考官方文档：[安装 SGLang](https://docs.sglang.com.cn/start/install.html)
- 准备本地模型路径（如 HuggingFace 权重目录）。

### 目录总览
- `inference/serving/sgl/sgl_serve.sh` — 通过 `python -m sglang.launch_server` 快速启动服务。
- `inference/serving/sgl/sgl_perf.sh` — 随机数据集的性能基准脚本，输出 CSV。
- `inference/serving/sgl/convert.sh` — 将 SGLang CSV 转为 Dashboard 统一格式。
- `inference/serving/sgl/output_result/` — 示例输出目录。
### 使用方法与vLLM类似：
- 启动服务：`bash inference/serving/sgl/sgl_serve.sh <MODEL_DIR> <TP>`
- 运行基准测试：`bash inference/serving/sgl/sgl_perf.sh`  




