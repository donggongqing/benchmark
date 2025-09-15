# vLLM 服务部署与性能测试
确保已安装好vLLM，可参考[官方安装指南](https://vllm.readthedocs.io/en/latest/getting_started/installation.html)

## 1、服务部署指南

### 部署命令
```bash
vllm serve /path/to/model \
    --host 0.0.0.0 \
    --port 8000 \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.95 \
    --dtype bfloat16 \
```

### 关键参数解析
| 参数 | 类型 | 默认值 | 说明 |
|------|------|-------|-----|
| `--tensor-parallel-size` | int | 1 | 张量并行度，建议等于GPU数量 |
| `--gpu-memory-utilization` | float | 0.9 | 显存利用率阈值(0.9=90%) |
| `--dtype` | str | auto | 计算精度(auto,bfloat16,float16,float32,half) |
| `--served-model-name` | str | None | 自定义模型名称 |

vllm serve --help 查看全部参数使用方法

## 2、性能测试
提供两个测试脚本，选择适合自己任务的测试脚本即可
vllm_perf_test.sh （无需额外依赖）
vllm_perf_test.py（支持config.json传入测试case，支持自定义输出路径。）


### vllm_perf_test.sh测试脚本
step1：

```
修改vllm_perf_test.sh配置

#修改模型路径(被测模型的路径)
MODEL_PATH="/data/models/deepseek-ai/deepseek-r1-distill-qwen-1.5b"

# 输入输出对
IO_PAIRS=(
    "1024 1024"
    "2048 1024"
)

# Concurrency and num-prompts (1-to-1 mapping)
CONCURRENCY_AND_PROMPTS=(
    "1 2"
    "2 4"
    "4 8"
     ...
    "256 512 "
)
```
step2：
```bash:
bash vllm_perf_test.sh
```

### vllm_perf_test.py测试脚本
与上方脚本功能一致，新增支持config.json传入测试case，支持自定义输出路径。


```
python vllm_perf_test.py --config <config.json> --output_dir </data/...>
```
config.json参数说明：
model_path：模型路径
perf_test_cases：测试case，每个case是一个列表，列表元素是一个4个元素的列表，分别是[输入长度，输出长度，并发数，发送请求数量]
```
[
  {
    "model_path": "/data/model/deepseek-r1-distill-llama-70b/",
    "perf_test_cases": [
      [16, 16, 1, 4],
      [32, 32, 2, 4],
      [64, 64, 4, 8]
    ]
  }
]
```
ps：使用dashboard_pipeline的测试脚本参考([查看步骤](./dashborad_pipeline/README.md))



