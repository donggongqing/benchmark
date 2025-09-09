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
    --served-model-name my_model
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

### vllm_benchmark测试脚本（建议新开一个包含vllm的终端）
step1：

```
修改vllm_benchmark.sh配置

#修改模型路径(被测模型的路径)
MODEL_PATH="/data/models/deepseek-ai/deepseek-r1-distill-qwen-1.5b"


# 修改测试输入输出以及并发度（2~9行）
# Tokens length configuration
INPUT_LIST=(128 256 512 1024)  （输入测试长度）
OUTPUT_LIST=(128) （输出测试长度，可拓展）
# Concurrency settings
CONCURRENCY_LIST=(1 4 8 16 32 64 128) （并发数量）
# Test num prompts
NUM_PROMPTS=(256) （发送请求数量，可拓展）
```
step2：
```bash:
bash vllm_benchmark.sh 
```
### vllm_perf_test测试脚本（建议新开一个包含vllm的终端）
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


## 3、convert.sh脚本使用转成Dashboard数据格式

### 脚本参数设置
```
vi convert.sh

对应参数修改
    --tp 8 \ （张量并行）
    --dp 1 \ （数据并行）
    --pp 1 \ （流水线并行）
    --ep 1 \ （专家并行）
    --data-type fp16 \
    --gpu 'H20' \
    --gpu-num 8 \
    --driver 'NVIDIA-Linux-x86' \
    --driver-version '570.124.06' \
    --backend cuda \
    --backend-version 12.8 \
    --engine cuda \
    --engine-version 12.8 \
    --serving vllm \
    --serving-version 0.7.3 \
    --source 'vllm'
```
### 执行convert.sh
```
bash convert.sh  模型名称（served-model-name）  模型别名
```
example:
```bash
bash convert.sh  deepseek-r1-distill-qwen-1.5b  DeepSeek-R1-Distill-Qwen-1.5B
```
如报错按提示安装缺少的依赖
```
pip install pandas
```
# pipeline示意图
```
graph TB
    A[部署服务]vllm_serve
    A --> B{原始数据}  使用vllm_benchmark.sh脚本
    B --> C[数据转换]  使用convert.sh脚本
    C --> D((Dashboard))
```




