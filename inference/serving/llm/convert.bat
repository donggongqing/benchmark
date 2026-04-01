@echo off
set MODEL_NAME=%1
set MODEL_ALIAS=%2

uv run --with pandas python convert_data.py ^
    --model %MODEL_NAME% ^
    --model-alias %MODEL_ALIAS% ^
    --data-type BF16 ^
    --driver "NVIDIA-Linux-x86_64" ^
    --driver-version "580.126.09" ^
    --backend cuda ^
    --backend-version 13.0 ^
    --engine cuda ^
    --engine-version 13.0 ^
    --serving vllm ^
    --serving-version 0.10.1.1 ^
    --gpu "H20" ^
    --gpu-num 8 ^
    --tp 8 ^
    --base-dir result_outputs ^
    --source "vllm"
