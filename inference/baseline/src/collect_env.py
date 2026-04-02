import subprocess
import shutil
import json
import platform

def get_nvidia_gpu_info():
    try:
        if not shutil.which("nvidia-smi"):
            return None
        # Query nvidia-smi for gpu name and VRAM
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            universal_newlines=True
        )
        # Extract driver version
        driver_out = subprocess.check_output(
            ["nvidia-smi"], universal_newlines=True
        )
        driver_version = "Unknown"
        for line in driver_out.split('\n'):
            if "Driver Version:" in line:
                parts = line.split()
                # Finding the actual version block, rough parsing:
                for i, part in enumerate(parts):
                    if part == "Version:":
                        driver_version = parts[i+1]
                        break
                break

        gpus = [line.strip() for line in output.strip().split('\n') if line.strip()]
        return {"driver_version": driver_version, "gpus": gpus}
    except Exception as e:
        print(f"Error querying nvidia-smi: {e}")
        return None

def get_framework_version(framework_name):
    try:
        output = subprocess.check_output(
            ["pip", "show", framework_name],
            universal_newlines=True
        )
        for line in output.split('\n'):
            if line.startswith("Version:"):
                return line.split(":")[1].strip()
    except Exception:
        pass
    return "Not Installed"

def get_cpu_info():
    try:
        output = subprocess.check_output("cat /proc/cpuinfo | grep 'model name' | head -n 1", shell=True, universal_newlines=True)
        if ":" in output:
            return output.split(":")[1].strip()
    except Exception:
        pass
    return platform.machine()

def get_backend_info():
    try:
        if shutil.which("nvcc"):
            out = subprocess.check_output(["nvcc", "--version"], universal_newlines=True)
            for line in out.split('\n'):
                if "release" in line.lower():
                    # Usually looks like: Cuda compilation tools, release 12.1, V12.1.105
                    return "CUDA " + line.split("release")[1].strip().split(",")[0]
        if shutil.which("mcc"):
            out = subprocess.check_output(["mcc", "--version"], universal_newlines=True)
            return "MUSA " + out.split('\n')[0].strip()
        if shutil.which("hipcc"):
            out = subprocess.check_output(["hipcc", "--version"], universal_newlines=True)
            return "ROCm " + out.split('\n')[0].strip()
    except Exception:
        pass
    return "Unknown Backend"

def collect_environment():
    info = {
        "os": platform.system(),
        "os_release": platform.release(),
        "architecture": platform.machine(),
        "cpu_model": get_cpu_info(),
        "hardware": {},
        "software": {}
    }

    # 1. Hardware Details
    nvidia_info = get_nvidia_gpu_info()
    if nvidia_info:
        info["hardware"]["vendor"] = "NVIDIA"
        info["hardware"]["details"] = nvidia_info
    else:
        info["hardware"]["vendor"] = "Wait_For_Extension (MUSA/ROCm/CANN)"

    # 2. Software Details
    info["software"]["sglang"] = get_framework_version("sglang")
    info["software"]["backend"] = get_backend_info()
    info["software"]["vllm"] = get_framework_version("vllm")
    
    return info

if __name__ == "__main__":
    env_info = collect_environment()
    print(json.dumps(env_info, indent=4))
