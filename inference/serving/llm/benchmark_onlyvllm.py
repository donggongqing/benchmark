import argparse
from benchmark_utils import LiteLLMService, VLLMService
from typing import Dict, Optional, Tuple, List, Any
from pathlib import Path
import os
import time
import datetime
import subprocess
import platform
import shlex
import asyncio
import signal
import json

# Global script mappings
VLLM_SCRIPTS = {
    'chatglm3-6b'   : 'vllm_chatglm3_6b.sh',
    'llama2-7b'     : 'vllm_llama2_7b.sh',
    'llama2-13b'    : 'vllm_llama2_13b.sh',
    'llama2-70b'    : 'vllm_llama2_70b.sh',
    'llama3-8b'     : 'vllm_llama3_8b.sh',
    'glm-4-9b'      : 'vllm_glm4_9b.sh',
    'qwen2-72b'     : 'vllm_qwen2_72b.sh',
    'qwen2.5-72b-instruct': 'vllm_qwen2.5_72b.sh',
    'qwen2-7b'      : 'vllm_qwen2_7b.sh',
    'openai'        : 'run_vllm_openai.sh',
    'meta-llama-3-70b': 'vllm_llama3_70b.sh',
    'deepseek-r1'   : 'vllm_deepseek.sh',
    'qwq-32b': 'vllm_qwq_32b.sh',
    'deepseek-r1-llama-8b': 'vllm_deepseek_llama_8b.sh',
    'deepseek-r1-llama-70b': 'vllm_deepseek_llama_70b.sh',
    'deepseek-r1-distill-qwen-14b': 'vllm_deepseek_distill_qwen_14b.sh',
    'deepseek-r1-distill-qwen-7b': 'vllm_deepseek_distill_qwen_7b.sh',
    'deepseek-r1-distill-qwen-32b': 'vllm_deepseek_distill_qwen_32b.sh',
    'deepseek-r1-distill-qwen-1.5b': 'vllm_deepseek_distill_qwen_1.5b.sh',
}
PINK = '\033[95m'
RED = '\033[91m'
RESET = '\033[0m'

# Get the absolute path to the script directory
CURRENT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))

class GracefulKiller:
    kill_now = False
    def __init__(self):
        signal.signal(signal.SIGINT, self.exit_gracefully)
        signal.signal(signal.SIGTERM, self.exit_gracefully)

    def exit_gracefully(self, *args):
        self.kill_now = True

async def shutdown_vllm(vllm_service):
    """Gracefully shutdown vLLM service"""
    print("Initiating graceful shutdown of vLLM service...")
    try:
        # Give time for current requests to complete
        await asyncio.sleep(5)
        
        # Get the event loop
        loop = asyncio.get_event_loop()
        
        # Cancel all pending tasks
        tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
        for task in tasks:
            task.cancel()
        
        # Wait for all tasks to complete with a timeout
        await asyncio.gather(*tasks, return_exceptions=True)
        
        # Finally terminate the service
        vllm_service.terminate()
        
    except Exception as e:
        print(f"Error during vLLM shutdown: {e}")
    finally:
        print("vLLM service shutdown complete")

def cleanup_processes(processes):
    """Cleanup benchmark processes"""
    for process in processes:
        if process and process.poll() is None:
            try:
                # Send SIGTERM and wait
                process.terminate()
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                # Force kill if timeout
                try:
                    process.kill()
                    process.wait(timeout=2)
                except:
                    print(f"Failed to kill process {process.pid}")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, choices=list(VLLM_SCRIPTS.keys()))
    parser.add_argument('--device', type=str, required=True, choices=['cpu', 'cuda', 'musa', 'maca'])
    parser.add_argument('--input', type=str, required=True, help="Comma-separated list of input token lengths")
    parser.add_argument('--output', type=str, required=True, help="Comma-separated list of output token lengths")
    parser.add_argument('--concurrent', type=str, required=True, help="Comma-separated list of concurrent request numbers")
    parser.add_argument('--proxy', action='store_true', required=False, help="use --proxy to run benchmark with proxy")
    parser.add_argument('--dry', action='store_true', required=False, help="use --dry to run benchmark without deploy services")
    return parser.parse_args()

def parse_list_arg(arg_str: str) -> List[int]:
    """Parse comma-separated string into list of integers"""
    return [int(x.strip()) for x in arg_str.split(',')]

def generate_workload_combinations(model: str, input_tokens: List[int], output_tokens: List[int], concurrent: List[int]) -> Dict[str, Dict[str, str]]:
    """Generate all possible combinations of parameters for workload"""
    workload = {}
    for inp in input_tokens:
        for out in output_tokens:
            for conc in concurrent:
                task_id = f"{model}_inp{inp}_out{out}_conc{conc}"
                workload[task_id] = {
                    "model": model,
                    "input_tokens": inp,
                    "output_tokens": out,
                    "concurrent": conc,
                    "task": "workload/task.sh"
                }
    return workload


def deploy_services(model: str, device: str, scripts: Dict[str, str]) -> Tuple[Optional[LiteLLMService], Optional[VLLMService]]:
    """
    Deploy services sequentially
    Returns:
        vllm_service or None if either fails
    """
    # 1. Start vLLM first
    print("\nStarting VLLM service...")
    vllm_service = VLLMService(
        model_name=model,
        device=device,
        script_path=scripts[model]
    )
    if not vllm_service.start():
        print("Failed to start VLLM service")
        return None

    return  vllm_service

def run_perf_script(model: str, script: str, save_dir: str, input_tokens: int, output_tokens: int, concurrent: int) -> Optional[subprocess.Popen]:
    """Run performance script as a background process"""
    command = f"bash {script} {model} {save_dir} {input_tokens} {output_tokens} {concurrent}"
    try:
        process = subprocess.Popen(
            command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        print(f"{PINK}Started benchmark process for {model} with PID: {process.pid}{RESET}")
        return process
    except subprocess.SubprocessError as e:
        print(f"Failed to start benchmark process: {e}")
        return None

def run_benchmark(workload: Dict[str, Dict[str, Any]]) -> List[subprocess.Popen]:
    """Run benchmarks as background processes and return list of processes"""
    processes = []
    for task_id, task_info in workload.items():
        time_stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        pth = f"{task_info['model']}_result/{task_id}_{time_stamp}"
        print(f"Running benchmark for {task_info['model']} - {task_id}...")
        process = run_perf_script(
            task_info['model'],
            task_info['task'],
            pth,
            task_info['input_tokens'],
            task_info['output_tokens'],
            task_info['concurrent']
        )
        if process:
            print(f"{PINK}Waiting for benchmark {task_id} to complete...{RESET}")
            process.wait()

            # Capture output
            stdout, stderr = process.communicate()
            time.sleep(15)
            
            # Check if the process completed successfully
            if process.returncode == 0:
                print(f"Benchmark {task_id} completed successfully")
            else:
                print(f"Benchmark {task_id} failed with return code {process.returncode}")
                print("\nProcess stdout:")
                print(stdout)
                print(f"{RED}\nProcess stderr:{RESET}")
                print(stderr)
            
            processes.append(process)

        time.sleep(10)  # Small delay between launches
    return processes

# Define the signal handler function
def signal_handler(sig, frame):
    print("\nReceived interrupt signal. Starting cleanup...")
    cleanup_processes(benchmark_processes)
    print("Cleanup complete. Exiting...")
    exit(0)

def main():
    args = parse_args()
    killer = GracefulKiller()
    global benchmark_processes
    
    scripts = {
        model: str(CURRENT_DIR / args.device / script_name)
        for model, script_name in VLLM_SCRIPTS.items()
    }
    
    litellm_service = None
    vllm_service = None
    benchmark_processes = []
    
    try:
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

        dry_run = args.dry
        proxy = args.proxy

        if dry_run:
            pass
        elif proxy:
            pass
            print("This script is only run vllm, not use proxy")
        else: 
            vllm_service = deploy_services(
                config_name=args.config,
                model=args.model,
                device=args.device,
                scripts=scripts
            )
            
            if not vllm_service:
                print("Failed to deploy services")
                return
              
        # Parse list arguments
        input_tokens = parse_list_arg(args.input)
        output_tokens = parse_list_arg(args.output)
        concurrent_requests = parse_list_arg(args.concurrent)
        
        # Generate workload combinations
        workload = generate_workload_combinations(
            args.model,
            input_tokens,
            output_tokens,
            concurrent_requests
        )

        # Print workload configuration in pink
        print(f"\n{PINK}Workload Configuration:")
        print(json.dumps(workload, indent=2))
        print(f"{RESET}")  # Reset color
        
        if not dry_run:
            print("Waiting for 3 minutes to initilize model ... then start benchmark")
            time.sleep(180)
        else: 
            print("Waiting for 30 seconds to initilize model ... then start benchmark")
            time.sleep(30)

        benchmark_processes = run_benchmark(workload)
        print("Started benchmarks. Waiting for completion...")

         # Register signal handler for SIGINT
        signal.signal(signal.SIGINT, signal_handler)

        # Wait for benchmark processes or interruption
        while any(p.poll() is None for p in benchmark_processes):
            time.sleep(3)
            if killer.kill_now:
                break
            
        print("All benchmarks completed or interrupted. Starting cleanup...")
            
    except KeyboardInterrupt:
        print("\nReceived interrupt signal. Starting graceful shutdown...")
    finally:
        print("Beginning shutdown sequence...")
        
        # First cleanup benchmark processes
        cleanup_processes(benchmark_processes)
        
        proxy = args.proxy
        dry_run = args.dry
        if dry_run:
            pass
        elif proxy:
            pass
        else:
            # handle vLLM shutdown
            if vllm_service:
                # Create new event loop for async shutdown
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    loop.run_until_complete(shutdown_vllm(vllm_service))
                finally:
                    loop.close()     
        print("Shutdown sequence complete")

if __name__ == "__main__":
    main()

