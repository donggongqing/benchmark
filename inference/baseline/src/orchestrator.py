import yaml
import subprocess
import time
import json
from tqdm import tqdm
from pathlib import Path
from .benchmark import run_sglang_benchmark, parse_and_write_results

def run_workload_matrix(engine, csv_writer, workload_yaml: str, host: str, port: int, result_dir, on_step_complete=None):
    """
    Parses a workload YAML file and runs all IO combinations and concurrency ranges.
    Wraps tests in try/except to gracefully handle OOMing.
    Tracks execution time for each test, test group, and total benchmark time.
    """
    logs_dir = Path(result_dir) / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    
    timing_file = Path(result_dir) / "elapsed_time.json"

    with open(workload_yaml, 'r') as f:
        config = yaml.safe_load(f)

    print(f"📦 Loaded workload configuration: {config.get('name', 'Unknown')}")
    
    timing_data = {
        "workload": config.get('name', 'Unknown'),
        "total_time_seconds": 0,
        "groups": []
    }
    

    # Calculate total tests for progress bar
    total_tests = 0
    for test_set in config.get('sets', []):
        io_pairs_len = len(test_set.get('io_pairs', []))
        conc_len = len(test_set.get('concurrency_and_prompts', []))
        req_rate_len = len(test_set.get('request_rate_and_prompts', []))
        total_tests += io_pairs_len * (conc_len + req_rate_len)
    
    pbar = tqdm(total=total_tests, desc="Overall Benchmark Progress", position=0, leave=True)
    
    total_start_time = time.time()

    for test_set in config.get('sets', []):
        group_name = test_set['name']
        tqdm.write(f"\n======== Running Set: {group_name} ========")
        
        group_timing = {
            "group_name": group_name,
            "group_total_seconds": 0,
            "tests": []
        }
        group_start_time = time.time()
        
        io_pairs = test_set.get('io_pairs', [])
        conc_prompts = test_set.get('concurrency_and_prompts', [])

        for input_len, output_len in io_pairs:
            for max_conc, prompts in conc_prompts:
                test_start = time.time()
                test_name = f"IO_{input_len}_{output_len}_Conc_{max_conc}_Prompts_{prompts}"
                pbar.set_postfix({'Latest': test_name, 'Status': 'Run'})
                status = "SUCCESS"
                try:
                    output_text = run_sglang_benchmark(input_len, output_len, max_conc, None, prompts, host, port, logs_dir)
                    parse_and_write_results(csv_writer, output_text, input_len, output_len, max_conc, None, prompts)
                    if on_step_complete: on_step_complete()
                except subprocess.CalledProcessError as e:
                    tqdm.write(f"    ❌ Test Failed (Likely OOM or Timeout). Exit code: {e.returncode}")
                    tqdm.write(f"    Error Output: {e.output.strip()[-200:] if hasattr(e, 'output') and e.output else 'N/A'}")
                    status = "FAILED_OOM"
                    failed_row = [input_len, output_len, max_conc, "N/A", prompts, "FAILED_OOM"] + [""] * 22
                    csv_writer.writerow(failed_row)
                    if on_step_complete: on_step_complete()

                    if engine:
                        tqdm.write("    🔄 [Local Mode] Restarting Engine to recover from failure...")
                        engine.restart_server()
                    else:
                        tqdm.write("    ⚠️ [Remote Mode] Assuming remote watchdog will restart server. Waiting 30s...")
                        time.sleep(30)
                        
                    tqdm.write(f"    ⏭️ Skipping remaining higher concurrencies for I/O {input_len}/{output_len} to save time.")
                    test_end = time.time()
                    group_timing["tests"].append({
                        "test": test_name,
                        "status": status,
                        "time_seconds": round(test_end - test_start, 2)
                    })
                    break
                
                test_end = time.time()
                group_timing["tests"].append({
                    "test": test_name,
                    "status": status,
                    "time_seconds": round(test_end - test_start, 2)
                })
                pbar.update(1)


            # Execute Request Rate tests
            for req_rate, prompts in test_set.get('request_rate_and_prompts', []):
                test_start = time.time()
                test_name = f"IO_{input_len}_{output_len}_QPS_{req_rate}_Prompts_{prompts}"
                pbar.set_postfix({'Latest': test_name, 'Status': 'Run'})
                status = "SUCCESS"
                try:
                    output_text = run_sglang_benchmark(input_len, output_len, None, req_rate, prompts, host, port, logs_dir)
                    parse_and_write_results(csv_writer, output_text, input_len, output_len, None, req_rate, prompts)
                    if on_step_complete: on_step_complete()
                except subprocess.CalledProcessError as e:
                    tqdm.write(f"    ❌ Test Failed (Likely OOM or Timeout). Exit code: {e.returncode}")
                    status = "FAILED_OOM"
                    failed_row = [input_len, output_len, "N/A", req_rate, prompts, "FAILED_OOM"] + [""] * 22
                    csv_writer.writerow(failed_row)
                    if on_step_complete: on_step_complete()

                    if engine:
                        engine.restart_server()
                    else:
                        time.sleep(30)
                        
                    test_end = time.time()
                    group_timing["tests"].append({
                        "test": test_name,
                        "status": status,
                        "time_seconds": round(test_end - test_start, 2)
                    })
                    break
                
                test_end = time.time()
                group_timing["tests"].append({
                    "test": test_name,
                    "status": status,
                    "time_seconds": round(test_end - test_start, 2)
                })
                pbar.update(1)


        group_end = time.time()
        group_timing["group_total_seconds"] = round(group_end - group_start_time, 2)
        timing_data["groups"].append(group_timing)
        
        # Save intermediate timing
        timing_data["total_time_seconds"] = round(time.time() - total_start_time, 2)
        with open(timing_file, "w") as f:
            json.dump(timing_data, f, indent=4)

    total_end = time.time()
    timing_data["total_time_seconds"] = round(total_end - total_start_time, 2)
    
    with open(timing_file, "w") as f:
        json.dump(timing_data, f, indent=4)
        
    pbar.close()
    print(f"⏱️ Timing summary recorded at {timing_file}")

