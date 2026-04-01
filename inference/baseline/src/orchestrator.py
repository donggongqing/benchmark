import yaml
import subprocess
import time
from .benchmark import run_sglang_benchmark, parse_and_write_results

def run_workload_matrix(engine, csv_writer, workload_yaml: str, host: str, port: int):
    """
    Parses a workload YAML file and runs all IO combinations and concurrency ranges.
    Wraps tests in try/except to gracefully handle OOMing.
    """
    with open(workload_yaml, 'r') as f:
        config = yaml.safe_load(f)

    print(f"📦 Loaded workload configuration: {config.get('name', 'Unknown')}")

    for test_set in config.get('sets', []):
        print(f"\n======== Running Set: {test_set['name']} ========")
        
        io_pairs = test_set.get('io_pairs', [])
        conc_prompts = test_set.get('concurrency_and_prompts', [])

        for input_len, output_len in io_pairs:
            # We iterate through concurrencies. If one OOMs, we skip the rest for this I/O!
            for max_conc, prompts in conc_prompts:
                try:
                    # Execute test client
                    output_text = run_sglang_benchmark(input_len, output_len, max_conc, prompts, host, port)
                    parse_and_write_results(csv_writer, output_text, input_len, output_len, max_conc, prompts)
                except subprocess.CalledProcessError as e:
                    print(f"    ❌ Test Failed (Likely OOM or Timeout). Exit code: {e.returncode}")
                    print(f"    Error Output: {e.output.strip()[-200:]}")
                    
                    # Log failure in CSV
                    failed_row = [input_len, output_len, max_conc, prompts, "FAILED_OOM"] + [""] * 22
                    csv_writer.writerow(failed_row)

                    if engine:
                        print("    🔄 [Local Mode] Restarting Engine to recover from failure...")
                        engine.restart_server()
                    else:
                        print("    ⚠️ [Remote Mode] Assuming remote watchdog will restart server. Waiting 30s...")
                        time.sleep(30)
                        
                    # CRITICAL OPTIMIZATION:
                    # If this concurrency OOMed, higher concurrencies for the *same* I/O length will definitely OOM.
                    # We BREAK the inner concurrency loop, and move straight to the next I/O pair turn!
                    print(f"    ⏭️ Skipping remaining higher concurrencies for I/O {input_len}/{output_len} to save time.")
                    break
