import re
import pandas as pd
import argparse
from datetime import datetime
from pathlib import Path
from typing import List, Optional

# Rename raw vllm_bench column names to the llmperf-style names
# the frontend columnMapping already expects (no value changes)
VLLM_BENCH_RENAME = {
    'input_len':                     'mean_input_tokens',
    'output_len':                    'mean_output_tokens',
    'max_concurrency':               'num_concurrent_requests',
    'num_prompts':                   'results_num_requests_started',
    'Successful_requests':           'results_num_completed_requests',
    'Mean_TTFT_ms':                  'results_ttft_s_mean',
    'Mean_TPOT_ms':                  'results_tpot_s_mean',
    'Mean_ITL_ms':                   'results_inter_token_latency_s_mean',
    'Mean_E2EL_ms':                  'results_end_to_end_latency_s_mean',
    'Output_token_throughput_tok_s': 'results_mean_output_throughput_token_per_s',
    'Total_Token_throughput_tok_s':  'results_request_output_throughput_token_per_s_mean',
    'Request_throughput_req_s':      'results_num_completed_requests_per_min',
}

class DataConverter:
    def __init__(
        self,
        model: str,
        data_type: str,
        driver: str,
        driver_version: str,
        backend: str,
        backend_version: str,
        engine: str,
        engine_version: str,
        serving: str,
        serving_version: str,
        gpu: str,
        gpu_num: int = 1,
        tp: int = 1,
        pp: Optional[int] = None,
        dp: Optional[int] = None,
        ep: Optional[int] = None,
        model_alias: Optional[str] = None,
        base_dir: str = "result_outputs",
        source: str = "in-house_benchmark",
    ):
        self.model = model
        self.model_alias = model_alias or model  # Use alias if provided, otherwise use original model name
        self.extra_info = {
            "model": self.model_alias,  # Use the alias in extra info
            "dataType": data_type,
            "driver": driver,
            "driverVersion": driver_version,
            "backend": backend,
            "backendVersion": backend_version,
            "engine": engine,
            "engineVersion": engine_version,
            "serving": serving,
            "servingVersion": serving_version,
            "gpu": gpu,
            "gpuNum": gpu_num,
            "tp": tp,
            "pp": pp,
            "dp": dp,
            "ep": ep,
            "source": source,
        }
        # Get the directory where convert_data.py is located
        current_dir = Path(__file__).parent
        # Set paths relative to the script location
        self.base_dir = current_dir / base_dir
        # Create output directory in convert_data folder using original model name for path
        self.output_dir = current_dir / "convert_data" / f"{model}_data"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def find_result_files(self) -> List[Path]:
        """Find all CSV result files for the specified model"""
        pattern = f"vllm_bench_{self.model}_*.csv"
        result_files = list(self.base_dir.glob(pattern))
        if not result_files:
            raise FileNotFoundError(
                f"No results files found for model {self.model} in {self.base_dir} "
                f"(pattern: {pattern})"
            )
        return result_files

    def process_summary_file(self, summary_file: Path) -> Optional[pd.DataFrame]:
        """Process a single summary JSON file"""
        try:
            with open(summary_file, 'r') as f:
                data = json.load(f)
            
            # Convert to DataFrame and add extra columns
            df = pd.DataFrame([data])
            
            # Replace model name with alias if it exists in the data
            if 'model' in df.columns:
                df['model'] = self.model_alias

            # Add extra info columns
            for key, value in self.extra_info.items():
                df[key] = value
                
            return df
        except Exception as e:
            print(f"Error processing summary file {summary_file}: {e}")
            return None

    def process_individual_file(self, individual_file: Path, timestamp: int) -> Optional[pd.DataFrame]:
        """Process a single individual responses JSON file"""
        try:
            with open(individual_file, 'r') as f:
                data = json.load(f)
            
            # Convert to DataFrame and add timestamp
            df = pd.DataFrame(data)
            df['timestamp'] = timestamp
            
            return df
        except Exception as e:
            print(f"Error processing individual file {individual_file}: {e}")
            return None

    def convert_all(self):
        """Append metadata columns to raw CSV files and combine them."""
        result_files = self.find_result_files()
        print(f"Found {len(result_files)} result files for model {self.model}")

        dfs = []
        for result_file in result_files:
            df = pd.read_csv(result_file)

            # Rename vllm_bench column names to llmperf-style names the frontend expects
            df = df.rename(columns={k: v for k, v in VLLM_BENCH_RENAME.items() if k in df.columns})

            # Extract timestamp from filename (pattern: _YYYYMMDD_HHMMSS_)
            ts_match = re.search(r'_(\d{8})_(\d{6})_', result_file.name)
            if ts_match:
                d, t = ts_match.group(1), ts_match.group(2)
                df['timestamp'] = f"{d[:4]}-{d[4:6]}-{d[6:8]}T{t[:2]}:{t[2:4]}:{t[4:6]}"
            else:
                df['timestamp'] = datetime.now().isoformat()

            for key, value in self.extra_info.items():
                df[key] = value

            dfs.append(df)
            print(f"Processed {result_file.name}")

        if dfs:
            combined = pd.concat(dfs, ignore_index=True)
            out_file = self.output_dir / f"results_{self.model}_summary.csv"
            combined.to_csv(out_file, index=False)
            print(f"Saved combined results to {out_file}")
        else:
            print("No data to save.")

def parse_args():
    parser = argparse.ArgumentParser(description='Convert benchmark results to CSV format')
    parser.add_argument('--model', type=str, required=True, help='Model name to process')
    parser.add_argument('--model-alias', type=str, help='Alias name for the model (optional)')
    parser.add_argument('--data-type', type=str, required=True, help='Data type')
    parser.add_argument('--driver', type=str, required=True, help='Driver name')
    parser.add_argument('--driver-version', type=str, required=True, help='Driver version')
    parser.add_argument('--backend', type=str, required=True, help='Backend name, e.g. musa, cuda')
    parser.add_argument('--backend-version', type=str, required=True, help='Backend version')
    parser.add_argument('--engine', type=str, required=True, help='Engine name, e.g. mtt, trt-llm')
    parser.add_argument('--engine-version', type=str, required=True, help='Engine version')
    parser.add_argument('--serving', type=str, required=True, help='Serving name, e.g. vllm, triton')
    parser.add_argument('--serving-version', type=str, required=True, help='Serving version')
    parser.add_argument('--gpu', type=str, required=True, help='GPU model, e.g. S4000, A100')
    parser.add_argument('--gpu-num', type=int, required=True, help='Number of GPUs')
    parser.add_argument('--tp', type=int, required=True, help='Tensor Parallelism')
    parser.add_argument('--pp', type=int, help='Pipeline Parallelism')
    parser.add_argument('--dp', type=int, help='Data Parallel')
    parser.add_argument('--ep', type=int, help='Expert Parallel')
    parser.add_argument('--base-dir', type=str, default='result_outputs', 
                       help='Base directory containing result outputs')
    parser.add_argument('--source', type=str, default='in-house_benchmark', 
                       help='Data source')
    return parser.parse_args()

def main():
    args = parse_args()
    
    converter = DataConverter(
        model=args.model,
        data_type=args.data_type,
        driver=args.driver,
        driver_version=args.driver_version,
        backend=args.backend,
        backend_version=args.backend_version,
        engine=args.engine,
        engine_version=args.engine_version,
        serving=args.serving,
        serving_version=args.serving_version,
        gpu=args.gpu,
        gpu_num=args.gpu_num,
        dp=args.dp,
        tp=args.tp,
        pp=args.pp,
        ep=args.ep,
        model_alias=args.model_alias,  # Pass the model alias
        base_dir=args.base_dir,
        source=args.source,
    )
    
    converter.convert_all()

if __name__ == "__main__":
    main()
