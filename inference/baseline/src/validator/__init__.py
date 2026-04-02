"""
Validator subsystem — checks benchmark results against theoretical GPU+model bounds.

Public API:
    validate_results(csv_path, model_path, gpu_name, tp, dtype_override, output_dir)
        → (dict, str) — (validation report, output path), saved as validation_warnings.json

    ResultValidator(gpu_name, model_path, tp, dtype_override)
        → .validate_row(row)     → list of warnings
        → .validate_report(csv)  → full report dict
        → .save_validation_report(csv, output_dir)

    load_gpu_specs()   → dict of GPU name → spec
    match_gpu_spec()   → spec dict for a GPU name string
    get_model_params() → model param dict from config.json

Integrated into run_eval.py after generate_final_report().
Writes validation_warnings.json alongside the final CSV.
"""

from .hardware import load_gpu_specs, match_gpu_spec, get_model_params
from .result_validator import ResultValidator, validate_results

__all__ = [
    "validate_results",
    "ResultValidator",
    "load_gpu_specs",
    "match_gpu_spec",
    "get_model_params",
]
