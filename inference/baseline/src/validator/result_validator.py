"""
Result Validator — checks benchmark results against theoretical GPU+model bounds.

Design philosophy: catch genuinely broken results, not minor deviations.
Real-world performance varies widely due to model architecture (MoE, GQA, MLA),
batching, scheduling, KV cache overhead, etc.  We only flag cases where the
measured value is absurdly far from the theoretical bound — enough to indicate
a measurement bug, misconfiguration, or system failure rather than normal variance.

Theoretical Performance Bounds:
  - TPOT floor:  model_bytes / (mem_bandwidth × tp)
  - TTFT floor:  2 × activated_params × input_tokens / (compute_tflops × tp × 1e12)
  - Throughput ceiling:  mem_bandwidth × tp / model_bytes

Thresholds (intentionally loose):
  - TPOT:  warn if < 0.5× floor (impossibly fast) or > 5× floor at concurrency≤2
  - TTFT:  warn if > 10× floor at concurrency=1 (absurdly slow prefill)
  - Throughput:  warn if > 2× ceiling (impossibly high) or < 1% of ceiling (catastrophic)
  - Concurrency scaling:  warn only on >50% throughput collapse (not gradual drops)
"""

import csv
import json
from pathlib import Path

from .hardware import load_gpu_specs, match_gpu_spec, get_model_params


# ── helpers ──────────────────────────────────────────────────────────────────

def _safe_float(val, default=None):
    try:
        return float(val)
    except (ValueError, TypeError):
        return default


def _model_bytes(params_b, dtype_bytes):
    """Total model weight size in bytes."""
    return params_b * 1e9 * dtype_bytes


# ── helpers for IO-mode detection ────────────────────────────────────────────

def _is_prefill_only(row):
    """Prefill-only test: output=1.  TPOT/output throughput are meaningless."""
    output_len = _safe_float(row.get("output", row.get("output_len")))
    return output_len is not None and output_len <= 1


def _is_decode_only(row):
    """Decode-only test: input=1.  TTFT is trivially small — skip it."""
    input_len = _safe_float(row.get("input", row.get("input_len")))
    return input_len is not None and input_len <= 1


def _get_concurrency(row):
    """Return numeric concurrency, or None if using request-rate mode."""
    val = row.get("concurrency", row.get("max_concurrency"))
    if val in (None, "", "N/A"):
        return None
    return _safe_float(val)


def _get_request_rate(row):
    """Return numeric request rate, or None if using concurrency mode."""
    val = row.get("request_rate")
    if val in (None, "", "N/A"):
        return None
    return _safe_float(val)


# ── single-row checks ───────────────────────────────────────────────────────

def _check_tpot(row, model_bytes, bandwidth_bytes_s, tp):
    """
    TPOT floor = model_bytes / (bandwidth × tp), converted to ms.
    Skipped for prefill-only tests (output≤1) where TPOT is meaningless.

    Two directions:
      - Below 0.5× floor → measurement bug (impossibly fast)
      - Above 5× floor at low concurrency (≤2) → system issue
    """
    if _is_prefill_only(row):
        return None

    mean_tpot = _safe_float(row.get("TPOT (ms) mean", row.get("Mean_TPOT_ms")))
    if mean_tpot is None or mean_tpot <= 0:
        return None

    floor_s = model_bytes / (bandwidth_bytes_s * tp)
    floor_ms = floor_s * 1000

    # Impossibly fast
    if mean_tpot < floor_ms * 0.5:
        return {
            "metric": "TPOT",
            "severity": "warning",
            "message": (
                f"TPOT {mean_tpot:.2f} ms is far below theoretical floor "
                f"{floor_ms:.2f} ms — possible measurement error or wrong dtype assumption."
            ),
            "measured": mean_tpot,
            "theoretical_floor": round(floor_ms, 4),
        }

    # Absurdly slow — only at low concurrency where the bound is tight.
    # In request-rate mode we skip this (no tight concurrency bound).
    concurrency = _get_concurrency(row)
    if concurrency is not None and concurrency <= 2 and mean_tpot > floor_ms * 5:
        ratio = mean_tpot / floor_ms
        return {
            "metric": "TPOT",
            "severity": "warning",
            "message": (
                f"TPOT {mean_tpot:.2f} ms is {ratio:.1f}× the theoretical floor "
                f"{floor_ms:.2f} ms at concurrency={int(concurrency)}. "
                f"Check for system issues (thermal throttling, memory pressure, process contention)."
            ),
            "measured": mean_tpot,
            "theoretical_floor": round(floor_ms, 4),
        }

    return None


def _check_ttft(row, activated_params_b, compute_tflops, tp):
    """
    TTFT floor ≈ 2 × activated_params × input_tokens / (tflops × tp × 1e12).
    Skipped for decode-only tests (input≤1) where TTFT is trivially small.
    Only meaningful at concurrency=1 (no queuing/batching overhead).
    """
    if _is_decode_only(row):
        return None

    mean_ttft = _safe_float(row.get("TTFT (ms) mean", row.get("Mean_TTFT_ms")))
    input_len = _safe_float(row.get("input", row.get("input_len")))
    concurrency = _get_concurrency(row)

    if mean_ttft is None or input_len is None or input_len <= 0:
        return None
    # Only valid at concurrency=1 or request-rate mode with very low rate
    if concurrency is not None and concurrency > 1:
        return None
    if compute_tflops <= 0:
        return None

    flops_needed = 2 * activated_params_b * 1e9 * input_len
    floor_s = flops_needed / (compute_tflops * tp * 1e12)
    floor_ms = floor_s * 1000

    if floor_ms > 0 and mean_ttft > floor_ms * 10:
        ratio = mean_ttft / floor_ms
        return {
            "metric": "TTFT",
            "severity": "warning",
            "message": (
                f"TTFT {mean_ttft:.2f} ms at input_len={int(input_len)} is "
                f"{ratio:.1f}× the theoretical floor {floor_ms:.2f} ms. "
                f"Possible system bottleneck or failed warmup."
            ),
            "measured": mean_ttft,
            "theoretical_floor": round(floor_ms, 4),
        }
    return None


def _check_throughput(row, bandwidth_bytes_s, model_bytes, tp):
    """
    Output throughput ceiling = bandwidth × tp / model_bytes (tok/s).
    Skipped for prefill-only tests (output≤1) where output throughput ≈ QPS.

    Two directions:
      - Above 2× ceiling → impossibly high
      - Below 1% of ceiling → catastrophically low
    """
    if _is_prefill_only(row):
        return None

    output_throughput = _safe_float(
        row.get("output_throughput", row.get("Output_token_throughput_tok_s"))
    )
    if output_throughput is None or output_throughput <= 0:
        return None

    ceiling = bandwidth_bytes_s * tp / model_bytes

    if output_throughput > ceiling * 2.0:
        ratio = output_throughput / ceiling
        return {
            "metric": "output_throughput",
            "severity": "warning",
            "message": (
                f"Output throughput {output_throughput:.1f} tok/s is {ratio:.1f}× "
                f"the theoretical ceiling {ceiling:.1f} tok/s. "
                f"Verify GPU spec or quantization assumptions."
            ),
            "measured": output_throughput,
            "theoretical_ceiling": round(ceiling, 2),
        }

    if output_throughput < ceiling * 0.01:
        pct = output_throughput / ceiling * 100
        return {
            "metric": "output_throughput",
            "severity": "warning",
            "message": (
                f"Output throughput {output_throughput:.1f} tok/s is only "
                f"{pct:.1f}% of theoretical ceiling {ceiling:.1f} tok/s. "
                f"Possible failed run or severe bottleneck."
            ),
            "measured": output_throughput,
            "theoretical_ceiling": round(ceiling, 2),
        }

    return None


# ── trend checks (cross-row) ────────────────────────────────────────────────

def _check_scaling(rows):
    """
    For a fixed (input, output) pair, check throughput scaling across
    concurrency levels OR request rates.  Flag only severe throughput
    collapse (>50% drop) — moderate drops are normal.

    Handles two modes:
      - Concurrency mode: rows have max_concurrency, sorted ascending
      - Request rate mode: rows have request_rate, sorted ascending
    Rows are separated by mode so they don't mix.
    """
    warnings = []
    from collections import defaultdict

    # Separate rows by mode: concurrency vs request-rate
    cc_groups = defaultdict(list)   # keyed by (input, output)
    rr_groups = defaultdict(list)

    for row in rows:
        key = (row.get("input", row.get("input_len", "")),
               row.get("output", row.get("output_len", "")))
        cc = _get_concurrency(row)
        rr = _get_request_rate(row)
        if cc is not None:
            cc_groups[key].append(row)
        elif rr is not None:
            rr_groups[key].append(row)

    # Check concurrency scaling
    for key, group in cc_groups.items():
        sorted_rows = sorted(group, key=lambda r: _get_concurrency(r) or 0)
        for i in range(1, len(sorted_rows)):
            prev_tp = _safe_float(sorted_rows[i - 1].get(
                "total_throughput", sorted_rows[i - 1].get("Total_Token_throughput_tok_s")))
            curr_tp = _safe_float(sorted_rows[i].get(
                "total_throughput", sorted_rows[i].get("Total_Token_throughput_tok_s")))
            prev_cc = _get_concurrency(sorted_rows[i - 1])
            curr_cc = _get_concurrency(sorted_rows[i])

            if prev_tp and curr_tp and prev_tp > 0:
                drop = (prev_tp - curr_tp) / prev_tp
                if drop > 0.50:
                    warnings.append({
                        "metric": "concurrency_scaling",
                        "severity": "warning",
                        "message": (
                            f"IO pair {key}: throughput collapsed {drop:.0%} "
                            f"when concurrency increased from {int(prev_cc)} to {int(curr_cc)}. "
                            f"({prev_tp:.1f} → {curr_tp:.1f} tok/s). "
                            f"Check for OOM, memory pressure, or scheduling failure."
                        ),
                    })

    # Check request-rate scaling
    for key, group in rr_groups.items():
        sorted_rows = sorted(group, key=lambda r: _get_request_rate(r) or 0)
        for i in range(1, len(sorted_rows)):
            prev_tp = _safe_float(sorted_rows[i - 1].get(
                "total_throughput", sorted_rows[i - 1].get("Total_Token_throughput_tok_s")))
            curr_tp = _safe_float(sorted_rows[i].get(
                "total_throughput", sorted_rows[i].get("Total_Token_throughput_tok_s")))
            prev_rr = _get_request_rate(sorted_rows[i - 1])
            curr_rr = _get_request_rate(sorted_rows[i])

            if prev_tp and curr_tp and prev_tp > 0:
                drop = (prev_tp - curr_tp) / prev_tp
                if drop > 0.50:
                    warnings.append({
                        "metric": "request_rate_scaling",
                        "severity": "warning",
                        "message": (
                            f"IO pair {key}: throughput collapsed {drop:.0%} "
                            f"when request rate increased from {prev_rr} to {curr_rr} qps. "
                            f"({prev_tp:.1f} → {curr_tp:.1f} tok/s). "
                            f"Server may be overloaded at this rate."
                        ),
                    })

    return warnings


# ── public API ───────────────────────────────────────────────────────────────

class ResultValidator:
    """
    Validates benchmark results against theoretical hardware bounds.

    Parameters
    ----------
    gpu_name : str
        GPU name string (as reported by nvidia-smi or env_info).
    model_path : str
        Path to model directory containing config.json.
    tp : int
        Tensor parallelism degree.
    dtype_override : str, optional
        Override model dtype for weight-size computation.
        One of "fp16", "bf16", "fp8", "fp32".  If None, reads from model config.
    """

    def __init__(self, gpu_name: str, model_path: str, tp: int = 1,
                 dtype_override: str = None):
        self.gpu_name = gpu_name
        self.tp = tp

        # Load GPU specs
        specs = load_gpu_specs()
        self.gpu_spec = match_gpu_spec(gpu_name, specs)

        # Load model params
        self.model_params = get_model_params(model_path)

        # Apply dtype override
        if dtype_override and self.model_params:
            dtype_map = {"fp32": 4, "fp16": 2, "bf16": 2, "fp8": 1}
            if dtype_override in dtype_map:
                self.model_params["dtype_bytes"] = dtype_map[dtype_override]

        # Pre-compute derived values
        self._model_bytes = None
        self._bandwidth_bytes_s = None
        self._compute_tflops = None

        if self.gpu_spec and self.model_params:
            self._model_bytes = _model_bytes(
                self.model_params["total_params_b"],
                self.model_params["dtype_bytes"],
            )
            self._bandwidth_bytes_s = self.gpu_spec["mem_bandwidth_tb_s"] * 1e12
            # Use bf16 TFLOPS as default compute reference
            self._compute_tflops = self.gpu_spec.get("bf16_tflops", 0)

    @property
    def ready(self):
        """True if both GPU spec and model params are available."""
        return self._model_bytes is not None

    def validate_row(self, row: dict) -> list:
        """Check a single result row. Returns list of warning dicts."""
        if not self.ready:
            return []

        warnings = []
        for check in [
            lambda: _check_tpot(row, self._model_bytes,
                                self._bandwidth_bytes_s, self.tp),
            lambda: _check_ttft(row, self.model_params["activated_params_b"],
                                self._compute_tflops, self.tp),
            lambda: _check_throughput(row, self._bandwidth_bytes_s,
                                      self._model_bytes, self.tp),
        ]:
            result = check()
            if result:
                warnings.append(result)
        return warnings

    def validate_report(self, csv_path: str) -> dict:
        """
        Validate an entire final report CSV. Returns a summary dict:
            {
                "csv_path": str,
                "gpu": str,
                "model_params_b": float,
                "tp": int,
                "ready": bool,
                "row_warnings": [{row_index, ...warning_dict}],
                "trend_warnings": [...],
            }
        """
        result = {
            "csv_path": str(csv_path),
            "gpu": self.gpu_name,
            "model_params_b": self.model_params["total_params_b"] if self.model_params else None,
            "tp": self.tp,
            "ready": self.ready,
            "row_warnings": [],
            "trend_warnings": [],
        }

        if not self.ready:
            result["row_warnings"].append({
                "row_index": 0,
                "metric": "setup",
                "severity": "error",
                "message": (
                    f"Cannot validate: GPU spec {'missing' if not self.gpu_spec else 'OK'}, "
                    f"model params {'missing' if not self.model_params else 'OK'}."
                ),
            })
            return result

        rows = []
        try:
            with open(csv_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for idx, row in enumerate(reader, start=1):
                    rows.append(row)
                    for w in self.validate_row(row):
                        w["row_index"] = idx
                        result["row_warnings"].append(w)
        except FileNotFoundError:
            result["row_warnings"].append({
                "row_index": 0,
                "metric": "file",
                "severity": "error",
                "message": f"CSV file not found: {csv_path}",
            })
            return result

        # Cross-row trend checks
        result["trend_warnings"] = _check_scaling(rows)

        return result

    def save_validation_report(self, csv_path: str, output_dir: str = None):
        """
        Run validation and write results to validation_warnings.json
        in the same directory as the CSV (or in output_dir if specified).
        """
        report = self.validate_report(csv_path)

        if output_dir:
            out_path = Path(output_dir) / "validation_warnings.json"
        else:
            out_path = Path(csv_path).parent / "validation_warnings.json"

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        return report, str(out_path)


# ── convenience function ────────────────────────────────────────────────────

def validate_results(csv_path: str, model_path: str, gpu_name: str,
                     tp: int = 1, dtype_override: str = None,
                     output_dir: str = None) -> tuple:
    """
    One-call convenience: validate a final report CSV and save the report.

    Returns (report_dict, output_path).
    """
    v = ResultValidator(gpu_name, model_path, tp, dtype_override)
    report, path = v.save_validation_report(csv_path, output_dir)
    return report, path
