import json
from pathlib import Path


def load_gpu_specs():
    """Load the GPU specifications lookup table."""
    specs_path = Path(__file__).parent / "gpu_specs.json"
    with open(specs_path, "r") as f:
        data = json.load(f)
    # Filter out comment/meta keys
    return {k: v for k, v in data.items() if not k.startswith("_")}


def match_gpu_spec(gpu_name: str, specs: dict = None):
    """
    Match a GPU name string (from nvidia-smi) to a spec entry.
    Uses substring matching — nvidia-smi may report slightly different names
    like 'NVIDIA H100 80GB HBM3' vs 'H100'.

    Returns the spec dict or None if no match found.
    """
    if specs is None:
        specs = load_gpu_specs()

    gpu_name_lower = gpu_name.lower().strip()

    # 1. Exact match
    if gpu_name in specs:
        return specs[gpu_name]

    # 2. Substring match — collect all candidates, pick the best one.
    #    "Best" = the spec key that shares the most characters with the query.
    #    e.g. "NVIDIA H100 80GB" should match "NVIDIA H100 80GB HBM3"
    #    over "NVIDIA H100 SXM5 80GB" because more of the query overlaps.
    candidates = []
    for spec_name, spec_data in specs.items():
        spec_lower = spec_name.lower()
        if spec_lower in gpu_name_lower or gpu_name_lower in spec_lower:
            # Score: length of the overlapping part (longer overlap = better match)
            overlap = len(gpu_name_lower) if gpu_name_lower in spec_lower else len(spec_lower)
            candidates.append((overlap, spec_name, spec_data))

    if candidates:
        # Highest overlap first; on tie, shortest spec name (most specific)
        candidates.sort(key=lambda x: (-x[0], len(x[1])))
        return candidates[0][2]

    # 3. Fuzzy: extract core identifier (e.g., "H100", "A100", "MI300X")
    import re
    core_patterns = [
        r"(B\d{3})",       # B200, B300
        r"(H\d{3})",       # H100, H200
        r"(A\d{3})",       # A100, A800
        r"(L\d{2}S?)",     # L40, L40S, L20
        r"(MI\d{3}X?)",    # MI300X, MI325X, MI355X
        r"(RTX\s*\d{4})",  # RTX 4090, RTX 3090
        r"(MTT\s*S\d+)",   # MTT S4000
    ]
    for pattern in core_patterns:
        m = re.search(pattern, gpu_name, re.IGNORECASE)
        if m:
            core = m.group(1).lower().replace(" ", "")
            for spec_name, spec_data in specs.items():
                if core in spec_name.lower().replace(" ", ""):
                    return spec_data

    return None


def get_model_params(model_path: str):
    """
    Extract model parameters from the model's config.json.

    Attempts to read from a local or network-share path.
    Returns a dict with:
        - total_params_b: Total parameters in billions (estimated)
        - dtype_bytes: Bytes per parameter (2 for bf16/fp16, 1 for fp8, 4 for fp32)
        - num_layers: Number of transformer layers
        - hidden_size: Hidden dimension
        - num_attention_heads: Number of attention heads
        - num_key_value_heads: Number of KV heads (for GQA)
        - vocab_size: Vocabulary size
        - is_moe: Whether the model is MoE
        - num_experts: Total experts (MoE only)
        - num_experts_per_tok: Activated experts per token (MoE only)
        - activated_params_b: Activated params in billions (MoE: subset; dense: same as total)

    Returns None if config.json cannot be read.
    """
    config_path = Path(model_path) / "config.json"
    try:
        with open(config_path, "r") as f:
            cfg = json.load(f)
    except (FileNotFoundError, PermissionError, json.JSONDecodeError):
        return None

    hidden_size = cfg.get("hidden_size", 0)
    num_layers = cfg.get("num_hidden_layers", cfg.get("num_layers", 0))
    num_heads = cfg.get("num_attention_heads", 0)
    num_kv_heads = cfg.get("num_key_value_heads", num_heads)
    vocab_size = cfg.get("vocab_size", 0)
    intermediate_size = cfg.get("intermediate_size", hidden_size * 4)

    # Determine dtype bytes
    torch_dtype = cfg.get("torch_dtype", "bfloat16")
    dtype_map = {
        "float32": 4, "fp32": 4,
        "float16": 2, "fp16": 2,
        "bfloat16": 2, "bf16": 2,
        "float8": 1, "fp8": 1, "float8_e4m3fn": 1,
    }
    dtype_bytes = dtype_map.get(torch_dtype, 2)

    # MoE detection
    num_experts = cfg.get("num_local_experts", cfg.get("num_experts", 0))
    num_experts_per_tok = cfg.get("num_experts_per_tok",
                                  cfg.get("num_selected_experts",
                                          cfg.get("top_k", 0)))
    is_moe = num_experts > 1

    # Estimate total parameters
    # Rough formula: embedding + layers × (attention + FFN) + final LM head
    # Attention: 4 × hidden² (Q, K, V, O projections, adjusted for GQA)
    # FFN dense: 3 × hidden × intermediate (gate, up, down for SwiGLU)
    # FFN MoE: num_experts × 3 × hidden × intermediate
    head_dim = hidden_size // num_heads if num_heads else 0
    attn_params = hidden_size * (num_heads + 2 * num_kv_heads) * head_dim if head_dim else 4 * hidden_size * hidden_size

    if is_moe:
        ffn_params_per_expert = 3 * hidden_size * intermediate_size
        ffn_params_total = num_experts * ffn_params_per_expert
        ffn_params_activated = num_experts_per_tok * ffn_params_per_expert
    else:
        ffn_params_total = 3 * hidden_size * intermediate_size
        ffn_params_activated = ffn_params_total

    layer_params_total = attn_params + ffn_params_total
    layer_params_activated = attn_params + ffn_params_activated

    embedding_params = vocab_size * hidden_size
    total_params = embedding_params + num_layers * layer_params_total + embedding_params  # +lm_head ≈ embedding
    activated_params = embedding_params + num_layers * layer_params_activated + embedding_params

    total_params_b = total_params / 1e9
    activated_params_b = activated_params / 1e9

    return {
        "total_params_b": round(total_params_b, 2),
        "activated_params_b": round(activated_params_b, 2),
        "dtype_bytes": dtype_bytes,
        "num_layers": num_layers,
        "hidden_size": hidden_size,
        "num_attention_heads": num_heads,
        "num_key_value_heads": num_kv_heads,
        "vocab_size": vocab_size,
        "intermediate_size": intermediate_size,
        "is_moe": is_moe,
        "num_experts": num_experts,
        "num_experts_per_tok": num_experts_per_tok,
    }
