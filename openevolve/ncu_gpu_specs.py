"""
GPU spec lookup for the NCU optimizer prompts.

The diagnosis/generation LLM needs peak numbers to judge "is 2% DRAM
throughput low?" — a raw percentage means little without the hardware
context. Runtime-detectable facts (name, SM count, memory size) come from
torch; peak bandwidth / FLOPS come from a small static table keyed by
substring match on the device name. Unknown GPUs simply omit the peak rows —
the prompt formatter skips missing fields.

All numbers are dense (non-sparsity) figures from NVIDIA datasheets.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

# Substring (lowercase) -> static peak specs. First match wins, so more
# specific entries must precede general ones (e.g. "h100 nvl" before "h100").
_STATIC_SPECS: list[tuple[str, Dict[str, Any]]] = [
    ("b200", {"architecture": "Blackwell", "peak_memory_bw_gbps": 8000, "peak_fp16_tflops": 2250}),
    ("gb200", {"architecture": "Blackwell", "peak_memory_bw_gbps": 8000, "peak_fp16_tflops": 2250}),
    ("h200", {"architecture": "Hopper", "peak_memory_bw_gbps": 4800, "peak_fp32_tflops": 67, "peak_fp16_tflops": 989}),
    ("h100 nvl", {"architecture": "Hopper", "peak_memory_bw_gbps": 3900, "peak_fp32_tflops": 60, "peak_fp16_tflops": 835}),
    ("h100 pcie", {"architecture": "Hopper", "peak_memory_bw_gbps": 2000, "peak_fp32_tflops": 51, "peak_fp16_tflops": 756}),
    ("h100", {"architecture": "Hopper", "peak_memory_bw_gbps": 3350, "peak_fp32_tflops": 67, "peak_fp16_tflops": 989}),
    ("a100", {"architecture": "Ampere", "peak_memory_bw_gbps": 2039, "peak_fp32_tflops": 19.5, "peak_fp16_tflops": 312}),
    ("a10g", {"architecture": "Ampere", "peak_memory_bw_gbps": 600, "peak_fp32_tflops": 31, "peak_fp16_tflops": 125}),
    ("l40s", {"architecture": "Ada", "peak_memory_bw_gbps": 864, "peak_fp32_tflops": 91.6, "peak_fp16_tflops": 362}),
    ("rtx 4090", {"architecture": "Ada", "peak_memory_bw_gbps": 1008, "peak_fp32_tflops": 82.6, "peak_fp16_tflops": 330}),
    ("v100", {"architecture": "Volta", "peak_memory_bw_gbps": 900, "peak_fp32_tflops": 15.7, "peak_fp16_tflops": 125}),
]


def get_gpu_specs() -> Optional[Dict[str, Any]]:
    """Detect the current CUDA device and merge runtime + static specs.

    Returns None when torch/CUDA is unavailable (the prompt section is then
    omitted entirely).
    """
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        name = torch.cuda.get_device_name(0)
        props = torch.cuda.get_device_properties(0)
    except Exception:
        return None

    specs: Dict[str, Any] = {
        "name": name,
        "sm_count": props.multi_processor_count,
        "memory_gb": round(props.total_memory / (1 << 30)),
        "compute_capability": f"{props.major}.{props.minor}",
    }

    lowered = name.lower()
    for needle, static in _STATIC_SPECS:
        if needle in lowered:
            specs.update(static)
            break

    return specs


_DISPLAY_FIELDS: list[tuple[str, str, str]] = [
    ("GPU", "name", ""),
    ("Architecture", "architecture", ""),
    ("Compute capability", "compute_capability", ""),
    ("SM count", "sm_count", ""),
    ("Memory size", "memory_gb", " GB"),
    ("Peak memory bandwidth", "peak_memory_bw_gbps", " GB/s"),
    ("Peak FP32", "peak_fp32_tflops", " TFLOPS"),
    ("Peak FP16 (tensor, dense)", "peak_fp16_tflops", " TFLOPS"),
]


def format_gpu_specs(specs: Optional[Dict[str, Any]]) -> str:
    if not specs:
        return "(GPU specs unavailable)"
    lines = []
    for label, key, unit in _DISPLAY_FIELDS:
        value = specs.get(key)
        if value is not None:
            lines.append(f"- {label}: {value}{unit}")
    return "\n".join(lines)
