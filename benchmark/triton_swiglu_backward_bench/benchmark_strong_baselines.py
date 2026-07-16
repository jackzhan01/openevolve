"""Compare evolved SwiGLU autograd-pair programs against strong baselines.

Usage:
    python benchmark_strong_baselines.py evolved_best/best_program.py
    python benchmark_strong_baselines.py  # Liger vs PyTorch only

Baselines:
    - PyTorch autograd (torch_oracle, always)
    - Naive Triton backward (always)
    - Liger SwiGLU (if liger_kernel installed)
    - Candidate evolved program (if provided)
"""

from __future__ import annotations

import importlib.util
import json
import statistics
import sys
import uuid
from typing import Callable

import torch

from backward_naive_triton import swiglu_backward_naive_triton
from task_spec import BENCHMARK_CASES, TestCase, _dtype, torch_oracle
from strong_baselines.liger_swiglu import (
    liger_available,
    make_liger_swiglu_autograd_pair_fns,
)


def _median_ms(fn: Callable, warmup: int = 10, reps: int = 50) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    times = []
    for _ in range(reps):
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(float(start.elapsed_time(end)))
    return float(statistics.median(times))


def _make_inputs(case: TestCase):
    torch.manual_seed(case.rows * 100000 + case.cols)
    dtype = _dtype(torch, case.dtype_name)
    a = torch.randn((case.rows, case.cols), device="cuda", dtype=dtype)
    b = torch.randn((case.rows, case.cols), device="cuda", dtype=dtype)
    dc = torch.randn((case.rows, case.cols), device="cuda", dtype=dtype)
    return dc, a, b


def _load_candidate(path: str):
    name = f"candidate_{uuid.uuid4().hex}"
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    fwd = getattr(mod, "swiglu_forward_with_saved", None)
    bwd = getattr(mod, "swiglu_backward_from_saved", None)
    if fwd is None or bwd is None:
        raise AttributeError("candidate must define swiglu_forward_with_saved and swiglu_backward_from_saved")
    return fwd, bwd


def _bwd_and_full_step_ms(
    fwd_with_saved: Callable,
    bwd_from_saved: Callable,
    dc: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    warmup: int = 10,
    reps: int = 50,
) -> tuple[float, float]:
    """Time backward-only and forward+backward for a (forward_with_saved, backward_from_saved)
    pair. Used identically for the candidate and for Liger so the two sides are measured at the
    same level: bwd-only excludes forward from the timed region for both, full-step times
    forward+backward together for both."""

    def _setup():
        _, saved = fwd_with_saved(a, b)
        return saved

    def _bwd(saved):
        bwd_from_saved(dc, saved)

    for _ in range(warmup):
        s = _setup()
        _bwd(s)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    times = []
    for _ in range(reps):
        s = _setup()
        start.record()
        _bwd(s)
        end.record()
        torch.cuda.synchronize()
        times.append(float(start.elapsed_time(end)))
    bwd_ms = float(statistics.median(times))

    def _full_step():
        _, saved = fwd_with_saved(a, b)
        bwd_from_saved(dc, saved)

    full_ms = _median_ms(_full_step, warmup=warmup, reps=reps)
    return bwd_ms, full_ms


def run_case(case: TestCase, candidate_fwd, candidate_bwd, liger_fwd_saved, liger_bwd_saved):
    dc, a, b = _make_inputs(case)

    pytorch_ms = _median_ms(lambda: torch_oracle(torch, dc, a, b))
    naive_ms = _median_ms(lambda: swiglu_backward_naive_triton(dc, a, b))

    liger_bwd_ms = None
    liger_full_ms = None
    if liger_fwd_saved is not None:
        liger_bwd_ms, liger_full_ms = _bwd_and_full_step_ms(liger_fwd_saved, liger_bwd_saved, dc, a, b)

    cand_bwd_ms = None
    cand_full_ms = None
    if candidate_fwd is not None:
        cand_bwd_ms, cand_full_ms = _bwd_and_full_step_ms(candidate_fwd, candidate_bwd, dc, a, b)

    def _speedup(base, cand):
        if cand is None or base is None:
            return None
        return round(base / max(cand, 1e-9), 4)

    return {
        "shape": [case.rows, case.cols],
        "dtype": case.dtype_name,
        # pytorch_autograd and naive are always forward+backward together (no clean
        # backward-only split available), so they are only comparable to *_full_step_ms.
        "pytorch_autograd_full_step_ms": round(pytorch_ms, 4),
        "naive_triton_full_step_ms": round(naive_ms, 4),
        "liger_bwd_ms": round(liger_bwd_ms, 4) if liger_bwd_ms is not None else None,
        "liger_full_step_ms": round(liger_full_ms, 4) if liger_full_ms is not None else None,
        "candidate_bwd_ms": round(cand_bwd_ms, 4) if cand_bwd_ms is not None else None,
        "candidate_full_step_ms": round(cand_full_ms, 4) if cand_full_ms is not None else None,
        # bwd-only vs bwd-only (both exclude forward from the timed region)
        "cand_speedup_vs_liger_bwd": _speedup(liger_bwd_ms, cand_bwd_ms),
        # full-step vs full-step (both include forward+backward)
        "cand_speedup_vs_liger_full_step": _speedup(liger_full_ms, cand_full_ms),
        "cand_speedup_vs_pytorch_full_step": _speedup(pytorch_ms, cand_full_ms),
        "liger_speedup_vs_pytorch_full_step": _speedup(pytorch_ms, liger_full_ms),
    }


def main(argv):
    if not torch.cuda.is_available():
        print("SKIP: CUDA not available.")
        return 0

    candidate_fwd, candidate_bwd = None, None
    if len(argv) == 2:
        print(f"Loading candidate: {argv[1]}")
        candidate_fwd, candidate_bwd = _load_candidate(argv[1])

    liger_fwd_saved = liger_bwd_saved = None
    if liger_available():
        liger_fwd_saved, liger_bwd_saved = make_liger_swiglu_autograd_pair_fns()
        print("Liger SwiGLU: available")
    else:
        print("Liger SwiGLU: not available")

    results = []
    for case in BENCHMARK_CASES:
        r = run_case(case, candidate_fwd, candidate_bwd, liger_fwd_saved, liger_bwd_saved)
        results.append(r)
        line = (
            f"  [{case.dtype_name:7s}] ({case.rows:5d},{case.cols:6d})  "
            f"pytorch_full={r['pytorch_autograd_full_step_ms']:.3f}ms"
        )
        if r["liger_bwd_ms"] is not None:
            line += (
                f"  liger_bwd={r['liger_bwd_ms']:.3f}ms liger_full={r['liger_full_step_ms']:.3f}ms"
                f" (x{r['liger_speedup_vs_pytorch_full_step']} vs pytorch)"
            )
        if r["candidate_bwd_ms"] is not None:
            line += (
                f"  cand_bwd={r['candidate_bwd_ms']:.3f}ms cand_full={r['candidate_full_step_ms']:.3f}ms"
                f" (vs_liger_bwd x{r['cand_speedup_vs_liger_bwd']}, vs_liger_full x{r['cand_speedup_vs_liger_full_step']})"
            )
        print(line)

    print("\n--- SUMMARY ---")
    if candidate_fwd:
        import math

        def _geomean(key):
            vals = [r[key] for r in results if r[key]]
            if not vals:
                return None
            return math.exp(sum(math.log(v) for v in vals) / len(vals))

        bwd_geomean = _geomean("cand_speedup_vs_liger_bwd")
        full_geomean = _geomean("cand_speedup_vs_liger_full_step")
        if bwd_geomean is not None:
            print(f"Candidate vs Liger backward-only: geomean speedup = {bwd_geomean:.4f}x")
        if full_geomean is not None:
            print(f"Candidate vs Liger full-step: geomean speedup = {full_geomean:.4f}x")

    out_path = "/u/wzhan/tmp/swiglu_baseline_compare.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Full results -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
