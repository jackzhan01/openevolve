"""Per-shape sweep: full-step speedup vs Liger for the best small/large/full programs.

Directly answers "is there a regime crossover?" — plot speedup-vs-Liger as a function of
numel for small-trained vs large-trained; a real regime shows the two curves crossing
(small ahead at low numel, large ahead at high numel). Uniform advantage = no regime.
"""

from __future__ import annotations

import importlib.util
import math
import statistics
import sys
import uuid

import torch

from task_spec import BENCHMARK_CASES, TestCase, _dtype, REGIME_SPLIT
from strong_baselines.liger_swiglu import make_liger_swiglu_autograd_pair_fns

TMP = "/u/wzhan/tmp"
PROGRAMS = {
    "small_r2": f"{TMP}/openevolve_swiglu_small_wg_liger_r2/best/best_program.py",
    "large_r1": f"{TMP}/openevolve_swiglu_large_wg_liger_r1/best/best_program.py",
    "full_r1": f"{TMP}/openevolve_swiglu_full_liger_r1/best/best_program.py",
}


def _median_ms(fn, warmup=10, reps=50):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
    t = []
    for _ in range(reps):
        s.record(); fn(); e.record(); torch.cuda.synchronize(); t.append(float(s.elapsed_time(e)))
    return float(statistics.median(t))


def _inputs(case):
    torch.manual_seed(case.rows * 100000 + case.cols)
    dt = _dtype(torch, case.dtype_name)
    a = torch.randn((case.rows, case.cols), device="cuda", dtype=dt)
    b = torch.randn((case.rows, case.cols), device="cuda", dtype=dt)
    dc = torch.randn((case.rows, case.cols), device="cuda", dtype=dt)
    return dc, a, b


def _full_ms(fwd, bwd, dc, a, b):
    try:
        return _median_ms(lambda: bwd(dc, fwd(a, b)[1]))
    except Exception:
        return float("inf")


def _load(path):
    spec = importlib.util.spec_from_file_location(f"m_{uuid.uuid4().hex}", path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m.swiglu_forward_with_saved, m.swiglu_backward_from_saved


def main():
    if not torch.cuda.is_available():
        print("SKIP: no CUDA"); return 0
    liger_fwd, liger_bwd = make_liger_swiglu_autograd_pair_fns()
    progs = {t: _load(p) for t, p in PROGRAMS.items()}

    shapes = sorted({(c.rows, c.cols) for c in BENCHMARK_CASES}, key=lambda x: x[0] * x[1])
    tags = list(progs)
    print(f"{'shape':>14} {'numel':>9} {'regime':>6} | " + " ".join(f"{t:>9}" for t in tags) + "   (full-step speedup vs Liger, dtype-avg)")
    for (r, c) in shapes:
        phi = r * c
        reg = "small" if phi < REGIME_SPLIT else "large"
        sp = {t: [] for t in tags}
        for dt in ("float32", "float16", "bfloat16"):
            case = TestCase(r, c, dt, 1e-2, 1e-2)
            dc, a, b = _inputs(case)
            lf = _full_ms(liger_fwd, lambda d, s: liger_bwd(d, s), dc, a, b)
            for t, (fwd, bwd) in progs.items():
                cf = _full_ms(fwd, bwd, dc, a, b)
                sp[t].append(lf / cf if math.isfinite(cf) and cf > 0 else float("nan"))
        avg = {t: (sum(v for v in sp[t] if v == v) / max(1, sum(1 for v in sp[t] if v == v))) for t in tags}
        print(f"{str((r,c)):>14} {phi:>9} {reg:>6} | " + " ".join(f"{avg[t]:>9.4f}" for t in tags))

    print("\n(> 1.0 means faster than Liger. A regime crossover = small column highest at "
          "low numel, large column highest at high numel.)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
