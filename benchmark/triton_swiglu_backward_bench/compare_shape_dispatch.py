"""Shape-aware dispatch comparison for SwiGLU autograd-pair programs.

Given the evolved best programs from three training regimes (full / small / large,
each with repeats), this measures every program on the FULL benchmark suite against
Liger and answers:

  1. Do the small/large specialists each win on their own shape regime?
  2. Does {small, large} + a single deployable THRESHOLD beat the single full-trained
     program?  (threshold = best single contiguous cut on regime_feature)
  3. As a ceiling reference, per-shape ORACLE dispatch (best of {small,large} per case).

All timings use the same convention as benchmark_strong_baselines.py: backward-only
excludes the forward from the timed region (for both candidate and Liger); full-step
times forward+backward together.  A program that fails on a shape (e.g. exceeds the
Triton tensor-numel limit) records inf latency there and is simply never dispatched to
that shape.

Usage:
    python compare_shape_dispatch.py            # auto-discovers the 6 run dirs in /u/wzhan/tmp
    python compare_shape_dispatch.py <full_dir> <small_dir> <large_dir> ...  # explicit dirs
"""

from __future__ import annotations

import importlib.util
import json
import math
import statistics
import sys
import uuid
from typing import Callable

import torch

from task_spec import (
    BENCHMARK_CASES,
    TestCase,
    _dtype,
    regime_feature,
    REGIME_SPLIT,
)
from strong_baselines.liger_swiglu import (
    liger_available,
    make_liger_swiglu_autograd_pair_fns,
)

TMP = "/u/wzhan/tmp"
DEFAULT_RUNS = {
    "full_r1": f"{TMP}/openevolve_swiglu_full_liger_r1",
    "full_r2": f"{TMP}/openevolve_swiglu_full_liger_r2",
    "small_r1": f"{TMP}/openevolve_swiglu_small_wg_liger_r1",
    "small_r2": f"{TMP}/openevolve_swiglu_small_wg_liger_r2",
    "large_r1": f"{TMP}/openevolve_swiglu_large_wg_liger_r1",
    "large_r2": f"{TMP}/openevolve_swiglu_large_wg_liger_r2",
}

INF = float("inf")


def _geomean(xs):
    xs = [x for x in xs if x and x > 0 and math.isfinite(x)]
    if not xs:
        return None
    return math.exp(sum(math.log(x) for x in xs) / len(xs))


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


def _bwd_full_ms(fwd, bwd, dc, a, b, warmup: int = 10, reps: int = 50):
    """Return (bwd_only_ms, full_step_ms); (inf, inf) if the program fails on this shape."""
    try:
        def setup():
            _, saved = fwd(a, b)
            return saved

        def do_bwd(saved):
            bwd(dc, saved)

        for _ in range(warmup):
            do_bwd(setup())
        torch.cuda.synchronize()

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        times = []
        for _ in range(reps):
            s = setup()
            start.record()
            do_bwd(s)
            end.record()
            torch.cuda.synchronize()
            times.append(float(start.elapsed_time(end)))
        bwd_ms = float(statistics.median(times))

        def full_step():
            _, saved = fwd(a, b)
            bwd(dc, saved)

        full_ms = _median_ms(full_step, warmup=warmup, reps=reps)
        return bwd_ms, full_ms
    except Exception as exc:  # numel limit, compile failure, etc.
        return INF, INF, repr(exc)[:200]


def _load_program(path: str):
    name = f"cand_{uuid.uuid4().hex}"
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.swiglu_forward_with_saved, mod.swiglu_backward_from_saved


def main(argv):
    if not torch.cuda.is_available():
        print("SKIP: CUDA not available.")
        return 0
    if not liger_available():
        print("ERROR: Liger not available; baseline required.")
        return 1

    runs = dict(DEFAULT_RUNS)
    liger_fwd, liger_bwd = make_liger_swiglu_autograd_pair_fns()

    # Load whichever programs exist.
    progs = {}
    for tag, d in runs.items():
        p = f"{d}/best/best_program.py"
        try:
            progs[tag] = _load_program(p)
            print(f"loaded {tag}: {p}")
        except Exception as exc:
            print(f"MISSING {tag}: {p} ({type(exc).__name__}: {exc})")

    # Measure every loaded program + Liger on every full-suite case.
    # results[tag][case_key] = {"bwd": ms, "full": ms}; also results["liger"][...]
    case_keys = []
    liger_ms = {}
    prog_ms = {tag: {} for tag in progs}
    fails = {tag: [] for tag in progs}

    for case in BENCHMARK_CASES:
        key = (case.rows, case.cols, case.dtype_name)
        case_keys.append((key, case))
        dc, a, b = _make_inputs(case)
        lb, lf = _bwd_full_ms(liger_fwd, lambda d, s: liger_bwd(d, s), dc, a, b)[:2]
        liger_ms[key] = {"bwd": lb, "full": lf}
        for tag, (fwd, bwd) in progs.items():
            out = _bwd_full_ms(fwd, bwd, dc, a, b)
            cb, cf = out[0], out[1]
            prog_ms[tag][key] = {"bwd": cb, "full": cf}
            if not math.isfinite(cf):
                fails[tag].append(key)

    # Per-program geomean full-step speedup vs Liger over cases it survives.
    def prog_geomean_full(tag):
        sp = [liger_ms[k]["full"] / prog_ms[tag][k]["full"]
              for (k, _) in case_keys if math.isfinite(prog_ms[tag][k]["full"])]
        return _geomean(sp)

    print("\n=== per-program geomean full-step speedup vs Liger (over survived cases) ===")
    for tag in progs:
        g = prog_geomean_full(tag)
        nf = len(fails[tag])
        print(f"  {tag:9s} geomean_full_vs_liger={g:.4f}  failed_cases={nf}"
              + (f" {fails[tag]}" if nf else ""))

    # Pick the best repeat per regime by full-suite full-step geomean.
    def best_repeat(prefix):
        cands = [t for t in progs if t.startswith(prefix)]
        cands = [(t, prog_geomean_full(t)) for t in cands]
        cands = [(t, g) for t, g in cands if g is not None]
        return max(cands, key=lambda x: x[1])[0] if cands else None

    full_best = best_repeat("full")
    small_best = best_repeat("small")
    large_best = best_repeat("large")
    print(f"\nbest repeat -> full={full_best} small={small_best} large={large_best}")

    if not (small_best and large_best):
        print("Need both small and large programs for dispatch; abort dispatch analysis.")
        return 0

    # Threshold dispatch: choose the single contiguous cut on regime_feature (numel)
    # that maximizes full-suite geomean full-step speedup vs Liger.
    numels = sorted({r * c for (r, c, _), _ in case_keys})
    # candidate thresholds = geometric midpoints between consecutive numels, plus extremes
    cand_thresholds = [0.0]
    for i in range(len(numels) - 1):
        cand_thresholds.append(math.sqrt(numels[i] * numels[i + 1]))
    cand_thresholds.append(INF)

    def dispatched_geomean(threshold):
        sp = []
        for (k, case) in case_keys:
            phi = case.rows * case.cols
            pick = small_best if phi < threshold else large_best
            cf = prog_ms[pick][k]["full"]
            if math.isfinite(cf):
                sp.append(liger_ms[k]["full"] / cf)
        return _geomean(sp)

    scored = [(t, dispatched_geomean(t)) for t in cand_thresholds]
    scored = [(t, g) for t, g in scored if g is not None]
    best_threshold, best_disp_g = max(scored, key=lambda x: x[1])

    # Oracle: per-case best of {small, large} (ceiling; not restricted to one cut).
    oracle_sp = []
    for (k, _) in case_keys:
        cfs = [prog_ms[t][k]["full"] for t in (small_best, large_best)]
        cfs = [x for x in cfs if math.isfinite(x)]
        if cfs:
            oracle_sp.append(liger_ms[k]["full"] / min(cfs))
    oracle_g = _geomean(oracle_sp)

    full_g = prog_geomean_full(full_best) if full_best else None

    print("\n=== MONEY COMPARISON (geomean full-step speedup vs Liger, full suite) ===")
    if full_g is not None:
        print(f"  full-trained single program ({full_best}) : {full_g:.4f}")
    thr_txt = "small-only" if best_threshold == INF else ("large-only" if best_threshold == 0.0 else f"numel<{best_threshold:.0f}->small else large")
    print(f"  {{small,large}} + threshold dispatch        : {best_disp_g:.4f}   (cut: {thr_txt})")
    print(f"  {{small,large}} + oracle dispatch (ceiling)  : {oracle_g:.4f}")
    print(f"\n  regime rule-of-thumb REGIME_SPLIT (numel)  : {REGIME_SPLIT}")
    print(f"  data-derived best threshold (numel)        : {best_threshold:.0f}")

    # Sanity (i): who wins per regime (small vs large full_ms), by shape.
    print("\n=== per-shape: small-trained vs large-trained full-step (ms), winner ===")
    seen = set()
    for (k, case) in case_keys:
        shape = (case.rows, case.cols)
        if shape in seen:
            continue
        seen.add(shape)
        # average over dtypes for a compact view
        def avg_full(tag):
            vals = [prog_ms[tag][(case.rows, case.cols, dt)]["full"]
                    for dt in ("float32", "float16", "bfloat16")
                    if (case.rows, case.cols, dt) in prog_ms[tag]]
            vals = [v for v in vals if math.isfinite(v)]
            return sum(vals) / len(vals) if vals else INF
        s = avg_full(small_best)
        l = avg_full(large_best)
        winner = "small" if s < l else ("large" if l < s else "tie")
        phi = case.rows * case.cols
        reg = "small" if phi < REGIME_SPLIT else "large"
        s_txt = f"{s:.4f}" if math.isfinite(s) else "FAIL"
        l_txt = f"{l:.4f}" if math.isfinite(l) else "FAIL"
        print(f"  {str(shape):>14} numel={phi:>9} [{reg}]  small={s_txt:>8}  large={l_txt:>8}  -> {winner}")

    out = {
        "full_best": full_best,
        "small_best": small_best,
        "large_best": large_best,
        "full_geomean_vs_liger": full_g,
        "threshold_dispatch_geomean_vs_liger": best_disp_g,
        "oracle_dispatch_geomean_vs_liger": oracle_g,
        "best_threshold_numel": None if not math.isfinite(best_threshold) else best_threshold,
        "regime_split_numel": REGIME_SPLIT,
        "per_program_geomean_full": {t: prog_geomean_full(t) for t in progs},
        "failures": {t: fails[t] for t in progs},
    }
    out_path = f"{TMP}/swiglu_shape_dispatch_compare.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nFull results -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
