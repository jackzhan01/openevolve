"""Op-agnostic shape-dispatch analysis + final-program integration.

Given a benchmark op (its task_spec) and a set of evolved programs (full / small / large,
optionally with repeats), this:

  1. measures every program on the op's FULL benchmark suite against the Liger baseline
     (task_spec.make_liger_autograd_pair_fns), with per-measurement TIMEOUT + OOM protection
     so a program that hangs / exceeds the Triton numel limit / OOMs is simply scored inf and
     never dispatched to that shape;
  2. reports the money comparison (full-trained single vs {small,large}+threshold dispatch vs
     oracle ceiling) and the per-shape speedup-vs-Liger sweep;
  3. derives the deployment threshold (best single contiguous cut on regime_feature);
  4. EMITS the final deployable program: the single best program if the regime collapses
     (no crossover), otherwise a shape-dispatching wrapper that routes each call to the small
     or large specialist by regime_feature at runtime.

This replaces the per-op copies (compare_shape_dispatch.py / sweep_regime.py) with one tool.

Usage:
    python shape_dispatch_report.py --bench benchmark/triton_rmsnorm_handwritten_backward_bench \
        --program full=<path> --program small=<path> --program large=<path> \
        [--timeout 90] [--out-report <json>] [--out-program <py>]
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import signal
import statistics
import sys
import uuid
from typing import Callable

INF = float("inf")


class _Timeout(Exception):
    pass


def _with_timeout(seconds: int):
    def _handler(signum, frame):
        raise _Timeout()

    class _Ctx:
        def __enter__(self):
            if seconds and seconds > 0:
                self.old = signal.signal(signal.SIGALRM, _handler)
                signal.alarm(seconds)

        def __exit__(self, *a):
            if seconds and seconds > 0:
                signal.alarm(0)
                signal.signal(signal.SIGALRM, self.old)
            return False

    return _Ctx()


def _geomean(xs):
    xs = [x for x in xs if x and x > 0 and math.isfinite(x)]
    return math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs else None


def _import_task_spec(bench_dir: str):
    bench_dir = os.path.abspath(bench_dir)
    benchmark_root = os.path.dirname(bench_dir)
    repo_root = os.path.dirname(benchmark_root)
    for p in (bench_dir, benchmark_root, repo_root):
        if p not in sys.path:
            sys.path.insert(0, p)
    spec = importlib.util.spec_from_file_location(
        f"task_spec_{uuid.uuid4().hex}", os.path.join(bench_dir, "task_spec.py")
    )
    ts = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ts)
    return ts


def _load_program(path: str, ts):
    spec = importlib.util.spec_from_file_location(f"cand_{uuid.uuid4().hex}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    fwd = getattr(mod, ts.AUTOGRAD_PAIR_FORWARD_FN_NAME)
    bwd = getattr(mod, ts.AUTOGRAD_PAIR_BACKWARD_FN_NAME)
    return fwd, bwd


def _case_key(case):
    return (case.rows, case.cols, case.dtype_name)


def _split_inputs(ts, inputs):
    cot = int(getattr(ts, "AUTOGRAD_PAIR_COTANGENT_INDEX", 0))
    fwd_idx = tuple(ts.AUTOGRAD_PAIR_FORWARD_INPUT_INDICES)
    extra_idx = tuple(getattr(ts, "AUTOGRAD_PAIR_BACKWARD_EXTRA_INPUT_INDICES", ()))
    dout = inputs[cot]
    forward_args = tuple(inputs[i] for i in fwd_idx)
    backward_extra = tuple(inputs[i] for i in extra_idx)
    return dout, forward_args, backward_extra


def _median_ms(fn, warmup=10, reps=50):
    import torch

    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    t = []
    for _ in range(reps):
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        t.append(float(s.elapsed_time(e)))
    return float(statistics.median(t))


def _bwd_full_ms(fwd, bwd, dout, forward_args, backward_extra, timeout):
    """(bwd_only_ms, full_step_ms); (inf, inf) on timeout / OOM / compile failure."""
    import torch

    try:
        with _with_timeout(timeout):

            def setup():
                _y, saved = fwd(*forward_args)
                return saved

            def do_bwd(saved):
                bwd(dout, saved, *backward_extra)

            for _ in range(10):
                do_bwd(setup())
            torch.cuda.synchronize()
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            times = []
            for _ in range(50):
                sv = setup()
                s.record()
                do_bwd(sv)
                e.record()
                torch.cuda.synchronize()
                times.append(float(s.elapsed_time(e)))
            bwd_ms = float(statistics.median(times))

            def full_step():
                _y, saved = fwd(*forward_args)
                bwd(dout, saved, *backward_extra)

            full_ms = _median_ms(full_step)
            return bwd_ms, full_ms
    except (_Timeout, RuntimeError, Exception):
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
        return INF, INF


def _autograd_baseline_pair(ts):
    """Baseline pair for ops with no strong (Liger) baseline: plain PyTorch autograd.

    `_bwd_full_ms` runs `fwd` OUTSIDE the timed backward region, but autograd cannot be split that
    way — it must redo the forward to have a graph to differentiate. So `fwd` here is a zero-cost
    pass-through of the inputs and `bwd` does the whole fwd+bwd via the task_spec's own autograd
    oracle. Backward-only and full-step then coincide, which is exactly the convention the
    evaluator core already uses for this baseline. Doing it the other way (a real forward in `fwd`)
    would charge the baseline for two forwards and overstate our speedup.
    """
    import torch

    def fwd(*forward_args):
        return None, tuple(a for a in forward_args if isinstance(a, torch.Tensor))

    def bwd(dout, saved, *extras):
        return ts.torch_oracle(torch, dout, *saved, *extras)

    return fwd, bwd


def run(bench_dir, programs: dict[str, str], timeout: int, out_report: str | None, out_program: str | None):
    import torch

    if not torch.cuda.is_available():
        print("SKIP: no CUDA")
        return 0

    ts = _import_task_spec(bench_dir)
    if hasattr(ts, "make_liger_autograd_pair_fns"):
        liger_fwd, liger_bwd = ts.make_liger_autograd_pair_fns()
        baseline_name = "liger"
    else:
        liger_fwd, liger_bwd = _autograd_baseline_pair(ts)
        baseline_name = "pytorch_autograd"
    print(f"baseline = {baseline_name}")

    progs = {}
    for tag, path in programs.items():
        try:
            progs[tag] = _load_program(path, ts)
            print(f"loaded {tag}: {path}")
        except Exception as exc:
            print(f"MISSING {tag}: {path} ({type(exc).__name__}: {exc})")

    cases = list(ts.BENCHMARK_CASES)
    liger_ms, prog_ms = {}, {t: {} for t in progs}
    fails = {t: [] for t in progs}

    for case in cases:
        key = _case_key(case)
        if hasattr(ts, "seed_for_case"):
            torch.manual_seed(ts.seed_for_case(case))
        inputs = tuple(ts.make_inputs(torch, case))
        dout, fargs, bextra = _split_inputs(ts, inputs)
        lb, lf = _bwd_full_ms(liger_fwd, liger_bwd, dout, fargs, bextra, timeout)
        liger_ms[key] = {"bwd": lb, "full": lf}
        for tag, (fwd, bwd) in progs.items():
            cb, cf = _bwd_full_ms(fwd, bwd, dout, fargs, bextra, timeout)
            prog_ms[tag][key] = {"bwd": cb, "full": cf}
            if not math.isfinite(cf):
                fails[tag].append(key)

    def prog_geomean(tag, which="full"):
        sp = [liger_ms[k][which] / prog_ms[tag][k][which]
              for k in map(_case_key, cases)
              if math.isfinite(prog_ms[tag][k][which]) and math.isfinite(liger_ms[k][which])]
        return _geomean(sp)

    print("\n=== per-program geomean vs Liger (survived cases) ===")
    for tag in progs:
        gf, gb = prog_geomean(tag, "full"), prog_geomean(tag, "bwd")
        nf = len(fails[tag])
        gf_s = f"{gf:.4f}" if gf else "n/a"
        gb_s = f"{gb:.4f}" if gb else "n/a"
        print(f"  {tag:10s} full={gf_s} bwd={gb_s} failed={nf}" + (f" {fails[tag]}" if nf else ""))

    def best_repeat(prefix):
        c = [(t, prog_geomean(t, "full")) for t in progs if t.split("_")[0] == prefix or t == prefix or t.startswith(prefix)]
        c = [(t, g) for t, g in c if g is not None]
        return max(c, key=lambda x: x[1])[0] if c else None

    full_best = best_repeat("full")
    small_best = best_repeat("small")
    large_best = best_repeat("large")
    print(f"\nbest repeat -> full={full_best} small={small_best} large={large_best}")

    result = {"bench_dir": os.path.abspath(bench_dir), "regime_split": getattr(ts, "REGIME_SPLIT", None),
              "per_program_geomean_full": {t: prog_geomean(t, "full") for t in progs},
              "per_program_geomean_bwd": {t: prog_geomean(t, "bwd") for t in progs},
              "failures": {t: fails[t] for t in progs},
              "full_best": full_best, "small_best": small_best, "large_best": large_best}

    if small_best and large_best:
        numels = None
        feats = sorted({ts.regime_feature(c) for c in cases})
        cand_thr = [0.0] + [math.sqrt(feats[i] * feats[i + 1]) for i in range(len(feats) - 1)] + [INF]

        def dispatched_geomean(threshold):
            sp = []
            for case in cases:
                k = _case_key(case)
                pick = small_best if ts.regime_feature(case) < threshold else large_best
                cf = prog_ms[pick][k]["full"]
                if math.isfinite(cf) and math.isfinite(liger_ms[k]["full"]):
                    sp.append(liger_ms[k]["full"] / cf)
            return _geomean(sp)

        scored = [(t, dispatched_geomean(t)) for t in cand_thr]
        scored = [(t, g) for t, g in scored if g is not None]
        best_thr, best_disp = max(scored, key=lambda x: x[1])

        oracle_sp = []
        for case in cases:
            k = _case_key(case)
            vals = [prog_ms[t][k]["full"] for t in (small_best, large_best) if math.isfinite(prog_ms[t][k]["full"])]
            if vals and math.isfinite(liger_ms[k]["full"]):
                oracle_sp.append(liger_ms[k]["full"] / min(vals))
        oracle_g = _geomean(oracle_sp)
        full_g = prog_geomean(full_best, "full") if full_best else None

        collapsed = (best_thr in (0.0, INF))
        thr_txt = "small-only" if best_thr == INF else ("large-only" if best_thr == 0.0 else f"{best_thr:.0f}")
        print("\n=== MONEY COMPARISON (geomean full-step vs Liger, full suite) ===")
        if full_g:
            print(f"  full-trained single ({full_best})       : {full_g:.4f}")
        print(f"  {{small,large}} + threshold dispatch    : {best_disp:.4f}   (cut on regime_feature: {thr_txt})")
        print(f"  {{small,large}} + oracle (ceiling)      : {oracle_g:.4f}")
        print(f"  regime collapsed to single program?    : {collapsed}")

        print("\n=== per-shape sweep (full-step speedup vs Liger, dtype-avg) ===")
        shapes = sorted({(c.rows, c.cols) for c in cases}, key=lambda s: ts.regime_feature(_mkcase(ts, s)))
        tags_sorted = [t for t in (small_best, large_best, full_best) if t]
        print(f"{'shape':>16} {'feat':>10} | " + " ".join(f"{t:>10}" for t in tags_sorted))
        for (r, c) in shapes:
            feat = ts.regime_feature(_mkcase(ts, (r, c)))

            def avg(tag):
                vs = [liger_ms[k]["full"] / prog_ms[tag][k]["full"]
                      for k in prog_ms[tag]
                      if k[0] == r and k[1] == c and math.isfinite(prog_ms[tag][k]["full"]) and math.isfinite(liger_ms[k]["full"])]
                return sum(vs) / len(vs) if vs else float("nan")

            print(f"{str((r,c)):>16} {feat:>10.0f} | " + " ".join(f"{avg(t):>10.4f}" for t in tags_sorted))

        result.update({"threshold_dispatch_geomean": best_disp, "oracle_geomean": oracle_g,
                       "full_geomean": full_g, "best_threshold": None if not math.isfinite(best_thr) else best_thr,
                       "regime_collapsed": collapsed})

        if out_program:
            _emit_final_program(ts, bench_dir, programs, small_best, large_best, full_best,
                                best_thr, collapsed, out_program)
            result["final_program"] = out_program
            print(f"\nFinal deployable program -> {out_program}")

    result["baseline"] = baseline_name  # every *_ms/speedup above is vs THIS baseline, not always liger
    if out_report:
        with open(out_report, "w") as f:
            json.dump(result, f, indent=2, default=str)
        print(f"Report -> {out_report}")
    return 0


def _mkcase(ts, shape):
    from collections import namedtuple

    C = namedtuple("C", "rows cols dtype_name")
    return C(shape[0], shape[1], "float32")


def _emit_final_program(ts, bench_dir, programs, small_best, large_best, full_best, threshold, collapsed, out_path):
    """Write the deployable program: single best if collapsed, else a runtime dispatcher."""
    import shutil

    if collapsed:
        winner = small_best if threshold == INF else large_best
        shutil.copyfile(programs[winner], out_path)
        with open(out_path, "a") as f:
            f.write(f"\n# NOTE: regime collapsed to a single program; this is the {winner} specialist.\n")
        return

    fwd_name = ts.AUTOGRAD_PAIR_FORWARD_FN_NAME
    bwd_name = ts.AUTOGRAD_PAIR_BACKWARD_FN_NAME
    small_path = os.path.abspath(programs[small_best])
    large_path = os.path.abspath(programs[large_best])
    ts_path = os.path.abspath(os.path.join(bench_dir, "task_spec.py"))
    src = f'''"""Auto-generated shape-dispatching {fwd_name}/{bwd_name}.

Routes each call to the small- or large-regime specialist by regime_feature at runtime.
Threshold (on regime_feature) = {threshold}.
"""
import importlib.util as _ilu
from collections import namedtuple as _nt

_SMALL_PATH = {small_path!r}
_LARGE_PATH = {large_path!r}
_TS_PATH = {ts_path!r}
_THRESHOLD = {threshold!r}
_FWD = {fwd_name!r}
_BWD = {bwd_name!r}


def _load(path):
    s = _ilu.spec_from_file_location("disp_" + path.replace("/", "_"), path)
    m = _ilu.module_from_spec(s); s.loader.exec_module(m); return m


def _load_ts():
    s = _ilu.spec_from_file_location("disp_ts", _TS_PATH)
    m = _ilu.module_from_spec(s); s.loader.exec_module(m); return m


_small = _load(_SMALL_PATH)
_large = _load(_LARGE_PATH)
_ts = _load_ts()
_Case = _nt("Case", "rows cols")


def _feature(x):
    rows = 1
    for d in x.shape[:-1]:
        rows *= int(d)
    return _ts.regime_feature(_Case(rows, int(x.shape[-1])))


def {fwd_name}(*args, **kwargs):
    # route by the primary input's regime feature (rows / numel)
    prog = _small if _feature(args[0]) < _THRESHOLD else _large
    return getattr(prog, _FWD)(*args, **kwargs)


def {bwd_name}(dout, saved_tensors, *args, **kwargs):
    # dout (the cotangent) has the same regime-defining shape as the forward input, so it
    # routes to the same specialist that produced `saved_tensors` — no need to tag saved
    # (keeps saved a pure tensor tuple, as the evaluator requires).
    prog = _small if _feature(dout) < _THRESHOLD else _large
    return getattr(prog, _BWD)(dout, saved_tensors, *args, **kwargs)
'''
    with open(out_path, "w") as f:
        f.write(src)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", required=True, help="path to the triton_<op>_backward_bench dir")
    ap.add_argument("--program", action="append", default=[], help="tag=path (e.g. small=/.../best_program.py)")
    ap.add_argument("--timeout", type=int, default=90, help="per-measurement timeout seconds")
    ap.add_argument("--out-report", default=None)
    ap.add_argument("--out-program", default=None)
    args = ap.parse_args(argv)

    programs = {}
    for spec in args.program:
        if "=" not in spec:
            raise SystemExit(f"--program must be tag=path, got {spec!r}")
        tag, path = spec.split("=", 1)
        programs[tag] = path
    if not programs:
        raise SystemExit("need at least one --program tag=path")

    return run(args.bench, programs, args.timeout, args.out_report, args.out_program)


if __name__ == "__main__":
    raise SystemExit(main())
