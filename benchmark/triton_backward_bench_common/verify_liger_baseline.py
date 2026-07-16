"""Op-agnostic correctness gate for a task_spec's Liger baseline wrapper.

Runs task_spec.make_liger_autograd_pair_fns() and checks its forward output and (grad...)
against the pure-PyTorch oracles, over the correctness cases plus a couple of large benchmark
shapes (to catch in-place / state-threading bugs that only bite at scale). This is THE gate
for the hand-written strong_baselines/liger_<op>.py wrapper.

Usage:
    python verify_liger_baseline.py --bench benchmark/triton_rmsnorm_handwritten_backward_bench
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
import uuid


def _import_task_spec(bench_dir: str):
    bench_dir = os.path.abspath(bench_dir)
    for p in (bench_dir, os.path.dirname(bench_dir), os.path.dirname(os.path.dirname(bench_dir))):
        if p not in sys.path:
            sys.path.insert(0, p)
    spec = importlib.util.spec_from_file_location(
        f"ts_{uuid.uuid4().hex}", os.path.join(bench_dir, "task_spec.py")
    )
    ts = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ts)
    return ts


def _split(ts, inputs):
    cot = int(getattr(ts, "AUTOGRAD_PAIR_COTANGENT_INDEX", 0))
    fwd_idx = tuple(ts.AUTOGRAD_PAIR_FORWARD_INPUT_INDICES)
    extra_idx = tuple(getattr(ts, "AUTOGRAD_PAIR_BACKWARD_EXTRA_INPUT_INDICES", ()))
    return inputs[cot], tuple(inputs[i] for i in fwd_idx), tuple(inputs[i] for i in extra_idx)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", required=True)
    ap.add_argument("--extra-large", type=int, default=2, help="how many largest benchmark shapes to also check")
    args = ap.parse_args(argv)

    import torch

    if not torch.cuda.is_available():
        print("SKIP: no CUDA")
        return 0

    ts = _import_task_spec(args.bench)
    liger_fwd, liger_bwd = ts.make_liger_autograd_pair_fns()

    # correctness cases + the N largest benchmark shapes (dedup by shape, keep case objects)
    cases = list(ts.CORRECTNESS_CASES)
    big = sorted(ts.BENCHMARK_CASES, key=lambda c: c.rows * c.cols, reverse=True)
    seen = {(c.rows, c.cols) for c in cases}
    for c in big:
        if (c.rows, c.cols) not in seen:
            cases.append(c)
            seen.add((c.rows, c.cols))
            if len(seen) - len(ts.CORRECTNESS_CASES) >= args.extra_large:
                break

    names = ts.OUTPUT_NAMES
    all_ok = True
    print(f"{'shape':>16} {'dtype':>9} | {'fwd':>5} " + " ".join(f"{n:>7}" for n in names) + "   max_abs_err")
    for case in cases:
        if hasattr(ts, "seed_for_case"):
            torch.manual_seed(ts.seed_for_case(case))
        inputs = tuple(ts.make_inputs(torch, case))
        dout, fargs, bextra = _split(ts, inputs)

        y_ref = (ts.autograd_pair_forward_oracle(torch, *fargs) if hasattr(ts, "autograd_pair_forward_oracle")
                 else ts.forward_oracle(torch, *fargs))
        g_ref = ts.torch_oracle(torch, *inputs)
        if not isinstance(g_ref, tuple):
            g_ref = (g_ref,)

        y, saved = liger_fwd(*fargs)
        grads = liger_bwd(dout, saved, *bextra)
        if not isinstance(grads, tuple):
            grads = (grads,)
        torch.cuda.synchronize()

        # Saved-reuse idempotency: the timing harness calls forward once then backward MANY times
        # on the SAME dout/saved, so backward must be pure. Some liger backward ops mutate saved
        # activations (or dout) in place (e.g. geglu stores grads back into a/b), which is correct
        # on the first call but corrupts every later one. Catch it here by calling backward again.
        grads2 = liger_bwd(dout, saved, *bextra)
        if not isinstance(grads2, tuple):
            grads2 = (grads2,)
        torch.cuda.synchronize()
        # Idempotency = "does backward corrupt its reused saved/dout?" A true in-place mutation
        # makes the 2nd call catastrophically wrong (whole tensor off, or NaN). A kernel that is
        # merely non-deterministic across launches (e.g. Liger's multi-block softmax backward uses
        # atomic/split reductions over 10^5 elements) differs by small amounts that can still exceed
        # the tight correctness atol. So use a LOOSE fixed tolerance here — big enough to ignore
        # cross-launch non-determinism, far smaller than a real mutation's error.
        idempotent = all(
            bool(torch.allclose(g1.float(), g2.float(), atol=1e-2, rtol=1e-2))
            for g1, g2 in zip(grads, grads2)
        )

        fwd_ok = bool(torch.allclose(y, y_ref, atol=ts._forward_atol(case) if hasattr(ts, "_forward_atol")
                                     else max(ts.atol(case, n) for n in names),
                                     rtol=max(ts.rtol(case, n) for n in names)))
        abs_errs = []
        rel_errs = []
        goks = []
        for n, a, b in zip(names, grads, g_ref):
            ok = bool(torch.allclose(a, b, atol=ts.atol(case, n), rtol=ts.rtol(case, n)))
            goks.append(ok)
            diff = (a.float() - b.float()).abs()
            abs_errs.append(float(diff.max()))
            denom = b.float().abs().clamp(min=1e-8)
            rel_errs.append(float((diff / denom).max()))
        ok = fwd_ok and all(goks) and idempotent
        all_ok = all_ok and ok
        note = ""
        if not (fwd_ok and all(goks)):
            note = "  <-- MISMATCH"
        elif not idempotent:
            note = "  <-- NOT IDEMPOTENT (backward mutates saved/dout; clone it)"
        print(f"{str((case.rows,case.cols)):>16} {case.dtype_name:>9} | "
              f"{'OK' if fwd_ok else 'FAIL':>5} "
              + " ".join(f"{'OK' if g else 'FAIL':>7}" for g in goks)
              + f"   abs={max(abs_errs):.2e} rel={max(rel_errs):.2e}"
              + f" idem={'OK' if idempotent else 'FAIL'}"
              + note)

    print("\nRESULT:", "ALL PASS" if all_ok else "FAILURES PRESENT")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
