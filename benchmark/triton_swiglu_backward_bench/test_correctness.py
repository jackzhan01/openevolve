"""Correctness tests for SwiGLU backward implementations.

Checks backward_ref.py, backward_naive_triton.py, and (if available)
the Liger strong baseline against torch_oracle.

Run with:
    conda run -n openev python test_correctness.py
"""

from __future__ import annotations

import sys

import torch

from backward_ref import swiglu_backward_ref
from backward_naive_triton import swiglu_backward_naive_triton
from task_spec import CORRECTNESS_CASES, TestCase, _dtype, torch_oracle
from strong_baselines.liger_swiglu import liger_available, make_liger_swiglu_backward_fn


def _allclose(a: torch.Tensor, b: torch.Tensor, atol: float, rtol: float) -> bool:
    return bool(torch.allclose(a.float(), b.float(), atol=atol, rtol=rtol))


def _max_err(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(torch.max(torch.abs(a.float() - b.float())).item())


def run_case(case: TestCase, liger_fn) -> bool:
    torch.manual_seed(case.rows * 100000 + case.cols)
    dtype = _dtype(torch, case.dtype_name)
    a = torch.randn((case.rows, case.cols), device="cuda", dtype=dtype)
    b = torch.randn((case.rows, case.cols), device="cuda", dtype=dtype)
    dc = torch.randn((case.rows, case.cols), device="cuda", dtype=dtype)

    da_ref, db_ref = torch_oracle(torch, dc, a, b)

    ok = True

    # backward_ref
    da_py, db_py = swiglu_backward_ref(dc, a, b)
    da_ok = _allclose(da_py, da_ref, case.atol_value, case.rtol_value)
    db_ok = _allclose(db_py, db_ref, case.atol_value, case.rtol_value)
    if not (da_ok and db_ok):
        print(f"  FAIL backward_ref [{case}]: da_err={_max_err(da_py,da_ref):.2e}  db_err={_max_err(db_py,db_ref):.2e}")
        ok = False

    # backward_naive_triton
    da_tri, db_tri = swiglu_backward_naive_triton(dc, a, b)
    da_ok = _allclose(da_tri, da_ref, case.atol_value, case.rtol_value)
    db_ok = _allclose(db_tri, db_ref, case.atol_value, case.rtol_value)
    if not (da_ok and db_ok):
        print(f"  FAIL naive_triton [{case}]: da_err={_max_err(da_tri,da_ref):.2e}  db_err={_max_err(db_tri,db_ref):.2e}")
        ok = False

    # Liger (optional)
    if liger_fn is not None:
        da_lg, db_lg = liger_fn(dc, a, b)
        da_ok = _allclose(da_lg, da_ref, case.atol_value, case.rtol_value)
        db_ok = _allclose(db_lg, db_ref, case.atol_value, case.rtol_value)
        if not (da_ok and db_ok):
            print(f"  FAIL liger [{case}]: da_err={_max_err(da_lg,da_ref):.2e}  db_err={_max_err(db_lg,db_ref):.2e}")
            ok = False

    return ok


def main() -> int:
    if not torch.cuda.is_available():
        print("SKIP: CUDA not available.")
        return 0

    liger_fn = None
    if liger_available():
        liger_fn = make_liger_swiglu_backward_fn()
        print("Liger SwiGLU: available")
    else:
        print("Liger SwiGLU: not available (skipped)")

    failures = 0
    for case in CORRECTNESS_CASES:
        passed = run_case(case, liger_fn)
        status = "PASS" if passed else "FAIL"
        print(f"  {status}  {case}")
        if not passed:
            failures += 1

    print(f"\n{len(CORRECTNESS_CASES) - failures}/{len(CORRECTNESS_CASES)} cases passed.")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
