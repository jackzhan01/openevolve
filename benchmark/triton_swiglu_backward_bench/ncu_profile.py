"""Bench-local NCU profiling shim.

All generic machinery (two-phase warmup/profiled harness, ncu_report
discovery, report parsing, aggregation) lives in
benchmark/triton_backward_bench_common/ncu_profile_core.py and is driven by
this bench's task_spec AUTOGRAD_PAIR_* conventions.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

BENCHMARK_DIR = Path(__file__).resolve().parent
REPO_ROOT = BENCHMARK_DIR.parents[1]
for _p in (REPO_ROOT, BENCHMARK_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from benchmark.triton_backward_bench_common.ncu_profile_core import (  # noqa: E402
    DEFAULT_METRICS,
    KernelProfile,
    ProfileResult,
    derive_flags,
    run_ncu_profile as _run_ncu_profile_core,
)

__all__ = [
    "DEFAULT_METRICS",
    "KernelProfile",
    "ProfileResult",
    "derive_flags",
    "run_ncu_profile",
]


def run_ncu_profile(program_path, case, **kwargs):
    return _run_ncu_profile_core(program_path, case, bench_dir=BENCHMARK_DIR, **kwargs)


def main(argv: list[str]) -> int:
    if len(argv) != 5:
        print(f"Usage: {{argv[0]}} PROGRAM_PATH ROWS COLS DTYPE_NAME")
        return 2
    program_path, rows, cols, dtype_name = argv[1:5]

    import task_spec  # noqa: E402

    case = task_spec.TestCase(int(rows), int(cols), dtype_name, 5e-2, 5e-2)
    result = run_ncu_profile(program_path, case)
    print(json.dumps(result.to_json(), indent=2, sort_keys=True, default=str))
    return 0 if result.ok else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
