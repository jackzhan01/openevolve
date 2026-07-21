"""Bench-local NCU profiling shim for the RMSNorm STANDALONE-BACKWARD bench.

Unlike the autograd-pair benches, this bench evolves a single backward
function (task_spec.CANDIDATE_FN_NAME = "rmsnorm_backward_triton"), so the
harness templates here call only the backward.  All generic machinery lives
in benchmark/triton_backward_bench_common/ncu_profile_core.py.
"""

from __future__ import annotations

import json
import sys
import textwrap
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

_IMPORT_PREAMBLE = textwrap.dedent("""\
    import importlib.util
    import sys
    from pathlib import Path

    import torch

    REPO_ROOT = Path({repo_root!r})
    BENCH_DIR = Path({bench_dir!r})
    for _p in (REPO_ROOT, BENCH_DIR):
        if str(_p) not in sys.path:
            sys.path.insert(0, str(_p))

    import task_spec

    spec = importlib.util.spec_from_file_location("ncu_profile_candidate", {program_path!r})
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    backward_fn = getattr(module, task_spec.CANDIDATE_FN_NAME)

    case = task_spec.TestCase({rows}, {cols}, {dtype_name!r}, {atol}, {rtol})
    """)

_WARMUP_TEMPLATE = _IMPORT_PREAMBLE + textwrap.dedent("""\
    torch.manual_seed(0)
    dy, x, weight, eps = task_spec.make_inputs(torch, case)

    for _ in range({warmup}):
        backward_fn(dy, x, weight, eps)
    torch.cuda.synchronize()

    torch.save([dy, x, weight, eps], {inputs_path!r})
    """)

_PROFILED_TEMPLATE = _IMPORT_PREAMBLE + textwrap.dedent("""\
    dy, x, weight, eps = torch.load({inputs_path!r}, map_location="cuda")

    backward_fn(dy, x, weight, eps)
    torch.cuda.synchronize()
    """)


def run_ncu_profile(program_path, case, **kwargs):
    return _run_ncu_profile_core(
        program_path,
        case,
        bench_dir=BENCHMARK_DIR,
        warmup_template=_WARMUP_TEMPLATE,
        profiled_template=_PROFILED_TEMPLATE,
        **kwargs,
    )


def main(argv: list[str]) -> int:
    if len(argv) != 5:
        print(f"Usage: {argv[0]} PROGRAM_PATH ROWS COLS DTYPE_NAME")
        return 2
    program_path, rows, cols, dtype_name = argv[1:5]

    import task_spec  # noqa: E402

    case = task_spec.TestCase(int(rows), int(cols), dtype_name, 5e-2, 5e-2)
    result = run_ncu_profile(program_path, case)
    print(json.dumps(result.to_json(), indent=2, sort_keys=True, default=str))
    return 0 if result.ok else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
