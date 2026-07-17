import os
import sys

os.environ.setdefault("AUTOGRAD_PAIR_SCORE_MODE", "speed_memory_min_weighted_geomean")
os.environ.setdefault("AUTOGRAD_PAIR_PERF_BASELINE", "liger")
os.environ.setdefault("AUTOGRAD_PAIR_SUITE", "large")

BENCHMARK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO_ROOT = os.path.dirname(BENCHMARK_DIR)
for _p in (BENCHMARK_DIR, REPO_ROOT):
    if _p in sys.path:
        sys.path.remove(_p)
    sys.path.insert(0, _p)

from benchmark.triton_backward_bench_common.autograd_pair_evaluator_core import (  # noqa: E402
    evaluate_isolated,
    main as core_main,
)

try:
    from benchmark.triton_fused_add_rms_norm_backward_bench import task_spec  # noqa: E402
except ImportError:  # pragma: no cover
    import task_spec  # type: ignore  # noqa: E402


def evaluate(program_path: str):
    # Isolated: a killed subprocess takes its CUDA context (and any deadlocked kernel) with it.
    return evaluate_isolated(os.path.abspath(__file__), program_path, task_spec)


if __name__ == "__main__":
    raise SystemExit(core_main(sys.argv, task_spec))
