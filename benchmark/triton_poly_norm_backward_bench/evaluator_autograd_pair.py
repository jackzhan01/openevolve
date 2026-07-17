import os
import sys

BENCHMARK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO_ROOT = os.path.dirname(BENCHMARK_DIR)
for _p in (BENCHMARK_DIR, REPO_ROOT):
    if _p in sys.path:
        sys.path.remove(_p)
    sys.path.insert(0, _p)

from benchmark.triton_backward_bench_common.autograd_pair_evaluator_core import (  # noqa: E402
    evaluate_autograd_pair_program,
    main as core_main,
)

try:
    from benchmark.triton_poly_norm_backward_bench import task_spec  # noqa: E402
except ImportError:  # pragma: no cover
    import task_spec  # type: ignore  # noqa: E402


# CORRECTNESS-ONLY: this evaluator vets the fusion SEED, which only needs to be a *correct*
# starting kernel. Timing/scoring on the full benchmark grid — including the largest shapes whose
# cross-row weight/bias reduction needs row-tiling — is evolve's job (the `large` specialist), not
# the seed's. Running the full grid here rejects correct seeds for a problem they need not solve yet
# (e.g. a naive reduction hitting Triton's per-tensor numel cap on 100k-row shapes).
def evaluate(program_path: str):
    return evaluate_autograd_pair_program(program_path, task_spec, run_benchmarks=False)


if __name__ == "__main__":
    raise SystemExit(core_main(sys.argv, task_spec, run_benchmarks=False))
