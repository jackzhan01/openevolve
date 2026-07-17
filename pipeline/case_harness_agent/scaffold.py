"""Phase 1 — deterministic scaffolding codegen for a Liger benchmark case.

Given a `CaseSpec` (small declarative description of an op + a few LLM-provided text slots),
this emits the mechanical, template-identical files of a `triton_<op>_backward_bench/` case:

  - strong_baselines/__init__.py
  - evaluator_autograd_pair_speed_memory_min_liger.py          (full suite, min score)
  - evaluator_autograd_pair_weighted_geomean_liger_small.py    (small suite, weighted geomean)
  - evaluator_autograd_pair_weighted_geomean_liger_large.py    (large suite, weighted geomean)
  - config_autograd_pair_speed_memory_min_liger.yaml
  - config_autograd_pair_weighted_geomean_liger_small.yaml
  - config_autograd_pair_weighted_geomean_liger_large.yaml
  - regime_hooks.py  (snippet: regime_feature/REGIME_SPLIT/case_weight/SMALL_CASES/LARGE_CASES/
                      make_liger_autograd_pair_fns — to be spliced into task_spec.py)

The MATH-bearing files (forward_ref, backward_ref, task_spec body, strong_baselines/liger_<op>.py)
are produced by the LLM layer (Phase 2/3), not here. This module only assembles the fixed
skeletons, so the class of bugs from hand-copying / shell-heredoc YAML mangling cannot occur.

All templates are Python string formatting (never shell), so indentation is exact.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
import textwrap


@dataclass
class CaseSpec:
    op: str                      # "swiglu"
    package: str                 # "triton_swiglu_backward_bench"
    op_title: str                # "SwiGLU"  (used in the system message)
    forward_fn: str              # "swiglu_forward_with_saved"
    backward_fn: str             # "swiglu_backward_from_saved"
    api_block: str               # the def-signature block shown to the LLM (task_spec.AUTOGRAD_PAIR_API)
    math_block: str              # forward/backward math description (indented content, no leading indent)
    saved_note: str              # one line on the evolvable saved-tensor contract for this op
    regime_feature_expr: str     # e.g. "case.rows * case.cols" (swiglu) or "case.rows" (rmsnorm)
    regime_feature_doc: str      # e.g. "total number of elements (rows*cols)"
    regime_split: object         # int/float rule-of-thumb cut on regime_feature
    small_note: str              # regime-specific guidance for the small specialist
    large_note: str              # regime-specific guidance for the large specialist
    # Performance baseline to evolve against: use a strong hand-optimized kernel when the op has
    # one (Liger), otherwise fall back to PyTorch autograd (always available for any forward).
    perf_baseline: str = "liger"          # "liger" | "pytorch_autograd"
    baseline_title: str = "Liger's hand-optimized"   # shown in the evolve prompt


# --------------------------------------------------------------------------- evaluators

_EVAL_MIN_LIGER = '''import os
import sys

os.environ.setdefault("AUTOGRAD_PAIR_SCORE_MODE", "speed_memory_min")
os.environ.setdefault("AUTOGRAD_PAIR_PERF_BASELINE", "{perf_baseline}")

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
    from benchmark.{package} import task_spec  # noqa: E402
except ImportError:  # pragma: no cover
    import task_spec  # type: ignore  # noqa: E402


def evaluate(program_path: str):
    # Isolated: a killed subprocess takes its CUDA context (and any deadlocked kernel) with it.
    return evaluate_isolated(os.path.abspath(__file__), program_path, task_spec)


if __name__ == "__main__":
    raise SystemExit(core_main(sys.argv, task_spec))
'''

_EVAL_WG_SUITE = '''import os
import sys

os.environ.setdefault("AUTOGRAD_PAIR_SCORE_MODE", "speed_memory_min_weighted_geomean")
os.environ.setdefault("AUTOGRAD_PAIR_PERF_BASELINE", "{perf_baseline}")
os.environ.setdefault("AUTOGRAD_PAIR_SUITE", "{suite}")

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
    from benchmark.{package} import task_spec  # noqa: E402
except ImportError:  # pragma: no cover
    import task_spec  # type: ignore  # noqa: E402


def evaluate(program_path: str):
    # Isolated: a killed subprocess takes its CUDA context (and any deadlocked kernel) with it.
    return evaluate_isolated(os.path.abspath(__file__), program_path, task_spec)


if __name__ == "__main__":
    raise SystemExit(core_main(sys.argv, task_spec))
'''


# --------------------------------------------------------------------------- configs

def _indent(block: str, spaces: int) -> str:
    pad = " " * spaces
    return "\n".join((pad + line if line.strip() else line) for line in block.rstrip("\n").splitlines())


def _config(spec: CaseSpec, score_desc: str, regime_note: str, suite: str | None) -> str:
    note = f"\n{_indent(regime_note, 4)}\n" if regime_note else ""
    body = (
        f"You are an expert Triton GPU programmer optimizing an autograd-pair kernel.\n"
        f"Your target to beat is {spec.baseline_title} {spec.op_title} implementation.\n"
        f"{note}\n"
        f"Improve the code inside the EVOLVE-BLOCK. The public API is:\n\n"
        f"{_indent(spec.api_block, 4)}\n\n"
        f"{_indent(spec.math_block, 4)}\n\n"
        f"{_indent(spec.saved_note, 4)}\n\n"
        f"The evaluator scores {score_desc} against {spec.baseline_title} forward/backward as the\n"
        f"baseline, with a small saved-memory penalty. Correctness is a hard requirement."
    )
    return (
        "max_iterations: 10\n"
        "checkpoint_interval: 5\n"
        'log_level: "INFO"\n'
        "max_tasks_per_child: 1\n\n"
        "llm:\n"
        '  primary_model: "gpt-5.5"\n'
        "  primary_model_weight: 1.0\n"
        '  api_base: "https://api.openai.com/v1"\n'
        '  api_key: "${OPENAI_API_KEY}"\n'
        "  temperature: 0.5\n"
        "  max_tokens: 20000\n"
        "  timeout: 300\n\n"
        "prompt:\n"
        "  system_message: |\n"
        f"{_indent(body, 4)}\n\n"
        "database:\n"
        "  population_size: 30\n"
        "  archive_size: 20\n"
        "  num_islands: 2\n"
        "  elite_selection_ratio: 0.25\n"
        "  exploitation_ratio: 0.75\n"
        "  similarity_threshold: 0.99\n\n"
        "evaluator:\n"
        # Timing the autograd baseline is ~100x slower per step than Liger, and the first
        # evaluation also pays Triton cold-compile — 300s only fits the Liger baseline.
        f"  timeout: {300 if spec.perf_baseline == 'liger' else 900}\n"
        "  parallel_evaluations: 1\n"
        "  cascade_evaluation: false\n\n"
        "diff_based_evolution: true\n"
        "max_code_length: 30000\n"
    )


# --------------------------------------------------------------------------- regime hooks

def _regime_hooks(spec: CaseSpec) -> str:
    return textwrap.dedent(f'''\
        # --- Shape-regime metadata (generated by case_harness_agent scaffold) ------------
        # regime_feature = {spec.regime_feature_doc}. REGIME_SPLIT is the rule-of-thumb cut used
        # only to build the small/large TRAINING suites; the deployment threshold is derived
        # afterward from where the trained programs' latency curves cross.
        REGIME_SPLIT = {spec.regime_split!r}


        def regime_feature(case: "TestCase") -> float:
            return float({spec.regime_feature_expr})


        def case_weight(case: "TestCase") -> float:
            """Weighted-geomean per-case weight: log2-distance from REGIME_SPLIT (emphasize the
            tail of each regime, de-emphasize near-boundary shapes)."""
            import math

            dist = abs(math.log2(regime_feature(case)) - math.log2(REGIME_SPLIT))
            return float(max(1.0, round(dist)))


        SMALL_CASES = [c for c in BENCHMARK_CASES if regime_feature(c) < REGIME_SPLIT]
        LARGE_CASES = [c for c in BENCHMARK_CASES if regime_feature(c) >= REGIME_SPLIT]


        def make_liger_autograd_pair_fns():
            """(forward_with_saved, backward_from_saved) from Liger, used by the evaluator core
            as the perf baseline when AUTOGRAD_PAIR_PERF_BASELINE=liger."""
            try:
                from benchmark.{spec.package}.strong_baselines.liger_{spec.op} import (
                    make_liger_{spec.op}_autograd_pair_fns,
                )
            except ImportError:
                from strong_baselines.liger_{spec.op} import make_liger_{spec.op}_autograd_pair_fns
            return make_liger_{spec.op}_autograd_pair_fns()
    ''')


# --------------------------------------------------------------------------- driver

def scaffold(spec: CaseSpec, bench_dir: str) -> list[str]:
    """Write the mechanical scaffolding files into bench_dir. Returns the paths written."""
    os.makedirs(os.path.join(bench_dir, "strong_baselines"), exist_ok=True)
    written = []

    def _write(rel, content):
        path = os.path.join(bench_dir, rel)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            f.write(content)
        written.append(path)

    _write("strong_baselines/__init__.py", "")
    _write(
        "evaluator_autograd_pair_speed_memory_min_liger.py",
        _EVAL_MIN_LIGER.format(package=spec.package, perf_baseline=spec.perf_baseline),
    )
    _write(
        "evaluator_autograd_pair_weighted_geomean_liger_small.py",
        _EVAL_WG_SUITE.format(package=spec.package, suite="small", perf_baseline=spec.perf_baseline),
    )
    _write(
        "evaluator_autograd_pair_weighted_geomean_liger_large.py",
        _EVAL_WG_SUITE.format(package=spec.package, suite="large", perf_baseline=spec.perf_baseline),
    )
    _write(
        "config_autograd_pair_speed_memory_min_liger.yaml",
        _config(spec, "min(backward_speedup, full_step_speedup)", "", None),
    )
    _write(
        "config_autograd_pair_weighted_geomean_liger_small.yaml",
        _config(spec, "a weighted geomean of per-shape min(backward_speedup, full_step_speedup)",
                spec.small_note, "small"),
    )
    _write(
        "config_autograd_pair_weighted_geomean_liger_large.yaml",
        _config(spec, "a weighted geomean of per-shape min(backward_speedup, full_step_speedup)",
                spec.large_note, "large"),
    )
    _write("regime_hooks.py", _regime_hooks(spec))
    return written
