"""NCU-aware wrapper around the KL-Divergence autograd-pair evaluator.

Runs the normal timing-based evaluator first. If the candidate is correct and
NCU mode is not "off", profiles it with `ncu` and returns the hardware metrics
as `ncu_*` keys in the metrics dict plus `ncu_report_path` in artifacts.

The metrics and report path are consumed by the NCU silent optimizer in
openevolve/process_parallel.py and are stripped before any program is stored
in the evolution database — they never influence scores or prompts.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

BENCHMARK_DIR = Path(__file__).resolve().parent
if str(BENCHMARK_DIR) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_DIR))

import ncu_profile  # noqa: E402
import task_spec  # noqa: E402
from evaluator_autograd_pair import evaluate as _base_evaluate  # noqa: E402

try:
    from openevolve.evaluation_result import EvaluationResult
except Exception:  # pragma: no cover
    from dataclasses import dataclass, field

    @dataclass
    class EvaluationResult:
        metrics: dict[str, float]
        artifacts: dict[str, str | bytes] = field(default_factory=dict)


# NCU_MODE is read at call time (not import time) so the OpenEvolve worker can
# toggle profiling per iteration via os.environ — only NCU-pass iterations pay
# the ~20-30s profiling cost.
NCU_TIMEOUT = int(os.environ.get("AUTOGRAD_PAIR_NCU_TIMEOUT", "120"))
NCU_WARMUP = int(os.environ.get("AUTOGRAD_PAIR_NCU_WARMUP", "5"))
NCU_SHAPE = os.environ.get("AUTOGRAD_PAIR_NCU_SHAPE")  # "rows,cols,dtype" override


def _ncu_mode() -> str:
    return os.environ.get("AUTOGRAD_PAIR_NCU_MODE", "always")  # off | always


def _representative_case():
    if NCU_SHAPE:
        rows, cols, dtype_name = NCU_SHAPE.split(",")
        return task_spec.TestCase(int(rows), int(cols), dtype_name.strip(), 5e-2, 5e-2)
    return max(task_spec.BENCHMARK_CASES, key=lambda c: c.rows * c.cols)


def evaluate(program_path: str) -> EvaluationResult:
    result = _base_evaluate(program_path)
    if result.metrics.get("correct", 0.0) != 1.0:
        return result

    if _ncu_mode() == "off":
        return result

    case = _representative_case()
    profile = ncu_profile.run_ncu_profile(
        program_path,
        case,
        warmup=NCU_WARMUP,
        timeout=NCU_TIMEOUT,
    )

    artifacts = dict(result.artifacts)
    metrics = dict(result.metrics)

    if not profile.ok:
        artifacts["ncu_profile_error"] = profile.error or "unknown ncu failure"
        return EvaluationResult(metrics=metrics, artifacts=artifacts)

    flags = ncu_profile.derive_flags(profile.aggregate)

    metrics["ncu_occupancy_pct"] = flags["occupancy_pct"] if flags["occupancy_pct"] is not None else -1.0
    metrics["ncu_long_scoreboard_stall_ratio"] = (
        flags["long_scoreboard_stall_ratio"] if flags["long_scoreboard_stall_ratio"] is not None else -1.0
    )
    metrics["ncu_sm_throughput_pct"] = flags["sm_throughput_pct"] if flags["sm_throughput_pct"] is not None else -1.0
    metrics["ncu_dram_throughput_pct"] = (
        flags["dram_throughput_pct"] if flags["dram_throughput_pct"] is not None else -1.0
    )

    if profile.report_path:
        artifacts["ncu_report_path"] = profile.report_path

    return EvaluationResult(metrics=metrics, artifacts=artifacts)


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(f"Usage: {argv[0]} PROGRAM_PATH")
        return 2
    result = evaluate(argv[1])
    print(json.dumps({"metrics": result.metrics, "artifacts": result.artifacts}, indent=2))
    return 0 if result.metrics.get("correct", 0.0) == 1.0 else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
