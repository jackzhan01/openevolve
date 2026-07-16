"""autodiff — the product interface.

Give it ONE PyTorch forward (in a file); it returns a dispatched forward+backward kernel and a
performance report, printing every stage and its live progress as it runs.

    from pipeline.case_harness_agent.autodiff import autodiff
    result = autodiff("my_rmsnorm.py", op="rmsnorm")
    print(result.program)   # the deployed fw+bwd, ready to import
    print(result.report)    # RESULTS_*.md
    print(result.metrics)   # dispatch_report.json (has the measured baseline + geomeans)

Or from the shell:

    python -m pipeline.case_harness_agent.autodiff --forward my_rmsnorm.py --op rmsnorm

CONTRACT / BOUNDARY (deliberate):
  * File-in, file-out. `forward` is a path to a .py file (a function name may be appended as
    `path.py:fn`; omit it and the single top-level function is used). Outputs are files on disk.
  * This assumes it is ALREADY running on a machine with a GPU. It NEVER allocates nodes / calls
    srun / sets up conda or API keys — that is the caller's environment (for our own Delta testing,
    `bootstrap.sh` does it before invoking this). Keeping scheduling out of the pipeline is what
    lets any user run it under any scheduler.
  * The three evolve groups (full / small / large) run SEQUENTIALLY on this one GPU. That is the
    portable default; parallelizing them across nodes is an environment concern layered on top,
    not something this function reaches out to do.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

from pipeline.case_harness_agent.orchestrate import (
    OrchestratorConfig, REPO_ROOT, _STAGE_ORDER, orchestrate,
)

_EVOLVE_GROUPS = ["full", "small", "large"]


@dataclass
class AutodiffResult:
    op: str
    bench_dir: Path
    program: Path       # the deployed, shape-dispatched forward+backward
    report: Path        # RESULTS_<op>_vs_<baseline>.md
    metrics: Path       # <op>_dispatch_report.json
    baseline: str       # what speedups were actually measured against ("liger" / "pytorch_autograd")


def autodiff(
    forward: str,
    *,
    op: str,
    api_key: str | None = None,
    baseline: str = "auto",
    liger_sources: tuple[str, ...] = (),
    iterations: int = 10,
    gpus: int = 1,
    bench_dir: str | Path | None = None,
    model: str = "gpt-5.5",
    api_base: str = "https://api.openai.com/v1",
    force: bool = False,
) -> AutodiffResult:
    """Run the whole pipeline on `forward` and return the artifact paths.

    Raises RuntimeError if any stage fails (the failing stage's tail was already printed live).
    """
    bench = Path(bench_dir) if bench_dir else (REPO_ROOT / "benchmark" / f"triton_{op}_backward_bench")
    # Stages address bench paths relative to REPO_ROOT (e.g. `b.relative_to(REPO_ROOT)`), so a
    # relative --bench-dir must be anchored there, not left relative / resolved against cwd.
    if not bench.is_absolute():
        bench = REPO_ROOT / bench
    cfg = OrchestratorConfig(
        op=op, bench_dir=bench, forward=forward,
        api_base=api_base, model=model,
        api_key=api_key or os.environ.get("OPENAI_API_KEY"),
        iterations=iterations, perf_baseline=baseline,
        liger_sources=tuple(liger_sources), force=force, gpus=gpus,
    )
    rc = orchestrate(cfg, _STAGE_ORDER, list(_EVOLVE_GROUPS))
    if rc != 0:
        raise RuntimeError(f"autodiff failed for op={op} (see the stage output above)")

    metrics = bench / f"{op}_dispatch_report.json"
    measured_baseline = "unknown"
    if metrics.exists():
        measured_baseline = str(json.loads(metrics.read_text()).get("baseline", "unknown"))
    reports = sorted(bench.glob(f"RESULTS_{op}_vs_*.md"))
    return AutodiffResult(
        op=op, bench_dir=bench,
        program=bench / f"{op}_final_dispatched.py",
        report=reports[0] if reports else bench / f"RESULTS_{op}_vs_{measured_baseline}.md",
        metrics=metrics,
        baseline=measured_baseline,
    )


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="autodiff — forward file in, dispatched fw+bwd + report out")
    ap.add_argument("--forward", required=True,
                    help="path to a .py file with the PyTorch forward (append ':fn' to pick a "
                         "function when the file has more than one)")
    ap.add_argument("--op", required=True, help="operator name, e.g. rmsnorm (names the output dir)")
    ap.add_argument("--perf-baseline", default="auto", choices=["auto", "liger", "pytorch_autograd"],
                    help="what to measure speedups against; 'liger' HARD-FAILS if Liger can't be "
                         "resolved instead of silently downgrading")
    ap.add_argument("--liger-source", action="append", default=[], dest="liger_sources",
                    help="Liger op source file (repeatable); needed when Liger's module name "
                         "differs from --op, e.g. op 'rmsnorm' is Liger's 'rms_norm'")
    ap.add_argument("--iterations", type=int, default=10, help="evolve iterations per group")
    ap.add_argument("--gpus", type=int, default=1, choices=[1, 3],
                    help="1 = the 3 groups evolve sequentially on one GPU (portable default); "
                         "3 = one group per GPU in parallel (needs 3 visible GPUs)")
    ap.add_argument("--bench-dir", default=None, help="output dir (default benchmark/triton_<op>_backward_bench)")
    ap.add_argument("--model", default="gpt-5.5")
    ap.add_argument("--api-base", default="https://api.openai.com/v1")
    ap.add_argument("--api-key", default=None)
    ap.add_argument("--force", action="store_true", help="redo stages even if artifacts exist")
    args = ap.parse_args(argv or sys.argv[1:])

    try:
        result = autodiff(
            args.forward, op=args.op, api_key=args.api_key, baseline=args.perf_baseline,
            liger_sources=tuple(args.liger_sources), iterations=args.iterations, gpus=args.gpus,
            bench_dir=args.bench_dir, model=args.model, api_base=args.api_base, force=args.force,
        )
    except RuntimeError as e:
        print(f"\n{e}", file=sys.stderr)
        return 1

    print("\n=== autodiff artifacts ===")
    print(f"  program : {result.program}")
    print(f"  report  : {result.report}")
    print(f"  metrics : {result.metrics}")
    print(f"  baseline: {result.baseline}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
