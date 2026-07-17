"""evograd — the product interface.

Give it ONE PyTorch forward — a path to a .py file, or the function object itself — and it
returns a dispatched forward+backward kernel and a performance report, printing every stage and
its live progress as it runs.

    from pipeline.case_harness_agent.evograd import evograd
    result = evograd("my_rmsnorm.py", op="rmsnorm")          # file form
    result = evograd(my_rmsnorm_fn, op="rmsnorm")            # callable form (self-contained
                                                              # torch-ops def; snapshotted to
                                                              # <bench_dir>/user_forward.py)
    print(result.program)   # the deployed fw+bwd, ready to import
    print(result.report)    # RESULTS_*.md
    print(result.metrics)   # dispatch_report.json (has the measured baseline + geomeans)

Or from the shell:

    python -m pipeline.case_harness_agent.evograd --forward my_rmsnorm.py --op rmsnorm

CONTRACT / BOUNDARY (deliberate):
  * File-in, file-out. `forward` is a path to a .py file (a function name may be appended as
    `path.py:fn`; omit it and the single top-level function is used). Outputs are files on disk.
    A callable is accepted as sugar over the same contract: it is snapshotted to
    `<bench_dir>/user_forward.py` at the entry (the stages themselves stay file-driven), so it
    must be a self-contained named `def` over torch ops — no lambdas, closures, or globals
    beyond math/torch/F.
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
import inspect
import json
import os
import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from pipeline.case_harness_agent.orchestrate import (
    OrchestratorConfig, REPO_ROOT, _STAGE_ORDER, orchestrate,
)

_EVOLVE_GROUPS = ["full", "small", "large"]

# Imports the materialized snapshot provides. A callable passed to evograd() may only reference
# these (plus builtins) as globals — anything else can't survive the snapshot and is rejected
# up front with a clear message instead of failing later inside a pipeline stage.
_SNAPSHOT_HEADER = "import math\n\nimport torch\nimport torch.nn.functional as F\n"
_SNAPSHOT_GLOBALS = {"math", "torch", "F"}


def _materialize_forward(fn: Callable, bench_dir: Path) -> str:
    """Snapshot a Python callable to `<bench_dir>/user_forward.py` and return 'path:fn' for it.

    The pipeline stages are subprocesses driven by a `--forward` string, so a live function
    object cannot cross that boundary — it is materialized to source once, here at the product
    entry, and the file-based chain runs unchanged. The snapshot doubles as the record of exactly
    which forward this bench dir was built from.
    """
    if not callable(fn):
        raise TypeError(f"forward must be a path string or a callable, got {type(fn).__name__}")
    if getattr(fn, "__name__", "<lambda>") == "<lambda>":
        raise ValueError("forward callable must be a named `def`, not a lambda (its source is "
                         "snapshotted to a file and re-imported by name)")
    if fn.__closure__:
        raise ValueError(
            f"forward callable {fn.__name__} captures enclosing variables (a closure); the "
            f"snapshot cannot carry them. Make it self-contained, or put it in a .py file and "
            f"pass the path.")
    try:
        src = textwrap.dedent(inspect.getsource(fn))
    except (OSError, TypeError) as e:
        raise ValueError(
            f"cannot read the source of {fn.__name__} (defined in a REPL or a built-in?): {e}. "
            f"Put it in a .py file and pass the path instead.") from e
    # Globals the body actually loads must be covered by the snapshot header.
    unsupported = sorted(
        n for n in fn.__code__.co_names
        if n in fn.__globals__ and n not in _SNAPSHOT_GLOBALS)
    if unsupported:
        raise ValueError(
            f"forward callable {fn.__name__} references globals the snapshot cannot carry: "
            f"{', '.join(unsupported)} (only math/torch/F are provided). Put the forward and its "
            f"helpers in a .py file and pass the path.")

    bench_dir.mkdir(parents=True, exist_ok=True)
    snap = bench_dir / "user_forward.py"
    snap.write_text(
        f'"""Materialized from the callable `{fn.__name__}` passed to evograd() — do not '
        f'edit."""\n{_SNAPSHOT_HEADER}\n\n{src}',
        encoding="utf-8")
    print(f"[evograd] forward callable `{fn.__name__}` materialized to {snap}")
    return f"{snap}:{fn.__name__}"


@dataclass
class EvogradResult:
    op: str
    bench_dir: Path
    program: Path       # the deployed, shape-dispatched forward+backward
    report: Path        # RESULTS_<op>_vs_<baseline>.md
    metrics: Path       # <op>_dispatch_report.json
    baseline: str       # what speedups were actually measured against ("liger" / "pytorch_autograd")


def evograd(
    forward: str | Path | Callable | None = None,
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
) -> EvogradResult:
    """Run the whole pipeline on `forward` and return the artifact paths.

    `forward` may be a .py path, a `path:fn`, or the callable itself. It can also be OMITTED for
    an op the registry knows (a built case, or one declared in the suite manifest) — its forward,
    Liger sources, and baseline are then looked up. Raises RuntimeError if any stage fails.
    """
    # Additive registry lookup: fill in whatever the caller didn't pass for a known op. Explicit
    # arguments always win; an unknown op with no forward is reported by get_op with the op list.
    if forward is None or (baseline in ("auto", "liger") and not liger_sources):
        from benchmark.op_registry import discover_ops
        info = discover_ops().get(op)
        if forward is None:
            if info is None or info.forward is None:
                from benchmark.op_registry import get_op
                get_op(op)  # raises KeyError listing known ops + the --forward hint
                raise ValueError(f"op {op!r} is registered but has no known forward; pass one")
            forward = info.forward
        if info is not None and not liger_sources:
            liger_sources = info.liger_sources

    bench = Path(bench_dir) if bench_dir else (REPO_ROOT / "benchmark" / f"triton_{op}_backward_bench")
    # Stages address bench paths relative to REPO_ROOT (e.g. `b.relative_to(REPO_ROOT)`), so a
    # relative --bench-dir must be anchored there, not left relative / resolved against cwd.
    if not bench.is_absolute():
        bench = REPO_ROOT / bench
    # A live callable can't cross the subprocess boundary the stages run behind — snapshot it to
    # a file in the bench dir and run the file-based chain unchanged.
    if not isinstance(forward, (str, Path)):
        forward = _materialize_forward(forward, bench)
    cfg = OrchestratorConfig(
        op=op, bench_dir=bench, forward=str(forward),
        api_base=api_base, model=model,
        api_key=api_key or os.environ.get("OPENAI_API_KEY"),
        iterations=iterations, perf_baseline=baseline,
        liger_sources=tuple(liger_sources), force=force, gpus=gpus,
    )
    rc = orchestrate(cfg, _STAGE_ORDER, list(_EVOLVE_GROUPS))
    if rc != 0:
        raise RuntimeError(f"evograd failed for op={op} (see the stage output above)")

    metrics = bench / f"{op}_dispatch_report.json"
    measured_baseline = "unknown"
    if metrics.exists():
        measured_baseline = str(json.loads(metrics.read_text()).get("baseline", "unknown"))
    reports = sorted(bench.glob(f"RESULTS_{op}_vs_*.md"))
    return EvogradResult(
        op=op, bench_dir=bench,
        program=bench / f"{op}_final_dispatched.py",
        report=reports[0] if reports else bench / f"RESULTS_{op}_vs_{measured_baseline}.md",
        metrics=metrics,
        baseline=measured_baseline,
    )


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="evograd — forward file in, dispatched fw+bwd + report out")
    ap.add_argument("--forward", default=None,
                    help="path to a .py file with the PyTorch forward (append ':fn' to pick a "
                         "function when the file has more than one). Optional for an op the "
                         "registry knows — run `python -m benchmark.op_registry` to list them")
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
        result = evograd(
            args.forward, op=args.op, api_key=args.api_key, baseline=args.perf_baseline,
            liger_sources=tuple(args.liger_sources), iterations=args.iterations, gpus=args.gpus,
            bench_dir=args.bench_dir, model=args.model, api_base=args.api_base, force=args.force,
        )
    except RuntimeError as e:
        print(f"\n{e}", file=sys.stderr)
        return 1

    print("\n=== evograd artifacts ===")
    print(f"  program : {result.program}")
    print(f"  report  : {result.report}")
    print(f"  metrics : {result.metrics}")
    print(f"  baseline: {result.baseline}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
