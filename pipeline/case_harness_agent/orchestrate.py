"""Phase 4 — end-to-end orchestrator for one autograd-pair benchmark case.

Given an operator that already has `task_spec.py` + `forward_ref.py` (the Phase-3 hand-work),
this drives the whole pipeline with one command and no manual glue:

    scaffold (evaluators/configs)  ->  Phase-2 Liger wrapper (+gate)  ->  fusion seed
      ->  evolve {full, small, large}  ->  shape-dispatch  ->  RESULTS report

It reuses the existing engines (each already has its own verification loop):
  - `case_harness_agent.scaffold`            (Phase 1)
  - `case_harness_agent.synthesize_liger_wrapper` (Phase 2, gate = verify_liger_baseline)
  - `run_pipeline_a_autograd_pair`           (fusion seed, correctness-verified)
  - `openevolve-run.py`                      (evolve)
  - `triton_backward_bench_common.shape_dispatch_report` (dispatch + final program)

Design:
  - **Stages** (`--stage prep,seed,evolve,dispatch,report,all`) so the slow, multi-node `evolve`
    stage can be run separately (on parallel srun nodes) while everything else runs on one node.
  - **Idempotent**: each stage skips if its artifact already exists (use `--force` to redo).
  - **Auto-derives** the CaseSpec, the fusion `--example-input`, and the seed pair-api/task-context
    from `task_spec.py`, so the operator author only writes task_spec + forward_ref.
  - `--dry-run` prints the commands without running them.

This does NOT srun/allocate nodes itself (that stays interactive); run it *on* a configured GPU
node. For the parallel evolve, launch three `--stage evolve --group {full,small,large}` on three
nodes, then `--stage dispatch,report` on one.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import re
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

from pipeline.case_harness_agent.scaffold import CaseSpec, scaffold

REPO_ROOT = Path(__file__).resolve().parents[2]
_COMMON = "benchmark/triton_backward_bench_common"

_DTYPE_TAG = {
    "torch.float32": "f32", "torch.float16": "f16", "torch.bfloat16": "bf16",
    "torch.int64": "i64", "torch.int32": "i32",
}

# evolve group -> (evaluator, config) filenames produced by scaffold
_GROUPS = {
    "full":  ("evaluator_autograd_pair_speed_memory_min_liger.py",
              "config_autograd_pair_speed_memory_min_liger.yaml"),
    "small": ("evaluator_autograd_pair_weighted_geomean_liger_small.py",
              "config_autograd_pair_weighted_geomean_liger_small.yaml"),
    "large": ("evaluator_autograd_pair_weighted_geomean_liger_large.py",
              "config_autograd_pair_weighted_geomean_liger_large.yaml"),
}


@dataclass
class OrchestratorConfig:
    op: str
    bench_dir: Path
    forward: str | None = None            # "module.path:fn" — required for the `taskspec` stage
    api_base: str = "https://api.openai.com/v1"
    model: str = "gpt-5.5"
    api_key: str | None = None
    iterations: int = 10
    max_attempts: int = 5
    seed_max_attempts: int = 8    # some ops (layernorm's cross-row dweight/dbias) need more tries
    dispatch_timeout: int = 90
    python: str = sys.executable
    force: bool = False
    dry_run: bool = False
    example_input: str | None = None      # auto-derived if None
    liger_sources: tuple[str, ...] = ()   # auto-derived if empty
    perf_baseline: str = "auto"           # auto | liger | pytorch_autograd
    gpus: int = 1                         # 1 = groups evolve sequentially; 3 = one group per GPU


# --------------------------------------------------------------------------- helpers

def _load_task_spec(bench_dir: Path):
    for p in (str(bench_dir), str(bench_dir.parent), str(bench_dir.parent.parent)):
        if p not in sys.path:
            sys.path.insert(0, p)
    spec = importlib.util.spec_from_file_location(
        f"ts_{uuid.uuid4().hex}", str(bench_dir / "task_spec.py"))
    ts = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ts)
    return ts


def _run(cmd: list[str], cfg: OrchestratorConfig, log_path: Path | None = None,
         line_prefix: str = "    ") -> int:
    """Run a sub-stage, STREAMING its output live (so the caller sees progress in real time).

    The old version buffered the whole child via `subprocess.PIPE` and only wrote the log at the
    end, so a multi-minute stage (taskspec's LLM attempts, the wrapper gate, every evolve iteration)
    showed NOTHING until it finished. We now read the merged stdout+stderr line by line and echo
    each line immediately, while also teeing to `log_path`. openevolve attaches a console handler to
    the root logger, so its per-iteration "Iteration N ... completed in Xs" lines arrive here live.
    """
    printable = " ".join(str(c) for c in cmd)
    print(f"  $ {printable}", flush=True)
    if cfg.dry_run:
        return 0
    env = dict(os.environ)
    if cfg.api_key:
        env["OPENAI_API_KEY"] = cfg.api_key
    tail: list[str] = []
    if log_path is not None:
        # A brand-new op's bench dir may not exist yet — the sub-stage creates it, but we open the
        # log BEFORE launching. (The old buffered _run wrote the log AFTER, so it never hit this.)
        log_path.parent.mkdir(parents=True, exist_ok=True)
    log_fh = open(log_path, "w", encoding="utf-8") if log_path is not None else None
    try:
        proc = subprocess.Popen(cmd, cwd=str(REPO_ROOT), env=env, text=True, bufsize=1,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        for line in proc.stdout:  # type: ignore[union-attr]
            line = line.rstrip("\n")
            print(f"{line_prefix}{line}", flush=True)
            if log_fh is not None:
                log_fh.write(line + "\n")
                log_fh.flush()
            tail.append(line)
            if len(tail) > 25:
                tail.pop(0)
        rc = proc.wait()
    finally:
        if log_fh is not None:
            log_fh.close()
    if rc != 0:
        print(f"  [FAILED rc={rc}] tail:\n" + "\n".join(tail), flush=True)
    return rc


def _case_spec_from_task_spec(op: str, ts, package: str) -> CaseSpec:
    """Auto-build the CaseSpec from an existing task_spec (only the evaluator/config text slots
    matter here; the regime hooks in task_spec are authoritative and scaffold's copy is discarded)."""
    title = op.replace("_", " ").title().replace(" ", "")
    math_block = getattr(ts, "AUTOGRAD_PAIR_TASK_CONTEXT", "").strip() or f"{op} forward/backward."
    saved_note = (ts.correctness_hint().strip() if hasattr(ts, "correctness_hint")
                  else "Flexible saved-tensor contract: forward saves what backward needs.")
    return CaseSpec(
        op=op,
        # The import package MUST be the actual bench-dir name — the generated evaluators do
        # `from benchmark.<package> import task_spec`. Reconstructing it as f"triton_{op}_backward_bench"
        # silently points at the wrong dir whenever bench_dir != that canonical name (e.g. an `_auto`
        # variant), and the evaluator then loads a DIFFERENT op's task_spec.
        package=package,
        op_title=title,
        forward_fn=ts.AUTOGRAD_PAIR_FORWARD_FN_NAME,
        backward_fn=ts.AUTOGRAD_PAIR_BACKWARD_FN_NAME,
        api_block=ts.AUTOGRAD_PAIR_API.strip(),
        math_block=math_block,
        saved_note=saved_note,
        regime_feature_expr="0",           # placeholder; task_spec's own regime hooks are used
        regime_feature_doc="see task_spec",
        regime_split=getattr(ts, "REGIME_SPLIT", 0),
        small_note="Small-regime suite: specialize for the single-pass / low-occupancy end.",
        large_note="Large-regime suite: specialize for the streaming / high-occupancy end.",
    )


def _derive_example_input(ts) -> str:
    """Build the fusion `--example-input` spec from make_inputs on a small correctness case."""
    import torch
    case = ts.CORRECTNESS_CASES[0]
    if hasattr(ts, "seed_for_case"):
        torch.manual_seed(ts.seed_for_case(case))
    inputs = list(ts.make_inputs(torch, case))
    fwd_idx = tuple(ts.AUTOGRAD_PAIR_FORWARD_INPUT_INDICES)
    parts = []
    for i in fwd_idx:
        t = inputs[i]
        # AtenIR's autograd mode requires a flat list of TENSORS; scalar config args (eps, ...)
        # are omitted and rely on the forward's default value.
        if not isinstance(t, torch.Tensor):
            continue
        tag = _DTYPE_TAG.get(str(t.dtype), "f32")
        shape = ",".join(str(d) for d in t.shape) if t.dim() else "1"
        parts.append(f"({shape}) {tag}")
    return "[" + ", ".join(parts) + "]"


def _write_seed_inputs(ts, work: Path) -> tuple[Path, Path]:
    work.mkdir(parents=True, exist_ok=True)
    api = work / "pair_api.py"
    ctx = work / "task_context.md"
    api.write_text(ts.AUTOGRAD_PAIR_API, encoding="utf-8")
    ctx.write_text(getattr(ts, "AUTOGRAD_PAIR_TASK_CONTEXT", ""), encoding="utf-8")
    return api, ctx


def _baseline_override(cfg: OrchestratorConfig) -> str | None:
    """CLI `--perf-baseline` as an override for `_detect_baseline` ("auto" => no override)."""
    return None if cfg.perf_baseline == "auto" else cfg.perf_baseline


def _resolve_baseline(cfg: OrchestratorConfig) -> tuple[str, str]:
    """Decide the perf baseline, and HARD-FAIL if `liger` was asked for but isn't usable.

    The silent-downgrade trap: `_detect_baseline` probes `import liger_kernel.ops.<op>`, but Liger's
    module names don't always match ours (our `rmsnorm` is Liger's `rms_norm`). Under `auto` that
    probe fails, the baseline quietly becomes `pytorch_autograd`, and every speedup in the report is
    measured against a far weaker baseline than the one it claims. A 4.9x over autograd and a 4.9x
    over Liger are completely different results, so an explicit `--perf-baseline liger` that cannot
    be honored must abort, never degrade.
    """
    from pipeline.case_harness_agent.synthesize_task_spec import _detect_baseline

    baseline, title = _detect_baseline(cfg.op, _baseline_override(cfg))
    if baseline == "liger":
        _resolve_liger_sources(cfg)   # raises if the Liger sources can't be found
    return baseline, title


def _resolve_liger_sources(cfg: OrchestratorConfig) -> list[str]:
    """The Liger op sources to show the wrapper synthesizer. Raises SystemExit if unusable."""
    if cfg.liger_sources:
        srcs = [Path(s) for s in cfg.liger_sources]
        missing = [str(p) for p in srcs if not p.is_file()]
        if missing:
            raise SystemExit(
                "error: --perf-baseline liger, but these --liger-source paths do not exist:\n  "
                + "\n  ".join(missing)
            )
        return [str(p) for p in srcs]

    try:
        import liger_kernel.ops as o
    except ImportError as e:
        raise SystemExit(
            f"error: perf baseline 'liger' requested but liger_kernel is not importable ({e}). "
            "Install it, or pass --liger-source explicitly, or use --perf-baseline pytorch_autograd."
        ) from e
    d = Path(o.__file__).parent
    cand = d / f"{cfg.op}.py"
    if not cand.is_file():
        raise SystemExit(
            f"error: perf baseline 'liger' requested but no Liger module for op '{cfg.op}': "
            f"{cand} does not exist. Liger often names it differently (e.g. our 'rmsnorm' is "
            f"Liger's 'rms_norm'). Pass the real paths with --liger-source (repeatable), or use "
            f"--perf-baseline pytorch_autograd."
        )
    return [str(cand), str(d / "utils.py")]


def _liger_source_args(cfg: OrchestratorConfig) -> list[str]:
    out = []
    for s in _resolve_liger_sources(cfg):
        out += ["--liger-source", s]
    return out


# --------------------------------------------------------------------------- correctness evaluator

_CORRECTNESS_EVAL = '''import os
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
    from benchmark.{package} import task_spec  # noqa: E402
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
'''


# --------------------------------------------------------------------------- stages

def stage_prep(cfg: OrchestratorConfig, ts) -> int:
    b = cfg.bench_dir
    baseline, baseline_title = _resolve_baseline(cfg)
    wrapper = b / "strong_baselines" / f"liger_{cfg.op}.py"
    if baseline == "liger" and wrapper.exists() and not cfg.force:
        print(f"[prep] wrapper exists, skip: {wrapper}")
        return 0
    print(f"[prep] scaffold evaluators/configs (perf baseline = {baseline})")
    spec = _case_spec_from_task_spec(cfg.op, ts, cfg.bench_dir.name)
    spec.perf_baseline = baseline
    spec.baseline_title = baseline_title
    if not cfg.dry_run:
        scaffold(spec, str(b))
        # task_spec already owns the regime hooks; drop scaffold's spliceable copy
        (b / "regime_hooks.py").unlink(missing_ok=True)
        (b / "evaluator_autograd_pair.py").write_text(
            _CORRECTNESS_EVAL.format(package=spec.package), encoding="utf-8")
    if baseline != "liger":
        # No hand-optimized kernel for this op -> evolve against PyTorch autograd; nothing to wrap.
        print("[prep] no Liger kernel for this op; baseline = PyTorch autograd (no wrapper needed)")
        return 0
    print("[prep] Phase-2 Liger wrapper + gate")
    cmd = [cfg.python, "-m", "pipeline.case_harness_agent.synthesize_liger_wrapper",
           "--bench", str(b.relative_to(REPO_ROOT)), "--op", cfg.op,
           *_liger_source_args(cfg), "--max-attempts", str(cfg.max_attempts),
           "--model", cfg.model, "--api-base", cfg.api_base]
    return _run(cmd, cfg, b / ".orch_prep.log")


def stage_seed(cfg: OrchestratorConfig, ts) -> int:
    b = cfg.bench_dir
    seed_dst = b / "initial_program_autograd_pair.py"
    if seed_dst.exists() and not cfg.force:
        print(f"[seed] seed exists, skip: {seed_dst}")
        return 0
    work = b / ".orch_seed"
    api, ctx = (work / "pair_api.py", work / "task_context.md")
    if not cfg.dry_run:
        api, ctx = _write_seed_inputs(ts, work)
    example_input = cfg.example_input or (_derive_example_input(ts) if not cfg.dry_run else "[AUTO]")
    rel = b.relative_to(REPO_ROOT)
    out_dir = work / "atenir"
    cmd = [cfg.python, "-u", "-m", "pipeline.run_pipeline_a_autograd_pair",
           "--forward", f"benchmark.{rel.name}.forward_ref:{cfg.op}_forward_ref",
           "--example-input", example_input,
           "--output-dir", str(out_dir),
           "--evaluator", str(rel / "evaluator_autograd_pair.py"),
           "--model", cfg.model, "--api-base", cfg.api_base,
           "--max-attempts", str(cfg.seed_max_attempts),
           "--pair-api-file", str(api), "--task-context-file", str(ctx)]
    rc = _run(cmd, cfg, b / ".orch_seed.log")
    best = out_dir / "best" / "initial_program_autograd_pair.py"
    if rc == 0 and not cfg.dry_run and best.exists():
        seed_dst.write_text(best.read_text(encoding="utf-8"), encoding="utf-8")
        print(f"[seed] landed -> {seed_dst}")
    return rc


def _evolve_cmd(cfg: OrchestratorConfig, group: str):
    """Build the openevolve command + log path for one group. Returns None if it can be skipped."""
    b = cfg.bench_dir
    ev, conf = _GROUPS[group]
    out = b / f"evolve_{group}_r1"
    best = out / "best_program.py"
    if best.exists() and not cfg.force:
        print(f"[evolve:{group}] best exists, skip: {best}")
        return None
    rel = b.relative_to(REPO_ROOT)
    cmd = [cfg.python, "openevolve-run.py",
           str(rel / "initial_program_autograd_pair.py"), str(rel / ev),
           "--config", str(rel / conf), "--iterations", str(cfg.iterations),
           "--output", str(out), "--save-best-to", str(best)]
    return cmd, b / f".orch_evolve_{group}.log"


def stage_evolve(cfg: OrchestratorConfig, group: str) -> int:
    built = _evolve_cmd(cfg, group)
    if built is None:
        return 0
    cmd, log = built
    return _run(cmd, cfg, log)


def stage_evolve_parallel(cfg: OrchestratorConfig, groups: list[str]) -> int:
    """Run the groups CONCURRENTLY, one per GPU (CUDA_VISIBLE_DEVICES=0,1,2,...).

    autodiff runs as a single process on one node, so 'parallel across N GPUs' is just N child
    processes each pinned to a distinct GPU — no multi-node scheduling, no srun from inside the
    pipeline. Each child's output goes to its own .orch_evolve_<group>.log; while they run we print
    a combined per-group iteration line from openevolve's live logs (interleaving 3 streams onto one
    console would be unreadable).
    """
    env0 = dict(os.environ)
    if cfg.api_key:
        env0["OPENAI_API_KEY"] = cfg.api_key
    procs: dict[str, subprocess.Popen] = {}
    logs: dict[str, object] = {}
    for i, g in enumerate(groups):
        built = _evolve_cmd(cfg, g)
        if built is None:
            continue
        cmd, log_path = built
        log_path.parent.mkdir(parents=True, exist_ok=True)
        env = dict(env0)
        env["CUDA_VISIBLE_DEVICES"] = str(i)
        print(f"  [evolve:{g}] -> GPU {i}: {' '.join(str(c) for c in cmd)}", flush=True)
        if cfg.dry_run:
            continue
        fh = open(log_path, "w", encoding="utf-8")
        logs[g] = fh
        procs[g] = subprocess.Popen(cmd, cwd=str(REPO_ROOT), env=env, text=True,
                                    stdout=fh, stderr=subprocess.STDOUT)
    if cfg.dry_run or not procs:
        return 0

    # tqdm-style single-line status that OVERWRITES in place (carriage return), so 3 groups x N
    # iterations don't scroll into dozens of lines. We drive \r by hand rather than use tqdm: this
    # stream is tee'd to a file/pane (not a tty), where tqdm disables its bar. Each group shows
    # iteration progress + the current program's score + the best score so far, updated only when a
    # group actually advances.
    import time as _t

    def _one(g: str) -> str:
        if g not in procs:
            return f"{g} skip"
        it, cur, best = _evolve_progress(cfg.bench_dir / f"evolve_{g}_r1")
        tag = "done" if procs[g].poll() is not None else f"{it}/{cfg.iterations}"
        sc = f" cur={cur:.3f} max={best:.3f}" if cur is not None and best is not None else ""
        return f"{g} {tag}{sc}"

    running = dict(procs)
    last, width = None, 0
    while running:
        _t.sleep(10)
        line = "   ".join(_one(g) for g in groups)
        if line != last:                   # only repaint when some group advanced
            pad = " " * max(0, width - len(line))
            print(f"\r  [evolve||] {line}{pad}", end="", flush=True)
            width, last = max(width, len(line)), line
        running = {g: p for g, p in procs.items() if p.poll() is None}
    print(flush=True)                      # finalize the transient line with a newline

    rc = 0
    for g, p in procs.items():
        logs[g].close()
        code = p.wait()
        if code != 0:
            tail = "\n".join((cfg.bench_dir / f".orch_evolve_{g}.log").read_text(
                errors="ignore").splitlines()[-15:])
            print(f"  [evolve:{g}] FAILED rc={code}\n{tail}", flush=True)
            rc = 1
        else:
            it, _cur, best = _evolve_progress(cfg.bench_dir / f"evolve_{g}_r1")
            print(f"  [evolve:{g}] done  ({it} iters, best score {best:.3f})"
                  if best is not None else f"  [evolve:{g}] done", flush=True)
    return rc


def _evolve_progress(out_dir: Path) -> tuple[int, float | None, float | None]:
    """From openevolve's live log: (latest iteration, current program score, best score so far).

    `current` is the most recent per-iteration `Metrics: combined_score=...`; `best` is the running
    max over them. Returns (0, None, None) before the first iteration completes.
    """
    logs = sorted(out_dir.glob("logs/openevolve_*.log"))
    if not logs:
        return 0, None, None
    it, cur, best = 0, None, None
    try:
        for line in logs[-1].read_text(errors="ignore").splitlines():
            m = re.search(r"Iteration (\d+):", line)
            if m:
                it = max(it, int(m.group(1)))
            if "Metrics: combined_score=" in line:
                m = re.search(r"combined_score=(-?[\d.]+)", line)
                if m:
                    cur = float(m.group(1))
                    best = cur if best is None else max(best, cur)
    except OSError:
        pass
    return it, cur, best


def _best_program(cfg: OrchestratorConfig, group: str) -> Path | None:
    """Prefer --save-best-to output; fall back to the newest checkpoint's best (node got cut off)."""
    b = cfg.bench_dir
    direct = b / f"evolve_{group}_r1" / "best_program.py"
    if direct.exists():
        return direct
    ckpts = sorted((b / f"evolve_{group}_r1" / "checkpoints").glob("checkpoint_*"),
                   key=lambda p: int(p.name.split("_")[-1]) if p.name.split("_")[-1].isdigit() else -1)
    for c in reversed(ckpts):
        bp = c / "best_program.py"
        if bp.exists():
            return bp
    return None


def stage_dispatch(cfg: OrchestratorConfig) -> int:
    b = cfg.bench_dir
    progs = []
    for g in ("full", "small", "large"):
        bp = _best_program(cfg, g)
        if bp is None:
            print(f"[dispatch] missing best for group {g}; run evolve first")
            if not cfg.dry_run:
                return 1
            bp = b / f"evolve_{g}_r1" / "best_program.py"
        progs += ["--program", f"{g}_r1={bp}"]
    cmd = [cfg.python, f"{_COMMON}/shape_dispatch_report.py",
           "--bench", str(b.relative_to(REPO_ROOT)), *progs,
           "--timeout", str(cfg.dispatch_timeout),
           "--out-report", str(b / f"{cfg.op}_dispatch_report.json"),
           "--out-program", str(b / f"{cfg.op}_final_dispatched.py")]
    return _run(cmd, cfg, b / ".orch_dispatch.log")


def stage_report(cfg: OrchestratorConfig, ts) -> int:
    b = cfg.bench_dir
    rj = b / f"{cfg.op}_dispatch_report.json"
    if not rj.exists():
        print(f"[report] no dispatch report at {rj}; run dispatch first")
        return 0 if cfg.dry_run else 1
    if cfg.dry_run:
        print(f"[report] would generate RESULTS_{cfg.op}_vs_<baseline>.md from {rj.name}")
        return 0
    r = json.loads(rj.read_text())
    md = _render_results(cfg.op, ts, r)
    out = b / f"RESULTS_{cfg.op}_vs_{_baseline_slug(r)}.md"
    out.write_text(md, encoding="utf-8")
    print(f"[report] wrote {out}")
    return 0


# The baseline is whatever `dispatch` actually measured against (liger when the op has a Liger
# kernel, plain PyTorch autograd otherwise). Never hard-code "Liger" into the report: a 4.9x over
# autograd and a 4.9x over Liger are completely different claims.
_BASELINE_TITLE = {"liger": "Liger", "pytorch_autograd": "PyTorch autograd"}


def _baseline_slug(r: dict) -> str:
    return str(r.get("baseline", "liger"))


def _baseline_title(r: dict) -> str:
    slug = _baseline_slug(r)
    return _BASELINE_TITLE.get(slug, slug)


def _render_results(op: str, ts, r: dict) -> str:
    base = _baseline_title(r)
    disp = r.get("threshold_dispatch_geomean")
    full = r.get("full_geomean")
    collapsed = r.get("regime_collapsed")
    thr = r.get("best_threshold")
    pbf = r.get("per_program_geomean_full", {})
    pbb = r.get("per_program_geomean_bwd", {})
    fb = r.get("full_best")
    verdict = ("collapsed to a single program (no useful regime)" if collapsed
               else f"a real regime; deploy the shape dispatcher (threshold on regime_feature = {thr})")
    gain = ""
    if disp and full:
        gain = f" ({(disp/full - 1)*100:+.1f}% vs the single full-trained program)"
    lines = [
        f"# {op} autograd-pair — benchmark results vs {base}",
        "",
        "Auto-generated by the Phase-4 orchestrator from the shape-dispatch report.",
        f"Performance baseline: **{base}**.",
        "",
        f"## Headline (geomean vs {base}, full suite)",
        "",
        "| deployment | full-step geomean |",
        "| --- | ---: |",
        f"| full-trained single program | {full:.3f} |" if full else "",
        f"| **{{small,large}} + threshold dispatch** | **{disp:.3f}**{gain} |" if disp else "",
        f"| regime collapsed? | **{'Yes' if collapsed else 'No'}** |",
        "",
        f"**Verdict:** {op} is {verdict}.",
        "",
        f"## Per-program geomean vs {base}",
        "",
        "| program | full-step | backward |",
        "| --- | ---: | ---: |",
    ]
    for k in sorted(pbf):
        lines.append(f"| {k} | {pbf.get(k, float('nan')):.3f} | {pbb.get(k, float('nan')):.3f} |")
    lines += [
        "",
        f"Best per regime: full=`{fb}`, small=`{r.get('small_best')}`, large=`{r.get('large_best')}`.",
        "",
        "Deployed program: "
        f"`{op}_final_dispatched.py`; full numbers in `{op}_dispatch_report.json`.",
        "",
    ]
    return "\n".join(x for x in lines if x is not None) + "\n"


# --------------------------------------------------------------------------- driver

# The regime (axis + split) is decided INSIDE `taskspec`, in the same LLM call that designs the
# benchmark shapes — so the shapes are swept ALONG the axis and both specialists are trainable.
# (An earlier design had a separate `regime` stage that measured a strategy-crossover on GPU; it was
# removed: two one-shot LLM kernels differ mostly in implementation quality, so the measurement was
# noise-dominated and once even overruled the correct axis. The axis is a property of the backward's
# math, not of a benchmark — reading the math is the right tool.)
_STAGE_ORDER = ["taskspec", "prep", "seed", "evolve", "dispatch", "report"]


def stage_taskspec(cfg: OrchestratorConfig) -> int:
    """Phase-3 Part-1: synthesize task_spec.py + forward_ref.py from a PyTorch forward."""
    if (cfg.bench_dir / "task_spec.py").exists() and not cfg.force:
        print(f"[taskspec] task_spec exists, skip: {cfg.bench_dir/'task_spec.py'}")
        return 0
    if not cfg.forward:
        print("[taskspec] --forward module:fn is required for this stage")
        return 1
    baseline, _ = _resolve_baseline(cfg)   # fail fast: don't synthesize against a downgraded baseline
    cmd = [cfg.python, "-m", "pipeline.case_harness_agent.synthesize_task_spec",
           "--op", cfg.op, "--forward", cfg.forward,
           "--bench-dir", str(cfg.bench_dir),
           "--model", cfg.model, "--api-base", cfg.api_base,
           "--max-attempts", str(cfg.max_attempts),
           "--perf-baseline", baseline]
    return _run(cmd, cfg, cfg.bench_dir / ".orch_taskspec.log")


_STAGE_DESC = {
    "taskspec": "synthesize task_spec + forward_ref from the PyTorch forward",
    "prep":     "scaffold evaluators/configs + synthesize & gate the Liger baseline wrapper",
    "seed":     "fusion agent writes the first fused backward kernel",
    "evolve":   "OpenEvolve evolves the {n} group(s) — SEQUENTIALLY on this GPU",
    "dispatch": "measure every program on the full shape grid, pick the regime threshold",
    "report":   "render RESULTS_<op>_vs_<baseline>.md",
}


def _banner(msg: str) -> None:
    print(f"\n{'='*72}\n[{time.strftime('%H:%M:%S')}] {msg}\n{'='*72}", flush=True)


def orchestrate(cfg: OrchestratorConfig, stages: list[str], groups: list[str]) -> int:
    t0 = time.time()
    active = [s for s in _STAGE_ORDER if s in stages]
    _banner(f"autodiff op={cfg.op}  stages={active}")
    baseline, _ = _resolve_baseline(cfg)
    print(f"perf baseline: {baseline} (--perf-baseline {cfg.perf_baseline})", flush=True)
    if "taskspec" in stages:
        _banner(f"STAGE taskspec — {_STAGE_DESC['taskspec']}")
        if stage_taskspec(cfg):
            return 1
    if not (cfg.bench_dir / "task_spec.py").exists() and cfg.dry_run:
        print("[dry-run] task_spec.py not written yet; later stages need it — stopping here")
        return 0
    ts = _load_task_spec(cfg.bench_dir)
    for st in _STAGE_ORDER:
        if st not in stages or st == "taskspec":
            continue
        if st == "evolve":
            parallel = cfg.gpus >= 3 and len(groups) > 1
            mode = f"PARALLEL across {min(cfg.gpus, len(groups))} GPUs" if parallel \
                else "SEQUENTIALLY on this GPU"
            _banner(f"STAGE evolve — {len(groups)} group(s), {mode}, x{cfg.iterations} iterations each")
            if parallel:
                if stage_evolve_parallel(cfg, groups):
                    return 1
            else:
                for i, g in enumerate(groups, 1):
                    _banner(f"  evolve group {g}  ({i}/{len(groups)})  x{cfg.iterations} iterations")
                    if stage_evolve(cfg, g):
                        return 1
            continue
        _banner(f"STAGE {st} — {_STAGE_DESC.get(st, '')}")
        rc = {"prep": lambda: stage_prep(cfg, ts),
              "seed": lambda: stage_seed(cfg, ts),
              "dispatch": lambda: stage_dispatch(cfg),
              "report": lambda: stage_report(cfg, ts)}[st]()
        if rc:
            return 1
    _banner(f"DONE op={cfg.op}  ({time.time()-t0:.0f}s total)")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description="Phase-4 end-to-end case orchestrator")
    ap.add_argument("--op", required=True, help="operator, e.g. geglu")
    ap.add_argument("--forward", default=None,
                    help="module.path:fn of a PyTorch forward — required for the taskspec stage "
                         "(omit if task_spec.py + forward_ref.py already exist)")
    ap.add_argument("--stage", default="all",
                    help="comma list of taskspec,prep,seed,evolve,dispatch,report or 'all'")
    ap.add_argument("--group", default="full,small,large",
                    help="evolve groups to run (comma list)")
    ap.add_argument("--iterations", type=int, default=10)
    ap.add_argument("--gpus", type=int, default=1, choices=[1, 3],
                    help="1 = groups evolve sequentially on one GPU; 3 = one group per GPU in "
                         "parallel (needs 3 visible GPUs, e.g. a --gpus=3 allocation)")
    ap.add_argument("--max-attempts", type=int, default=5)
    ap.add_argument("--dispatch-timeout", type=int, default=90)
    ap.add_argument("--api-base", default="https://api.openai.com/v1")
    ap.add_argument("--model", default="gpt-5.5")
    ap.add_argument("--api-key", default=None)
    ap.add_argument("--example-input", default=None, help="override the auto-derived fusion spec")
    ap.add_argument("--perf-baseline", default="auto", choices=["auto", "liger", "pytorch_autograd"],
                    help="what to measure speedups against. 'auto' probes for a Liger kernel and "
                         "falls back to PyTorch autograd; 'liger' HARD-FAILS if Liger can't be "
                         "resolved rather than silently downgrading")
    ap.add_argument("--liger-source", action="append", default=[], dest="liger_sources",
                    help="path to a Liger op source file (repeatable). Needed when Liger's module "
                         "name differs from --op (e.g. op 'rmsnorm' is Liger's 'rms_norm')")
    ap.add_argument("--force", action="store_true", help="redo stages even if artifacts exist")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv or sys.argv[1:])

    bench_dir = REPO_ROOT / "benchmark" / f"triton_{args.op}_backward_bench"
    stages = _STAGE_ORDER if args.stage == "all" else [s.strip() for s in args.stage.split(",")]
    groups = [g.strip() for g in args.group.split(",")]
    # task_spec is an input for every stage except `taskspec`, which generates it.
    if "taskspec" not in stages and not (bench_dir / "task_spec.py").exists():
        print(f"error: {bench_dir}/task_spec.py not found — run with --stage taskspec --forward mod:fn")
        return 2
    if "taskspec" in stages and not args.forward and not (bench_dir / "task_spec.py").exists():
        print("error: --forward module:fn is required to generate task_spec")
        return 2
    cfg = OrchestratorConfig(
        op=args.op, bench_dir=bench_dir, forward=args.forward,
        api_base=args.api_base, model=args.model,
        api_key=args.api_key, iterations=args.iterations, max_attempts=args.max_attempts,
        dispatch_timeout=args.dispatch_timeout, force=args.force, dry_run=args.dry_run,
        example_input=args.example_input,
        liger_sources=tuple(args.liger_sources), perf_baseline=args.perf_baseline,
        gpus=args.gpus,
    )
    return orchestrate(cfg, stages, groups)


if __name__ == "__main__":
    raise SystemExit(main())
