"""Shared Triton-native ncu profiling core for autograd-pair benchmarks.

Generic re-implementation of triton_layernorm_backward_bench/ncu_profile.py,
driven entirely by each benchmark's task_spec conventions instead of
hard-coded function names and input unpacking:

    AUTOGRAD_PAIR_FORWARD_FN_NAME            candidate forward symbol
    AUTOGRAD_PAIR_BACKWARD_FN_NAME           candidate backward symbol
    AUTOGRAD_PAIR_COTANGENT_INDEX            index of dout in make_inputs() output (default 0)
    AUTOGRAD_PAIR_FORWARD_INPUT_INDICES      indices of forward args in make_inputs() output
    AUTOGRAD_PAIR_BACKWARD_EXTRA_INPUT_INDICES  extra backward args (e.g. eps), default ()

All TestCases are assumed to be (rows, cols, dtype_name, atol_value, rtol_value),
which every autograd-pair bench in this repo uses.

Profiling is split into two subprocesses: phase 1 (no ncu) generates inputs and
warms Triton's JIT cache, then saves the exact tensors; phase 2 (under ncu)
loads them and makes exactly one forward+backward call, so the captured report
contains only the candidate's own kernels.
"""

from __future__ import annotations

import importlib
import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

# Kept short on purpose: collected with `--metrics` (1-2 replay passes), not
# `--set full` (~45 passes) -- per-evaluation ncu cost must stay small enough
# to run inside OpenEvolve's evaluator timeout.
DEFAULT_METRICS: tuple[str, ...] = (
    "gpu__time_duration.sum",
    "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "dram__throughput.avg.pct_of_peak_sustained_elapsed",
    "dram__bytes_read.sum.pct_of_peak_sustained_elapsed",
    "sm__warps_active.avg.pct_of_peak_sustained_active",  # achieved occupancy
    "sm__maximum_warps_per_active_cycle_pct",  # theoretical occupancy
    "launch__registers_per_thread",
    "launch__waves_per_multiprocessor",
    "launch__grid_size",
    "device__attribute_multiprocessor_count",
    "smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio",
    "smsp__sass_inst_executed_op_local_ld.sum",
    "smsp__sass_inst_executed_op_local_st.sum",
)

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

    forward_fn = getattr(module, task_spec.AUTOGRAD_PAIR_FORWARD_FN_NAME)
    backward_fn = getattr(module, task_spec.AUTOGRAD_PAIR_BACKWARD_FN_NAME)

    case = task_spec.TestCase({rows}, {cols}, {dtype_name!r}, {atol}, {rtol})

    _COT_IDX = int(getattr(task_spec, "AUTOGRAD_PAIR_COTANGENT_INDEX", 0))
    _FWD_IDX = tuple(task_spec.AUTOGRAD_PAIR_FORWARD_INPUT_INDICES)
    _EXTRA_IDX = tuple(getattr(task_spec, "AUTOGRAD_PAIR_BACKWARD_EXTRA_INPUT_INDICES", ()))
    """)

# Phase 1, run WITHOUT ncu: generates inputs (RNG kernels must not show up in
# the profile) and runs `warmup` iterations so Triton's JIT/autotune cache is
# hot before ncu attaches. The exact tensors are saved for phase 2.
_WARMUP_TEMPLATE = _IMPORT_PREAMBLE + textwrap.dedent("""\
    torch.manual_seed(task_spec.seed_for_case(case))
    inputs = list(task_spec.make_inputs(torch, case))

    dout = inputs[_COT_IDX]
    fwd_args = [inputs[i] for i in _FWD_IDX]
    extra_args = [inputs[i] for i in _EXTRA_IDX]

    for _ in range({warmup}):
        y, saved = forward_fn(*fwd_args)
        backward_fn(dout, saved, *extra_args)
    torch.cuda.synchronize()

    torch.save(inputs, {inputs_path!r})
    """)

# Phase 2, run UNDER ncu: loads the exact inputs phase 1 produced and makes
# exactly one forward+backward call -- nothing else in this process's lifetime
# for ncu to capture besides the candidate's own kernels.
_PROFILED_TEMPLATE = _IMPORT_PREAMBLE + textwrap.dedent("""\
    inputs = torch.load({inputs_path!r}, map_location="cuda")

    dout = inputs[_COT_IDX]
    fwd_args = [inputs[i] for i in _FWD_IDX]
    extra_args = [inputs[i] for i in _EXTRA_IDX]

    y, saved = forward_fn(*fwd_args)
    backward_fn(dout, saved, *extra_args)
    torch.cuda.synchronize()
    """)


@dataclass
class KernelProfile:
    name: str
    duration_ns: float | None
    metrics: dict[str, Any]
    top_rule: dict[str, Any] | None


@dataclass
class ProfileResult:
    ok: bool
    error: str | None = None
    kernels: list[KernelProfile] = field(default_factory=list)
    aggregate: dict[str, Any] = field(default_factory=dict)
    top_rule: dict[str, Any] | None = None
    report_path: str | None = None

    def to_json(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "error": self.error,
            "aggregate": self.aggregate,
            "top_rule": self.top_rule,
            "report_path": self.report_path,
            "kernels": [
                {
                    "name": k.name,
                    "duration_ns": k.duration_ns,
                    "metrics": k.metrics,
                    "top_rule": k.top_rule,
                }
                for k in self.kernels
            ],
        }


def _ncu_report_dir_next_to_binary() -> Path | None:
    """Derive the extras/python dir from the `ncu` binary actually on PATH."""
    ncu_bin = shutil.which("ncu")
    if not ncu_bin:
        return None
    p = Path(ncu_bin).resolve()
    for ancestor in (p.parent, p.parent.parent, p.parent.parent.parent):
        candidate = ancestor / "extras" / "python"
        if (candidate / "ncu_report.py").exists():
            return candidate
    return None


def _find_ncu_report_module():
    """Locate the ncu_report module shipped with Nsight Compute."""
    try:
        return importlib.import_module("ncu_report")
    except ImportError:
        pass

    candidates: list[Path] = []
    env_path = os.environ.get("NCU_PYTHON_PATH")
    if env_path:
        candidates.append(Path(env_path))

    beside_binary = _ncu_report_dir_next_to_binary()
    if beside_binary:
        candidates.append(beside_binary)

    for root in ("/usr/local", "/opt", "/opt/nvidia", "/opt/cuda"):
        p = Path(root)
        if not p.is_dir():
            continue
        candidates.extend(p.glob("cuda*/nsight-compute-*/extras/python"))
        candidates.extend(p.glob("nsight-compute-*/extras/python"))
        candidates.extend(p.glob("nsight-compute/*/extras/python"))
        candidates.extend(p.glob("hpc_sdk/*/*/profilers/Nsight_Compute/extras/python"))

    for c in candidates:
        if c.is_dir() and str(c) not in sys.path:
            sys.path.insert(0, str(c))
            try:
                return importlib.import_module("ncu_report")
            except ImportError:
                continue

    raise RuntimeError(
        "Could not import ncu_report. Set NCU_PYTHON_PATH to the directory "
        "containing ncu_report.py (under <cuda>/nsight-compute-*/extras/python, "
        "or <hpc_sdk>/profilers/Nsight_Compute/extras/python)."
    )


def _iter_indexed(obj, by_idx_name: str, count_name: str) -> Iterator[Any]:
    """Yield obj.<by_idx_name>(i) for i = 0, 1, ...

    Uses obj.<count_name>() when available. The probing fallback MUST treat a
    None return as end-of-sequence: ncu_report returns None (rather than
    raising) for out-of-range indices, so a bare while/try loop spins forever.
    """
    by_idx = getattr(obj, by_idx_name)
    if hasattr(obj, count_name):
        try:
            n = getattr(obj, count_name)()
            for i in range(n):
                yield by_idx(i)
            return
        except Exception:
            pass
    i = 0
    while True:
        try:
            item = by_idx(i)
        except Exception:
            return
        if item is None:
            return
        yield item
        i += 1


def _render_script(template: str, *, repo_root: Path, bench_dir: Path, program_path: str, case, **extra) -> str:
    return template.format(
        repo_root=str(repo_root),
        bench_dir=str(bench_dir),
        program_path=str(Path(program_path).resolve()),
        rows=case.rows,
        cols=case.cols,
        dtype_name=case.dtype_name,
        atol=case.atol_value,
        rtol=case.rtol_value,
        **extra,
    )


def _safe_value(action, name: str):
    try:
        return action[name].value()
    except Exception:
        return None


def run_ncu_profile(
    program_path: str,
    case,
    *,
    bench_dir: str | Path,
    metrics: tuple[str, ...] = DEFAULT_METRICS,
    warmup: int = 5,
    timeout: int = 120,
    ncu_bin: str = "ncu",
    python_bin: str | None = None,
    warmup_template: str | None = None,
    profiled_template: str | None = None,
) -> ProfileResult:
    """Profile one forward+backward call of `program_path` at `case` with ncu.

    Best-effort: any failure (ncu missing, no perf-counter permission, timeout,
    parse error) returns ProfileResult(ok=False, error=...) rather than raising.

    warmup_template / profiled_template default to the autograd-pair harness;
    benches with a different candidate contract (e.g. a standalone backward
    function) can pass their own script templates using the same placeholders.
    """
    warmup_template = warmup_template or _WARMUP_TEMPLATE
    profiled_template = profiled_template or _PROFILED_TEMPLATE
    bench_dir = Path(bench_dir).resolve()
    repo_root = bench_dir.parents[1]
    python_bin = python_bin or os.environ.get("NCU_PROFILE_PYTHON") or sys.executable
    try:
        ncu_report = _find_ncu_report_module()
    except RuntimeError as exc:
        return ProfileResult(ok=False, error=str(exc))

    with tempfile.TemporaryDirectory(prefix="ncu_profile_") as tmp:
        tmp_dir = Path(tmp)
        inputs_path = tmp_dir / "inputs.pt"

        # Phase 1 (no ncu): generate inputs, warm the JIT cache, save tensors.
        warmup_script = tmp_dir / "warmup.py"
        warmup_script.write_text(
            _render_script(
                warmup_template,
                repo_root=repo_root,
                bench_dir=bench_dir,
                program_path=program_path,
                case=case,
                warmup=warmup,
                inputs_path=str(inputs_path),
            ),
            encoding="utf-8",
        )
        try:
            warmup_completed = subprocess.run(
                [python_bin, str(warmup_script)],
                cwd=str(bench_dir),
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            return ProfileResult(ok=False, error=f"warmup phase timed out after {timeout}s")
        except FileNotFoundError:
            return ProfileResult(ok=False, error=f"'{python_bin}' not found")
        if warmup_completed.returncode != 0 or not inputs_path.exists():
            return ProfileResult(
                ok=False,
                error=f"warmup phase failed (code={warmup_completed.returncode}): {warmup_completed.stderr[-2000:]}",
            )

        # Phase 2 (under ncu): one forward+backward call on the saved inputs.
        profiled_script = tmp_dir / "profiled.py"
        profiled_script.write_text(
            _render_script(
                profiled_template,
                repo_root=repo_root,
                bench_dir=bench_dir,
                program_path=program_path,
                case=case,
                inputs_path=str(inputs_path),
            ),
            encoding="utf-8",
        )
        report_base = tmp_dir / "report"

        cmd = [
            ncu_bin,
            "--metrics",
            ",".join(metrics),
            "--target-processes",
            "all",
            "-o",
            str(report_base),
            python_bin,
            str(profiled_script),
        ]
        try:
            completed = subprocess.run(
                cmd,
                cwd=str(bench_dir),
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            return ProfileResult(ok=False, error=f"ncu timed out after {timeout}s")
        except FileNotFoundError:
            return ProfileResult(ok=False, error=f"'{ncu_bin}' not found on PATH")

        report_path = report_base.with_suffix(".ncu-rep")
        if completed.returncode != 0 or not report_path.exists():
            return ProfileResult(
                ok=False,
                error=f"ncu failed (code={completed.returncode}): {completed.stderr[-2000:]}",
            )

        try:
            report = ncu_report.load_report(str(report_path))
        except Exception as exc:
            return ProfileResult(ok=False, error=f"failed to load .ncu-rep: {exc}")

        kernels: list[KernelProfile] = []
        for rng in _iter_indexed(report, "range_by_idx", "num_ranges"):
            for action in _iter_indexed(rng, "action_by_idx", "num_actions"):
                kmetrics = {name: _safe_value(action, name) for name in metrics}
                duration = kmetrics.get("gpu__time_duration.sum")

                top_rule = None
                try:
                    for rule in action.rule_results_as_dicts():
                        speedup = rule.get("estimated_speedup_pct")
                        if speedup is None:
                            continue
                        if top_rule is None or speedup > top_rule.get("estimated_speedup_pct", -1):
                            top_rule = rule
                except Exception:
                    pass

                kernels.append(
                    KernelProfile(
                        name=action.name(),
                        duration_ns=duration,
                        metrics=kmetrics,
                        top_rule=top_rule,
                    )
                )

        # Copy the report out before the TemporaryDirectory is deleted.
        persistent_report_path: str | None = None
        if kernels:
            try:
                fd, persistent_path = tempfile.mkstemp(suffix=".ncu-rep", prefix="ncu_saved_")
                os.close(fd)
                shutil.copy2(str(report_path), persistent_path)
                persistent_report_path = persistent_path
            except Exception:
                pass  # optimizer will fall back to pre-extracted metrics

    if not kernels:
        return ProfileResult(ok=False, error="ncu produced no kernel actions")

    aggregate: dict[str, Any] = {"total_kernels": len(kernels)}
    total_time = sum(k.duration_ns for k in kernels if k.duration_ns)
    aggregate["total_duration_ns"] = total_time or None

    for name in metrics:
        if name == "gpu__time_duration.sum":
            continue
        weighted_sum, weight_sum, plain_sum, plain_count = 0.0, 0.0, 0.0, 0
        for k in kernels:
            v = k.metrics.get(name)
            if v is None:
                continue
            plain_sum += v
            plain_count += 1
            w = k.duration_ns or 0.0
            weighted_sum += v * w
            weight_sum += w
        if plain_count == 0:
            aggregate[name] = None
        elif weight_sum > 0:
            # Time-weighted average: a long-running kernel should dominate the
            # aggregate reading over a microsecond-scale reduction kernel.
            aggregate[name] = weighted_sum / weight_sum
        else:
            aggregate[name] = plain_sum / plain_count

    overall_top_rule = None
    for k in kernels:
        if k.top_rule and (
            overall_top_rule is None
            or k.top_rule.get("estimated_speedup_pct", -1)
            > overall_top_rule.get("estimated_speedup_pct", -1)
        ):
            overall_top_rule = k.top_rule

    return ProfileResult(
        ok=True,
        kernels=kernels,
        aggregate=aggregate,
        top_rule=overall_top_rule,
        report_path=persistent_report_path,
    )


def derive_flags(aggregate: dict[str, Any]) -> dict[str, Any]:
    """Flat view of the raw aggregate numbers for the evaluator's ncu_* metrics."""
    local_ld = aggregate.get("smsp__sass_inst_executed_op_local_ld.sum") or 0
    local_st = aggregate.get("smsp__sass_inst_executed_op_local_st.sum") or 0
    return {
        "register_spilling": bool(local_ld > 0 or local_st > 0),
        "occupancy_pct": aggregate.get("sm__warps_active.avg.pct_of_peak_sustained_active"),
        "long_scoreboard_stall_ratio": aggregate.get(
            "smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio"
        ),
        "sm_throughput_pct": aggregate.get("sm__throughput.avg.pct_of_peak_sustained_elapsed"),
        "dram_throughput_pct": aggregate.get("dram__throughput.avg.pct_of_peak_sustained_elapsed"),
    }
