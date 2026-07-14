"""Evaluator for LayerNorm autograd-pair candidates with saved tensors.

This evaluator is intentionally separate from ``evaluator.py``.  The existing
benchmark keeps the standalone backward API:

    layernorm_backward_triton(dy, x, weight, bias, eps)

This file evaluates a richer contract where the candidate controls both the
forward saved tensors and the backward that consumes them:

    layernorm_forward_with_saved(x, weight, bias, eps) -> (y, saved_tensors)
    layernorm_backward_from_saved(dy, saved_tensors, eps) -> (dx, dweight, dbias)
"""

from __future__ import annotations

from contextlib import contextmanager
import importlib.util
import json
import math
import os
import statistics
import sys
import tempfile
import traceback
import uuid
from pathlib import Path
from typing import Any, Callable

import torch

BENCHMARK_DIR = Path(__file__).resolve().parent
REPO_ROOT = BENCHMARK_DIR.parents[1]
for _path in (REPO_ROOT, BENCHMARK_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

try:
    from benchmark.triton_layernorm_backward_bench import task_spec  # type: ignore
except ImportError:  # pragma: no cover - direct script execution
    import task_spec  # type: ignore

try:
    from openevolve.evaluation_result import EvaluationResult
except Exception:  # pragma: no cover
    from dataclasses import dataclass, field

    @dataclass
    class EvaluationResult:
        metrics: dict[str, float]
        artifacts: dict[str, str | bytes] = field(default_factory=dict)


FORWARD_FN_NAME = "layernorm_forward_with_saved"
BACKWARD_FN_NAME = "layernorm_backward_from_saved"
SCORE_MODE = os.environ.get("AUTOGRAD_PAIR_SCORE_MODE", "speed_only")
FULL_STEP_WEIGHT = float(os.environ.get("AUTOGRAD_PAIR_FULL_STEP_WEIGHT", "0.5"))
MEMORY_PENALTY_WEIGHT = float(os.environ.get("AUTOGRAD_PAIR_MEMORY_PENALTY_WEIGHT", "0.05"))
PERFORMANCE_BASELINE = os.environ.get("AUTOGRAD_PAIR_PERF_BASELINE", "pytorch_autograd").strip().lower()
CAPTURE_NATIVE_OUTPUT = os.environ.get("AUTOGRAD_PAIR_CAPTURE_NATIVE_OUTPUT", "1").lower() not in ("0", "false", "no")
NATIVE_OUTPUT_TAIL_BYTES = int(os.environ.get("AUTOGRAD_PAIR_NATIVE_OUTPUT_TAIL_BYTES", "65536"))
GEOMEAN_WEIGHTS_ENV = os.environ.get("AUTOGRAD_PAIR_GEOMEAN_WEIGHTS", "").strip()
WORST_CASE_GUARD = os.environ.get("AUTOGRAD_PAIR_WORST_CASE_GUARD", "1").lower() not in ("0", "false", "no")


def _env_int(name: str, default: int, *, minimum: int) -> int:
    value = int(os.environ.get(name, str(default)))
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {value}")
    return value


BENCHMARK_WARMUP = _env_int("LAYERNORM_BENCHMARK_WARMUP", 10, minimum=0)
BENCHMARK_REPS = _env_int("LAYERNORM_BENCHMARK_REPS", 50, minimum=1)


def _json(data: dict[str, Any]) -> str:
    return json.dumps(data, indent=2, sort_keys=True)


def _result(metrics: dict[str, float], artifacts: dict[str, Any]) -> EvaluationResult:
    return EvaluationResult(metrics=metrics, artifacts={k: _json(v) for k, v in artifacts.items()})


@contextmanager
def _capture_native_output():
    """Capture native compiler dumps from Triton/ptxas during candidate execution.

    Triton can print very large PTX reproducers directly to process stdout/stderr
    for invalid generated kernels. Capturing file descriptors keeps OpenEvolve's
    terminal output small and avoids slow terminal writes for doomed candidates.
    """
    if not CAPTURE_NATIVE_OUTPUT:
        yield None
        return

    sys.stdout.flush()
    sys.stderr.flush()
    fd, path = tempfile.mkstemp(prefix="layernorm_autograd_pair_native_", suffix=".log")
    saved_stdout = os.dup(1)
    saved_stderr = os.dup(2)
    try:
        os.dup2(fd, 1)
        os.dup2(fd, 2)
        os.close(fd)
        yield path
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        os.dup2(saved_stdout, 1)
        os.dup2(saved_stderr, 2)
        os.close(saved_stdout)
        os.close(saved_stderr)


def _native_output_tail(path: str | None) -> dict[str, Any] | None:
    if not path:
        return None
    try:
        size = os.path.getsize(path)
        with open(path, "rb") as handle:
            if size > NATIVE_OUTPUT_TAIL_BYTES:
                handle.seek(-NATIVE_OUTPUT_TAIL_BYTES, os.SEEK_END)
            data = handle.read()
        text = data.decode("utf-8", errors="replace")
        return {
            "captured": True,
            "bytes": size,
            "tail_bytes": min(size, NATIVE_OUTPUT_TAIL_BYTES),
            "tail": text,
        }
    except Exception as exc:
        return {"captured": False, "error_type": type(exc).__name__, "error_message": str(exc)}
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


def _load_module(program_path: str):
    module_name = f"autograd_pair_candidate_{uuid.uuid4().hex}"
    spec = importlib.util.spec_from_file_location(module_name, program_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not create module spec for {program_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _validate_api(module) -> tuple[Callable, Callable]:
    forward = getattr(module, FORWARD_FN_NAME, None)
    backward = getattr(module, BACKWARD_FN_NAME, None)
    if forward is None or not callable(forward):
        raise AttributeError(f"candidate must define callable {FORWARD_FN_NAME}")
    if backward is None or not callable(backward):
        raise AttributeError(f"candidate must define callable {BACKWARD_FN_NAME}")
    return forward, backward


def _normalize_saved(saved: Any) -> tuple[torch.Tensor, ...]:
    if isinstance(saved, torch.Tensor):
        saved_tuple = (saved,)
    elif isinstance(saved, (tuple, list)):
        saved_tuple = tuple(saved)
    else:
        raise TypeError("saved_tensors must be a Tensor or a tuple/list of Tensors")
    if not all(isinstance(t, torch.Tensor) for t in saved_tuple):
        raise TypeError("all saved_tensors entries must be torch.Tensor instances")
    return saved_tuple


def _saved_bytes(saved: tuple[torch.Tensor, ...]) -> int:
    return int(sum(t.numel() * t.element_size() for t in saved))


def _input_bytes(*tensors: torch.Tensor) -> int:
    return int(sum(t.numel() * t.element_size() for t in tensors))


def _max_errors(candidate, reference) -> tuple[float, float]:
    diff = (candidate.float() - reference.float()).abs()
    max_abs = float(torch.max(diff).item())
    denom = torch.clamp(reference.float().abs(), min=1e-8)
    max_rel = float(torch.max(diff / denom).item())
    return max_abs, max_rel


def _make_function(forward_fn: Callable, backward_fn: Callable):
    class CandidateLayerNormFunction(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float):
            y, saved = forward_fn(x, weight, bias, eps)
            saved_tensors = _normalize_saved(saved)
            ctx.save_for_backward(*saved_tensors)
            ctx.eps = float(eps)
            return y

        @staticmethod
        def backward(ctx, dy: torch.Tensor):
            saved_tensors = ctx.saved_tensors
            dx, dweight, dbias = backward_fn(dy, saved_tensors, ctx.eps)
            return dx, dweight, dbias, None

    return CandidateLayerNormFunction


def _pair_outputs(forward_fn: Callable, backward_fn: Callable, dy, x, weight, bias, eps):
    cls = _make_function(forward_fn, backward_fn)
    x_req = x.detach().clone().requires_grad_(True)
    weight_req = weight.detach().clone().requires_grad_(True)
    bias_req = bias.detach().clone().requires_grad_(True)
    y = cls.apply(x_req, weight_req, bias_req, eps)
    torch.autograd.backward(y, dy.detach().clone())
    return x_req.grad, weight_req.grad, bias_req.grad


def _case_report(case) -> dict[str, Any]:
    return (
        dict(task_spec.case_metadata(case))
        if hasattr(task_spec, "case_metadata")
        else {"case": repr(case)}
    )


def _run_correctness(forward_fn: Callable, backward_fn: Callable, cases):
    reports = []
    passed = 0
    passed_by_output = {name: 0 for name in task_spec.OUTPUT_NAMES}
    for case in cases:
        report = _case_report(case)
        try:
            torch.manual_seed(task_spec.seed_for_case(case))
            dy, x, weight, bias, eps = task_spec.make_inputs(torch, case)
            expected = task_spec.torch_oracle(torch, dy, x, weight, bias, eps)
            actual = _pair_outputs(forward_fn, backward_fn, dy, x, weight, bias, eps)
            torch.cuda.synchronize()

            with torch.no_grad():
                y, saved = forward_fn(x, weight, bias, eps)
                saved_tensors = _normalize_saved(saved)
            report["forward_shape"] = list(y.shape)
            report["saved_tensors"] = [
                {
                    "shape": list(t.shape),
                    "dtype": str(t.dtype),
                    "bytes": t.numel() * t.element_size(),
                }
                for t in saved_tensors
            ]
            report["saved_bytes"] = _saved_bytes(saved_tensors)

            correct = True
            for name, got, ref in zip(task_spec.OUTPUT_NAMES, actual, expected):
                if got is None:
                    ok = False
                    max_abs = max_rel = float("inf")
                else:
                    max_abs, max_rel = _max_errors(got, ref)
                    ok = bool(
                        torch.allclose(
                            got,
                            ref,
                            atol=task_spec.atol(case, name),
                            rtol=task_spec.rtol(case, name),
                        )
                    )
                report[f"{name}_correct"] = ok
                report[f"{name}_max_abs_error"] = max_abs
                report[f"{name}_max_rel_error"] = max_rel
                correct = correct and ok
                passed_by_output[name] += int(ok)
            report["correct"] = correct
            passed += int(correct)
        except Exception as exc:
            report.update(
                {
                    "correct": False,
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                    "traceback": traceback.format_exc(limit=8),
                }
            )
        reports.append(report)

    total = max(1, len(cases))
    return {
        "passed": passed,
        "total": len(cases),
        "partial_correctness": passed / total,
        **{
            f"{name}_correctness": passed_by_output[name] / total for name in task_spec.OUTPUT_NAMES
        },
        "reports": reports,
    }


def _median_ms(fn: Callable[[], object], warmup: int = BENCHMARK_WARMUP, reps: int = BENCHMARK_REPS) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    times = []
    for _ in range(reps):
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(float(start.elapsed_time(end)))
    return float(statistics.median(times))


def _median_ms_timed_region(
    setup: Callable[[], Any],
    timed: Callable[[Any], object],
    warmup: int = BENCHMARK_WARMUP,
    reps: int = BENCHMARK_REPS,
) -> float:
    for _ in range(warmup):
        state = setup()
        timed(state)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    times = []
    for _ in range(reps):
        state = setup()
        start.record()
        timed(state)
        end.record()
        torch.cuda.synchronize()
        times.append(float(start.elapsed_time(end)))
    return float(statistics.median(times))


def _liger_saved_state(dy, x, weight, bias, eps):
    from liger_kernel.ops.layer_norm import layer_norm_forward  # type: ignore[import-not-found]

    _, x_2d, mean, rstd, _block_size, _num_warps = layer_norm_forward(x, weight, bias, eps)
    dy_2d = dy.view(-1, dy.shape[-1]).contiguous()
    return dy_2d, x_2d, weight, bias, mean, rstd


def _liger_backward_from_saved(state):
    from liger_kernel.ops.layer_norm import layer_norm_backward  # type: ignore[import-not-found]

    dy_2d, x_2d, weight, bias, mean, rstd = state
    return layer_norm_backward(dy_2d, x_2d, weight, bias, mean, rstd)


def _weighted_geomean(speedups: list[float], weights: list[float]) -> float:
    if not speedups:
        return 0.0
    total_weight = sum(weights)
    if total_weight <= 0.0:
        raise ValueError("geomean weights must sum to a positive value")
    return math.exp(sum(w * math.log(max(s, 1e-12)) for s, w in zip(speedups, weights)) / total_weight)


def _geomean_weights_for_cases(cases: list[dict[str, Any]]) -> list[float]:
    if not GEOMEAN_WEIGHTS_ENV:
        return [1.0 for _ in cases]
    weights = [float(part.strip()) for part in GEOMEAN_WEIGHTS_ENV.split(",") if part.strip()]
    if any(weight <= 0.0 for weight in weights):
        raise ValueError("AUTOGRAD_PAIR_GEOMEAN_WEIGHTS must contain positive values")
    if len(weights) == len(cases):
        return weights

    unique_shapes: list[tuple[int, ...]] = []
    for case in cases:
        shape = tuple(int(dim) for dim in case["shape"])
        if shape not in unique_shapes:
            unique_shapes.append(shape)
    if len(weights) == len(unique_shapes):
        shape_to_weight = dict(zip(unique_shapes, weights))
        return [shape_to_weight[tuple(int(dim) for dim in case["shape"])] for case in cases]

    raise ValueError(
        "AUTOGRAD_PAIR_GEOMEAN_WEIGHTS length must match number of benchmark cases "
        f"({len(cases)}) or unique shapes ({len(unique_shapes)}), got {len(weights)}"
    )


def _benchmark_case(forward_fn: Callable, backward_fn: Callable, case) -> dict[str, Any]:
    torch.manual_seed(task_spec.seed_for_case(case))
    dy, x, weight, bias, eps = task_spec.make_inputs(torch, case)

    def forward_only():
        return forward_fn(x, weight, bias, eps)

    def setup_saved():
        _y, saved = forward_fn(x, weight, bias, eps)
        return _normalize_saved(saved)

    def backward_from_saved(saved_tensors):
        return backward_fn(dy, saved_tensors, eps)

    def candidate_raw_full_step():
        _y, saved = forward_fn(x, weight, bias, eps)
        saved_tensors = _normalize_saved(saved)
        return backward_fn(dy, saved_tensors, eps)

    cls = _make_function(forward_fn, backward_fn)

    def candidate_autograd_full_step():
        x_req = x.detach().clone().requires_grad_(True)
        weight_req = weight.detach().clone().requires_grad_(True)
        bias_req = bias.detach().clone().requires_grad_(True)
        y = cls.apply(x_req, weight_req, bias_req, eps)
        torch.autograd.backward(y, dy.detach().clone())
        return x_req.grad, weight_req.grad, bias_req.grad

    def pytorch_full_step():
        x_ref = x.detach().clone().requires_grad_(True)
        weight_ref = weight.detach().clone().requires_grad_(True)
        bias_ref = bias.detach().clone().requires_grad_(True)
        xf = x_ref.float()
        mean = xf.mean(dim=-1, keepdim=True)
        var = ((xf - mean) * (xf - mean)).mean(dim=-1, keepdim=True)
        rstd = torch.rsqrt(var + float(eps))
        y = ((xf - mean) * rstd * weight_ref.float() + bias_ref.float()).to(x.dtype)
        torch.autograd.backward(y, dy.detach().clone())
        return x_ref.grad, weight_ref.grad, bias_ref.grad

    def setup_liger_saved():
        return _liger_saved_state(dy, x, weight, bias, eps)

    def liger_raw_full_step():
        state = _liger_saved_state(dy, x, weight, bias, eps)
        return _liger_backward_from_saved(state)

    with torch.no_grad():
        _y, saved = forward_fn(x, weight, bias, eps)
        saved_tensors = _normalize_saved(saved)

    forward_ms = _median_ms(forward_only)
    backward_ms = _median_ms_timed_region(setup_saved, backward_from_saved)
    raw_full_ms = _median_ms(candidate_raw_full_step)
    autograd_full_ms = _median_ms(candidate_autograd_full_step)
    if PERFORMANCE_BASELINE == "pytorch_autograd":
        baseline_ms = _median_ms(lambda: task_spec.torch_oracle(torch, dy, x, weight, bias, eps))
        baseline_full_ms = _median_ms(pytorch_full_step)
        baseline_autograd_full_ms = baseline_full_ms
    elif PERFORMANCE_BASELINE == "liger":
        baseline_ms = _median_ms_timed_region(setup_liger_saved, _liger_backward_from_saved)
        baseline_full_ms = _median_ms(liger_raw_full_step)
        baseline_autograd_full_ms = None
    else:
        raise ValueError("AUTOGRAD_PAIR_PERF_BASELINE must be 'pytorch_autograd' or 'liger'")
    saved_byte_count = _saved_bytes(saved_tensors)
    input_byte_count = _input_bytes(x, weight, bias)
    return {
        **_case_report(case),
        "forward_ms": forward_ms,
        "backward_from_saved_ms": backward_ms,
        "forward_backward_full_step_ms": raw_full_ms,
        "raw_forward_backward_full_step_ms": raw_full_ms,
        "autograd_forward_backward_full_step_ms": autograd_full_ms,
        "baseline_backward_ms": baseline_ms,
        "baseline_full_step_ms": baseline_full_ms,
        "baseline_raw_full_step_ms": baseline_full_ms,
        "baseline_autograd_full_step_ms": baseline_autograd_full_ms,
        "pytorch_autograd_backward_ms": baseline_ms,
        "pytorch_autograd_full_step_ms": baseline_full_ms,
        "speedup_vs_baseline_backward": baseline_ms / max(backward_ms, 1e-9),
        "speedup_vs_baseline_full_step": baseline_full_ms / max(raw_full_ms, 1e-9),
        "speedup_vs_baseline_raw_full_step": baseline_full_ms / max(raw_full_ms, 1e-9),
        "speedup_vs_baseline_autograd_full_step": (
            baseline_autograd_full_ms / max(autograd_full_ms, 1e-9)
            if baseline_autograd_full_ms is not None
            else None
        ),
        "speedup_vs_pytorch_autograd_backward": baseline_ms / max(backward_ms, 1e-9),
        "speedup_vs_pytorch_autograd_full_step": baseline_full_ms / max(raw_full_ms, 1e-9),
        "saved_bytes": saved_byte_count,
        "input_bytes": input_byte_count,
        "saved_memory_ratio": saved_byte_count / max(input_byte_count, 1),
        "saved_tensors": [
            {"shape": list(t.shape), "dtype": str(t.dtype), "bytes": t.numel() * t.element_size()}
            for t in saved_tensors
        ],
    }


def _run_benchmarks(forward_fn: Callable, backward_fn: Callable):
    cases = []
    totals = {
        "forward_ms": 0.0,
        "backward_from_saved_ms": 0.0,
        "forward_backward_full_step_ms": 0.0,
        "raw_forward_backward_full_step_ms": 0.0,
        "autograd_forward_backward_full_step_ms": 0.0,
        "baseline_backward_ms": 0.0,
        "baseline_full_step_ms": 0.0,
        "baseline_raw_full_step_ms": 0.0,
        "saved_bytes": 0.0,
        "input_bytes": 0.0,
    }
    for case in task_spec.BENCHMARK_CASES:
        report = _benchmark_case(forward_fn, backward_fn, case)
        cases.append(report)
        for key in totals:
            totals[key] += float(report[key])
    totals["speedup_vs_baseline_backward"] = totals["baseline_backward_ms"] / max(
        totals["backward_from_saved_ms"], 1e-9
    )
    totals["speedup_vs_baseline_full_step"] = totals["baseline_full_step_ms"] / max(
        totals["forward_backward_full_step_ms"], 1e-9
    )
    totals["speedup_vs_baseline_raw_full_step"] = totals["baseline_raw_full_step_ms"] / max(
        totals["raw_forward_backward_full_step_ms"], 1e-9
    )
    totals["pytorch_autograd_backward_ms"] = totals["baseline_backward_ms"]
    totals["pytorch_autograd_full_step_ms"] = totals["baseline_full_step_ms"]
    totals["speedup_vs_pytorch_autograd_backward"] = totals["speedup_vs_baseline_backward"]
    totals["speedup_vs_pytorch_autograd_full_step"] = totals["speedup_vs_baseline_full_step"]
    backward_speedups = [
        float(case["speedup_vs_baseline_backward"])
        for case in cases
        if float(case["speedup_vs_baseline_backward"]) > 0.0
    ]
    full_step_speedups = [
        float(case["speedup_vs_baseline_raw_full_step"])
        for case in cases
        if float(case["speedup_vs_baseline_raw_full_step"]) > 0.0
    ]
    min_speedups = [
        min(float(case["speedup_vs_baseline_backward"]), float(case["speedup_vs_baseline_raw_full_step"]))
        for case in cases
        if float(case["speedup_vs_baseline_backward"]) > 0.0
        and float(case["speedup_vs_baseline_raw_full_step"]) > 0.0
    ]
    geomean_weights = _geomean_weights_for_cases(cases)
    totals["geomean_speedup_vs_baseline_backward"] = math.exp(
        sum(math.log(speedup) for speedup in backward_speedups) / max(len(backward_speedups), 1)
    )
    totals["geomean_speedup_vs_baseline_full_step"] = math.exp(
        sum(math.log(speedup) for speedup in full_step_speedups) / max(len(full_step_speedups), 1)
    )
    totals["geomean_min_speedup_per_case"] = math.exp(
        sum(math.log(speedup) for speedup in min_speedups) / max(len(min_speedups), 1)
    )
    totals["weighted_geomean_speedup_vs_baseline_backward"] = _weighted_geomean(backward_speedups, geomean_weights)
    totals["weighted_geomean_speedup_vs_baseline_full_step"] = _weighted_geomean(full_step_speedups, geomean_weights)
    totals["weighted_geomean_min_speedup_per_case"] = _weighted_geomean(min_speedups, geomean_weights)
    totals["worst_case_speedup_vs_baseline_backward"] = min(backward_speedups) if backward_speedups else 0.0
    totals["worst_case_speedup_vs_baseline_full_step"] = min(full_step_speedups) if full_step_speedups else 0.0
    totals["worst_case_min_speedup"] = min(min_speedups) if min_speedups else 0.0
    totals["saved_memory_ratio"] = totals["saved_bytes"] / max(totals["input_bytes"], 1e-9)
    return {"aggregate": totals, "cases": cases, "geomean_weights": geomean_weights}


def _score_from_aggregate(aggregate: dict[str, float]) -> tuple[float, dict[str, float]]:
    backward_speedup = float(aggregate["speedup_vs_baseline_backward"])
    full_step_speedup = float(aggregate["speedup_vs_baseline_full_step"])
    saved_memory_ratio = float(aggregate["saved_memory_ratio"])
    weighted_speedup = (1.0 - FULL_STEP_WEIGHT) * backward_speedup + FULL_STEP_WEIGHT * full_step_speedup
    min_speedup = min(backward_speedup, full_step_speedup)
    geomean_backward_speedup = float(aggregate.get("geomean_speedup_vs_baseline_backward", backward_speedup))
    geomean_full_step_speedup = float(aggregate.get("geomean_speedup_vs_baseline_full_step", full_step_speedup))
    geomean_min_speedup = min(geomean_backward_speedup, geomean_full_step_speedup)
    weighted_geomean_backward_speedup = float(
        aggregate.get("weighted_geomean_speedup_vs_baseline_backward", geomean_backward_speedup)
    )
    weighted_geomean_full_step_speedup = float(
        aggregate.get("weighted_geomean_speedup_vs_baseline_full_step", geomean_full_step_speedup)
    )
    weighted_geomean_min_speedup = float(
        aggregate.get(
            "weighted_geomean_min_speedup_per_case",
            min(weighted_geomean_backward_speedup, weighted_geomean_full_step_speedup),
        )
    )
    worst_case_min_speedup = float(aggregate.get("worst_case_min_speedup", min_speedup))
    worst_case_guard_factor = min(1.0, worst_case_min_speedup) if WORST_CASE_GUARD else 1.0
    memory_penalty_factor = 1.0 + MEMORY_PENALTY_WEIGHT * saved_memory_ratio

    if SCORE_MODE == "speed_memory":
        score = weighted_speedup / memory_penalty_factor
    elif SCORE_MODE == "speed_memory_min":
        score = min_speedup / memory_penalty_factor
    elif SCORE_MODE == "speed_memory_min_geomean":
        score = geomean_min_speedup / memory_penalty_factor
    elif SCORE_MODE == "speed_memory_min_weighted_geomean":
        score = (weighted_geomean_min_speedup * worst_case_guard_factor) / memory_penalty_factor
    else:
        score = backward_speedup

    return score, {
        "backward_speedup": backward_speedup,
        "full_step_speedup": full_step_speedup,
        "weighted_speedup": weighted_speedup,
        "min_speedup": min_speedup,
        "geomean_backward_speedup": geomean_backward_speedup,
        "geomean_full_step_speedup": geomean_full_step_speedup,
        "geomean_min_speedup": geomean_min_speedup,
        "weighted_geomean_backward_speedup": weighted_geomean_backward_speedup,
        "weighted_geomean_full_step_speedup": weighted_geomean_full_step_speedup,
        "weighted_geomean_min_speedup": weighted_geomean_min_speedup,
        "worst_case_min_speedup": worst_case_min_speedup,
        "worst_case_guard_factor": worst_case_guard_factor,
        "saved_memory_ratio": saved_memory_ratio,
        "memory_penalty_factor": memory_penalty_factor,
        "score_mode_speed_memory": 1.0 if SCORE_MODE == "speed_memory" else 0.0,
        "score_mode_speed_memory_min": 1.0 if SCORE_MODE == "speed_memory_min" else 0.0,
        "score_mode_speed_memory_min_geomean": 1.0 if SCORE_MODE == "speed_memory_min_geomean" else 0.0,
        "score_mode_speed_memory_min_weighted_geomean": 1.0
        if SCORE_MODE == "speed_memory_min_weighted_geomean"
        else 0.0,
    }


def evaluate(program_path: str) -> EvaluationResult:
    if not torch.cuda.is_available():
        return _result(
            {"combined_score": -1e9, "correct": 0.0},
            {
                "failure": {
                    "error_type": "RuntimeUnavailable",
                    "error_message": "CUDA is not available",
                }
            },
        )
    try:
        module = _load_module(program_path)
        forward_fn, backward_fn = _validate_api(module)
    except Exception as exc:
        return _result(
            {"combined_score": -1e9, "correct": 0.0},
            {
                "failure": {
                    "error_type": "ImportOrApiError",
                    "error_message": str(exc),
                    "traceback": traceback.format_exc(limit=8),
                }
            },
        )

    correctness = None
    native_output_path = None
    try:
        with _capture_native_output() as captured_path:
            native_output_path = captured_path
            correctness = _run_correctness(forward_fn, backward_fn, task_spec.CORRECTNESS_CASES)
            if correctness["passed"] == correctness["total"]:
                benchmark = _run_benchmarks(forward_fn, backward_fn)
    except Exception as exc:
        native_output = _native_output_tail(native_output_path)
        artifacts: dict[str, Any] = {
            "failure": {
                "error_type": "CandidateExecutionError",
                "error_message": str(exc),
                "traceback": traceback.format_exc(limit=8),
            }
        }
        if correctness is not None:
            artifacts["correctness"] = correctness
        if native_output and native_output.get("bytes", 0) > 0:
            artifacts["native_output"] = native_output
        return _result({"combined_score": -1e9, "correct": 0.0}, artifacts)

    native_output = _native_output_tail(native_output_path)
    if correctness["passed"] != correctness["total"]:
        metrics = {
            "combined_score": -1e6 + float(correctness["partial_correctness"]),
            "correct": 0.0,
            "partial_correctness": float(correctness["partial_correctness"]),
        }
        metrics.update({f"{name}_correct": float(correctness[f"{name}_correctness"]) for name in task_spec.OUTPUT_NAMES})
        artifacts = {"correctness": correctness}
        if native_output and native_output.get("bytes", 0) > 0:
            artifacts["native_output"] = native_output
        return _result(metrics, artifacts)

    aggregate = benchmark["aggregate"]
    combined_score, score_details = _score_from_aggregate(aggregate)
    metrics = {
        "combined_score": float(combined_score),
        "correct": 1.0,
        "partial_correctness": 1.0,
        "speedup": float(aggregate["speedup_vs_pytorch_autograd_backward"]),
        "full_step_speedup": float(aggregate["speedup_vs_pytorch_autograd_full_step"]),
        "weighted_speedup": float(score_details["weighted_speedup"]),
        "min_speedup": float(score_details["min_speedup"]),
        "geomean_backward_speedup": float(score_details["geomean_backward_speedup"]),
        "geomean_full_step_speedup": float(score_details["geomean_full_step_speedup"]),
        "geomean_min_speedup": float(score_details["geomean_min_speedup"]),
        "weighted_geomean_backward_speedup": float(score_details["weighted_geomean_backward_speedup"]),
        "weighted_geomean_full_step_speedup": float(score_details["weighted_geomean_full_step_speedup"]),
        "weighted_geomean_min_speedup": float(score_details["weighted_geomean_min_speedup"]),
        "worst_case_min_speedup": float(score_details["worst_case_min_speedup"]),
        "worst_case_guard_factor": float(score_details["worst_case_guard_factor"]),
        "forward_ms": float(aggregate["forward_ms"]),
        "backward_from_saved_ms": float(aggregate["backward_from_saved_ms"]),
        "forward_backward_full_step_ms": float(aggregate["forward_backward_full_step_ms"]),
        "raw_forward_backward_full_step_ms": float(aggregate["raw_forward_backward_full_step_ms"]),
        "autograd_forward_backward_full_step_ms": float(aggregate["autograd_forward_backward_full_step_ms"]),
        "baseline_latency_ms": float(aggregate["pytorch_autograd_backward_ms"]),
        "baseline_full_step_ms": float(aggregate["pytorch_autograd_full_step_ms"]),
        "baseline_raw_full_step_ms": float(aggregate["baseline_raw_full_step_ms"]),
        "saved_bytes": float(aggregate["saved_bytes"]),
        "input_bytes": float(aggregate["input_bytes"]),
        "saved_memory_ratio": float(score_details["saved_memory_ratio"]),
        "memory_penalty_factor": float(score_details["memory_penalty_factor"]),
        "score_mode_speed_memory": float(score_details["score_mode_speed_memory"]),
        "score_mode_speed_memory_min": float(score_details["score_mode_speed_memory_min"]),
        "score_mode_speed_memory_min_geomean": float(score_details["score_mode_speed_memory_min_geomean"]),
        "score_mode_speed_memory_min_weighted_geomean": float(
            score_details["score_mode_speed_memory_min_weighted_geomean"]
        ),
    }
    metrics.update({f"{name}_correct": 1.0 for name in task_spec.OUTPUT_NAMES})
    benchmark["score_mode"] = SCORE_MODE
    benchmark["performance_baseline"] = PERFORMANCE_BASELINE
    benchmark["full_step_weight"] = FULL_STEP_WEIGHT
    benchmark["memory_penalty_weight"] = MEMORY_PENALTY_WEIGHT
    benchmark["geomean_weights"] = benchmark.get("geomean_weights", [])
    benchmark["worst_case_guard"] = WORST_CASE_GUARD
    benchmark["warmup"] = BENCHMARK_WARMUP
    benchmark["reps"] = BENCHMARK_REPS
    if hasattr(task_spec, "benchmark_suite_name"):
        benchmark["suite"] = task_spec.benchmark_suite_name()
    if hasattr(task_spec, "benchmark_shapes"):
        benchmark["shapes"] = [list(shape) for shape in task_spec.benchmark_shapes()]
    if hasattr(task_spec, "benchmark_dtype_names"):
        benchmark["dtypes"] = task_spec.benchmark_dtype_names()
    return _result(metrics, {"correctness": correctness, "benchmark": benchmark})


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(f"Usage: {argv[0]} PROGRAM_PATH")
        return 2
    result = evaluate(argv[1])
    print(json.dumps({"metrics": result.metrics, "artifacts": result.artifacts}, indent=2))
    return 0 if result.metrics.get("correct", 0.0) == 1.0 else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
