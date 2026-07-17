"""Shared evaluator core for autograd-pair Triton backward benchmarks.

Each task supplies a ``task_spec.py`` with the regular backward benchmark
contract plus a small amount of autograd-pair metadata.  This core evaluates:

    <op>_forward_with_saved(*forward_inputs) -> (y, saved_tensors)
    <op>_backward_from_saved(dout, saved_tensors, *extra_args) -> gradients

The score follows the current min-speedup objective:

    min(backward_speedup, full_step_speedup) / (1 + memory_weight * saved_ratio)
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
import importlib.util
import json
import math
import os
import statistics
import subprocess
import sys
import tempfile
import traceback
import uuid
from typing import Any, Callable, Sequence

try:
    from openevolve.evaluation_result import EvaluationResult
except Exception:  # pragma: no cover

    @dataclass
    class EvaluationResult:
        metrics: dict[str, float]
        artifacts: dict[str, str | bytes] = field(default_factory=dict)


COMPILE_ERROR_SCORE = -1e9
CORRECTNESS_ERROR_SCORE = -1e6
SCORE_MODE = os.environ.get("AUTOGRAD_PAIR_SCORE_MODE", "speed_only")
PERFORMANCE_BASELINE = os.environ.get("AUTOGRAD_PAIR_PERF_BASELINE", "pytorch_autograd").strip().lower()
# Which benchmark sub-suite to score on: full (task_spec.BENCHMARK_CASES),
# small (task_spec.SMALL_CASES), or large (task_spec.LARGE_CASES). Correctness always
# runs on the full CORRECTNESS_CASES regardless of this setting.
SUITE = os.environ.get("AUTOGRAD_PAIR_SUITE", "full").strip().lower()
MEMORY_PENALTY_WEIGHT = float(os.environ.get("AUTOGRAD_PAIR_MEMORY_PENALTY_WEIGHT", "0.05"))
CAPTURE_NATIVE_OUTPUT = os.environ.get("AUTOGRAD_PAIR_CAPTURE_NATIVE_OUTPUT", "1").lower() not in (
    "0",
    "false",
    "no",
)
NATIVE_OUTPUT_TAIL_BYTES = int(os.environ.get("AUTOGRAD_PAIR_NATIVE_OUTPUT_TAIL_BYTES", "65536"))
BENCHMARK_WARMUP = int(os.environ.get("AUTOGRAD_PAIR_BENCHMARK_WARMUP", "10"))
BENCHMARK_REPS = int(os.environ.get("AUTOGRAD_PAIR_BENCHMARK_REPS", "50"))
BASELINE_TIMING_CACHE = os.environ.get("AUTOGRAD_PAIR_BASELINE_TIMING_CACHE", "1").lower() not in (
    "0",
    "false",
    "no",
)

# Pristine copies of the process's original stdout/stderr fds, taken at import. If an evaluation
# is abandoned mid-_capture_native_output (openevolve's timeout drops the worker thread on the
# floor), fds 1/2 stay dup2'd to its temp file and the console goes dark for the rest of the
# process. Every evaluate() entry restores them (self-healing) instead.
try:
    _PRISTINE_STDOUT_FD = os.dup(1)
    _PRISTINE_STDERR_FD = os.dup(2)
except OSError:  # pragma: no cover — no usable std fds (daemonized caller)
    _PRISTINE_STDOUT_FD = _PRISTINE_STDERR_FD = -1


def _restore_std_fds() -> None:
    if _PRISTINE_STDOUT_FD < 0:
        return
    try:
        sys.stdout.flush()
        sys.stderr.flush()
    except Exception:
        pass
    os.dup2(_PRISTINE_STDOUT_FD, 1)
    os.dup2(_PRISTINE_STDERR_FD, 2)


def _json(data: dict[str, Any]) -> str:
    return json.dumps(data, indent=2, sort_keys=True)


def _result(metrics: dict[str, float], artifacts: dict[str, Any]) -> EvaluationResult:
    return EvaluationResult(metrics=metrics, artifacts={key: _json(value) for key, value in artifacts.items()})


def _output_metric_names(output_names: Sequence[str]) -> dict[str, float]:
    return {f"{name}_correct": 0.0 for name in output_names}


def _load_module(program_path: str, prefix: str):
    module_name = f"{prefix}_{uuid.uuid4().hex}"
    spec = importlib.util.spec_from_file_location(module_name, program_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not create module spec for {program_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _failure(
    task_spec,
    score: float,
    error_type: str,
    error_message: str,
    details: dict[str, Any] | None = None,
) -> EvaluationResult:
    metrics = {
        "combined_score": float(score),
        "correct": 0.0,
        "partial_correctness": 0.0,
        "speedup": 0.0,
        "full_step_speedup": 0.0,
        "backward_from_saved_ms": 0.0,
        "forward_backward_full_step_ms": 0.0,
        "baseline_latency_ms": 0.0,
        "baseline_full_step_ms": 0.0,
        "saved_memory_ratio": 0.0,
    }
    metrics.update(_output_metric_names(task_spec.OUTPUT_NAMES))
    return _result(
        metrics,
        {
            "failure": {
                "error_type": error_type,
                "error_message": error_message,
                "details": details or {},
            }
        },
    )


def check_runtime():
    try:
        import torch
        import triton  # noqa: F401
    except Exception as exc:
        return None, f"Failed to import torch/triton: {exc}"

    if not torch.cuda.is_available():
        return torch, "CUDA is not available; run this evaluator on a GPU node"

    return torch, None


def validate_api(module, function_name: str) -> Callable:
    fn = getattr(module, function_name, None)
    if fn is None or not callable(fn):
        raise AttributeError(f"candidate must define callable {function_name}")
    return fn


def _normalize_outputs(output: Any, output_names: Sequence[str]) -> tuple[Any, ...]:
    if len(output_names) == 1:
        return (output,)
    if not isinstance(output, tuple) or len(output) != len(output_names):
        expected = ", ".join(output_names)
        raise TypeError(f"candidate must return a tuple ({expected})")
    return output


def _normalize_saved(saved: Any) -> tuple[Any, ...]:
    try:
        import torch
    except Exception:  # pragma: no cover
        torch = None
    if torch is not None and isinstance(saved, torch.Tensor):
        saved_tuple = (saved,)
    elif isinstance(saved, (tuple, list)):
        saved_tuple = tuple(saved)
    else:
        raise TypeError("saved_tensors must be a Tensor or a tuple/list of Tensors")
    if torch is not None and not all(isinstance(t, torch.Tensor) for t in saved_tuple):
        raise TypeError("all saved_tensors entries must be torch.Tensor instances")
    return saved_tuple


def _tensor_bytes(tensor: Any) -> int:
    return int(tensor.numel() * tensor.element_size())


def _saved_bytes(saved_tensors: Sequence[Any]) -> int:
    return int(sum(_tensor_bytes(tensor) for tensor in saved_tensors))


def _input_bytes(inputs: Sequence[Any]) -> int:
    return int(sum(_tensor_bytes(tensor) for tensor in inputs if hasattr(tensor, "numel") and hasattr(tensor, "element_size")))


def _max_errors(torch_module, candidate, reference) -> tuple[float, float]:
    diff = (candidate.float() - reference.float()).abs()
    max_abs = float(torch_module.max(diff).item())
    denom = torch_module.clamp(reference.float().abs(), min=1e-8)
    max_rel = float(torch_module.max(diff / denom).item())
    return max_abs, max_rel


def _case_report(task_spec, case) -> dict[str, Any]:
    if hasattr(task_spec, "case_metadata"):
        return dict(task_spec.case_metadata(case))
    return {"case": repr(case)}


def _pair_inputs(task_spec, torch_module, case) -> tuple[Any, ...]:
    if hasattr(task_spec, "make_autograd_pair_inputs"):
        return tuple(task_spec.make_autograd_pair_inputs(torch_module, case))
    return tuple(task_spec.make_inputs(torch_module, case))


def _forward_indices(task_spec) -> tuple[int, ...]:
    return tuple(task_spec.AUTOGRAD_PAIR_FORWARD_INPUT_INDICES)


def _cotangent_index(task_spec) -> int:
    return int(getattr(task_spec, "AUTOGRAD_PAIR_COTANGENT_INDEX", 0))


def _backward_extra_indices(task_spec) -> tuple[int, ...]:
    return tuple(getattr(task_spec, "AUTOGRAD_PAIR_BACKWARD_EXTRA_INPUT_INDICES", ()))


def _memory_input_indices(task_spec) -> tuple[int, ...]:
    return tuple(getattr(task_spec, "AUTOGRAD_PAIR_MEMORY_INPUT_INDICES", _forward_indices(task_spec)))


def _forward_args(task_spec, inputs: Sequence[Any]) -> tuple[Any, ...]:
    return tuple(inputs[index] for index in _forward_indices(task_spec))


def _backward_extra_args(task_spec, inputs: Sequence[Any]) -> tuple[Any, ...]:
    return tuple(inputs[index] for index in _backward_extra_indices(task_spec))


def _memory_inputs(task_spec, inputs: Sequence[Any]) -> tuple[Any, ...]:
    return tuple(inputs[index] for index in _memory_input_indices(task_spec))


def _forward_oracle(task_spec, torch_module, inputs: Sequence[Any]):
    forward_args = _forward_args(task_spec, inputs)
    if hasattr(task_spec, "autograd_pair_forward_oracle"):
        return task_spec.autograd_pair_forward_oracle(torch_module, *forward_args)
    if hasattr(task_spec, "forward_oracle"):
        return task_spec.forward_oracle(torch_module, *forward_args)
    raise AttributeError("task_spec must define autograd_pair_forward_oracle or forward_oracle")


def _gradient_oracle(task_spec, torch_module, inputs: Sequence[Any]) -> tuple[Any, ...]:
    if hasattr(task_spec, "autograd_pair_torch_oracle"):
        expected = task_spec.autograd_pair_torch_oracle(torch_module, *inputs)
    else:
        expected = task_spec.torch_oracle(torch_module, *inputs)
    return _normalize_outputs(expected, task_spec.OUTPUT_NAMES)


def _liger_baseline_fns(task_spec) -> tuple[Callable, Callable]:
    make_fns = getattr(task_spec, "make_liger_autograd_pair_fns", None)
    if make_fns is None:
        raise AttributeError(
            "AUTOGRAD_PAIR_PERF_BASELINE=liger requires task_spec.make_liger_autograd_pair_fns() "
            "returning (forward_with_saved, backward_from_saved) built from the Liger kernel"
        )
    return make_fns()


def _smoke_benchmark_shapes(torch_module, task_spec, forward_fn: Callable, backward_fn: Callable) -> dict[str, Any] | None:
    """One untimed fw+bwd pass per benchmark case; None if all pass, else a failure record.

    Catches shape-dependent crashes — and, together with the CALLER's subprocess timeout,
    shape-dependent deadlocks — that the small correctness shapes cannot see.

    Progress goes to STDERR line-by-line (stdout must stay clean for the report JSON): if the
    caller's timeout kills this process, the last [smoke] line in its partial stderr names the
    hanging case, and the repair prompt can hand the LLM that shape."""
    cases = _selected_benchmark_cases(task_spec)
    print(f"[smoke] correctness passed; smoke-running {len(cases)} benchmark shapes once each",
          file=sys.stderr, flush=True)
    for case in cases:
        print(f"[smoke] {_case_report(task_spec, case)}", file=sys.stderr, flush=True)
        try:
            if hasattr(task_spec, "seed_for_case"):
                torch_module.manual_seed(task_spec.seed_for_case(case))
            inputs = _pair_inputs(task_spec, torch_module, case)
            dout = inputs[_cotangent_index(task_spec)]
            forward_args = _forward_args(task_spec, inputs)
            backward_extra_args = _backward_extra_args(task_spec, inputs)
            with torch_module.no_grad():
                _y, saved = forward_fn(*forward_args)
                backward_fn(dout, _normalize_saved(saved), *backward_extra_args)
            torch_module.cuda.synchronize()
            del inputs, dout, saved
        except Exception as exc:
            return {
                "error_type": "BenchmarkShapeSmokeError",
                "error_message": f"{type(exc).__name__}: {exc}",
                "case": _case_report(task_spec, case),
                "traceback": traceback.format_exc(limit=8),
            }
    return None


def _selected_benchmark_cases(task_spec) -> list[Any]:
    if SUITE == "full":
        return list(task_spec.BENCHMARK_CASES)
    if SUITE == "small":
        cases = getattr(task_spec, "SMALL_CASES", None)
    elif SUITE == "large":
        cases = getattr(task_spec, "LARGE_CASES", None)
    else:
        raise ValueError(
            f"Unsupported AUTOGRAD_PAIR_SUITE={SUITE!r}; expected 'small', 'large', or 'full'"
        )
    if cases is None:
        raise AttributeError(
            f"task_spec must define {SUITE.upper()}_CASES to use AUTOGRAD_PAIR_SUITE={SUITE}"
        )
    return list(cases)


def _case_weight(task_spec, case) -> float:
    fn = getattr(task_spec, "case_weight", None)
    if fn is None:
        return 1.0
    return float(fn(case))


def _forward_atol(task_spec, case) -> float:
    if hasattr(task_spec, "autograd_pair_forward_atol"):
        return float(task_spec.autograd_pair_forward_atol(case))
    return max(float(task_spec.atol(case, name)) for name in task_spec.OUTPUT_NAMES)


def _forward_rtol(task_spec, case) -> float:
    if hasattr(task_spec, "autograd_pair_forward_rtol"):
        return float(task_spec.autograd_pair_forward_rtol(case))
    return max(float(task_spec.rtol(case, name)) for name in task_spec.OUTPUT_NAMES)


@contextmanager
def _capture_native_output():
    if not CAPTURE_NATIVE_OUTPUT:
        yield None
        return

    sys.stdout.flush()
    sys.stderr.flush()
    fd, path = tempfile.mkstemp(prefix="autograd_pair_native_", suffix=".log")
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
        return {
            "captured": True,
            "bytes": size,
            "tail_bytes": min(size, NATIVE_OUTPUT_TAIL_BYTES),
            "tail": data.decode("utf-8", errors="replace"),
        }
    except Exception as exc:
        return {"captured": False, "error_type": type(exc).__name__, "error_message": str(exc)}
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


def _run_correctness(
    torch_module,
    task_spec,
    forward_fn: Callable,
    backward_fn: Callable,
    cases: Sequence[Any],
) -> dict[str, Any]:
    passed_cases = 0
    passed_by_output = {name: 0 for name in task_spec.OUTPUT_NAMES}
    reports = []

    for case in cases:
        case_data = _case_report(task_spec, case)
        try:
            if hasattr(task_spec, "seed_for_case"):
                torch_module.manual_seed(task_spec.seed_for_case(case))
            inputs = _pair_inputs(task_spec, torch_module, case)
            dout = inputs[_cotangent_index(task_spec)]
            forward_args = _forward_args(task_spec, inputs)
            backward_extra_args = _backward_extra_args(task_spec, inputs)
            expected_y = _forward_oracle(task_spec, torch_module, inputs)
            expected_grads = _gradient_oracle(task_spec, torch_module, inputs)

            actual_y, saved = forward_fn(*forward_args)
            saved_tensors = _normalize_saved(saved)
            actual_grads = _normalize_outputs(
                backward_fn(dout, saved_tensors, *backward_extra_args),
                task_spec.OUTPUT_NAMES,
            )
            torch_module.cuda.synchronize()

            report = {
                **case_data,
                "correct": True,
                "forward_shape": list(actual_y.shape),
                "saved_tensors": [
                    {"shape": list(t.shape), "dtype": str(t.dtype), "bytes": _tensor_bytes(t)}
                    for t in saved_tensors
                ],
                "saved_bytes": _saved_bytes(saved_tensors),
            }

            if actual_y.shape != expected_y.shape:
                report.update(
                    {
                        "correct": False,
                        "forward_correct": False,
                        "error_type": "ForwardShapeMismatch",
                        "error_message": f"forward shape {tuple(actual_y.shape)} != {tuple(expected_y.shape)}",
                    }
                )
                reports.append(report)
                continue

            # Compare in fp32 like _max_errors does: a candidate may legitimately return
            # higher-precision tensors (fp32 accumulation on a half-precision case), and
            # torch.allclose raises on mixed dtypes instead of casting.
            forward_max_abs, forward_max_rel = _max_errors(torch_module, actual_y, expected_y)
            forward_ok = bool(
                torch_module.allclose(
                    actual_y.float(),
                    expected_y.float(),
                    atol=_forward_atol(task_spec, case),
                    rtol=_forward_rtol(task_spec, case),
                )
            )
            report["forward_correct"] = forward_ok
            report["forward_max_abs_error"] = forward_max_abs
            report["forward_max_rel_error"] = forward_max_rel
            report["correct"] = report["correct"] and forward_ok

            shape_errors = []
            for name, actual_tensor, expected_tensor in zip(task_spec.OUTPUT_NAMES, actual_grads, expected_grads):
                if actual_tensor.shape != expected_tensor.shape:
                    shape_errors.append(
                        f"{name} shape {tuple(actual_tensor.shape)} != {tuple(expected_tensor.shape)}"
                    )
            if shape_errors:
                report.update(
                    {
                        "correct": False,
                        **{f"{name}_correct": False for name in task_spec.OUTPUT_NAMES},
                        "error_type": "ShapeMismatch",
                        "error_message": "; ".join(shape_errors),
                    }
                )
                reports.append(report)
                continue

            for name, actual_tensor, expected_tensor in zip(task_spec.OUTPUT_NAMES, actual_grads, expected_grads):
                atol = task_spec.atol(case, name)
                rtol = task_spec.rtol(case, name)
                max_abs, max_rel = _max_errors(torch_module, actual_tensor, expected_tensor)
                is_correct = bool(torch_module.allclose(
                    actual_tensor.float(), expected_tensor.float(), atol=atol, rtol=rtol))
                passed_by_output[name] += int(is_correct)
                report[f"{name}_correct"] = is_correct
                report[f"{name}_max_abs_error"] = max_abs
                report[f"{name}_max_rel_error"] = max_rel
                report[f"{name}_atol"] = atol
                report[f"{name}_rtol"] = rtol
                report["correct"] = report["correct"] and is_correct

            if hasattr(task_spec, "correctness_hint"):
                report["hint"] = task_spec.correctness_hint()

            passed_cases += int(report["correct"])
            reports.append(report)
        except Exception as exc:
            reports.append(
                {
                    **case_data,
                    "correct": False,
                    "forward_correct": False,
                    **{f"{name}_correct": False for name in task_spec.OUTPUT_NAMES},
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                    "traceback": traceback.format_exc(limit=8),
                }
            )

    total = max(1, len(cases))
    return {
        "passed": passed_cases,
        "total": len(cases),
        "partial_correctness": passed_cases / total,
        **{f"{name}_correctness": passed_by_output[name] / total for name in task_spec.OUTPUT_NAMES},
        "reports": reports,
    }


def _median_ms(fn: Callable[[], object], warmup: int = BENCHMARK_WARMUP, reps: int = BENCHMARK_REPS) -> float:
    import torch

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
    import torch

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


# --------------------------------------------------------------------------- baseline cache
# The baseline's latency per case is a constant for the whole evolve run (same shapes, same
# baseline, same GPU) — but naively it gets re-timed inside EVERY candidate evaluation, which
# under the slow autograd baseline dominates evaluation time. Cache it on disk next to the
# task_spec, keyed by everything the number depends on. Evaluations run in separate worker
# processes, so the cache must be a file, not a dict; writes are atomic (tmp + rename) and
# merge the freshest file contents so concurrent workers don't clobber each other's entries.

_BASELINE_CACHE_MEMO: dict[str, dict[str, Any]] = {}


def _baseline_cache_path(task_spec) -> str | None:
    spec_file = getattr(task_spec, "__file__", None)
    if not BASELINE_TIMING_CACHE or not spec_file:
        return None
    return os.path.join(os.path.dirname(os.path.abspath(spec_file)), ".baseline_timing_cache.json")


def _baseline_cache_key(torch_module, task_spec, case) -> str:
    gpu = torch_module.cuda.get_device_name(0) if torch_module.cuda.is_available() else "cpu"
    return json.dumps(
        {
            "baseline": PERFORMANCE_BASELINE,
            "gpu": gpu,
            "warmup": BENCHMARK_WARMUP,
            "reps": BENCHMARK_REPS,
            "case": _case_report(task_spec, case),
        },
        sort_keys=True,
        default=repr,
    )


def _baseline_cache_get(torch_module, task_spec, case) -> tuple[float, float] | None:
    path = _baseline_cache_path(task_spec)
    if path is None:
        return None
    if path not in _BASELINE_CACHE_MEMO:
        try:
            with open(path, encoding="utf-8") as fh:
                _BASELINE_CACHE_MEMO[path] = json.load(fh)
        except Exception:
            _BASELINE_CACHE_MEMO[path] = {}
    hit = _BASELINE_CACHE_MEMO[path].get(_baseline_cache_key(torch_module, task_spec, case))
    return (float(hit[0]), float(hit[1])) if hit else None


def _baseline_cache_put(torch_module, task_spec, case, baseline_ms: float, baseline_full_ms: float) -> None:
    path = _baseline_cache_path(task_spec)
    if path is None:
        return
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
    except Exception:
        data = {}
    data[_baseline_cache_key(torch_module, task_spec, case)] = [baseline_ms, baseline_full_ms]
    tmp = f"{path}.{os.getpid()}.tmp"
    try:
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=1, sort_keys=True)
        os.replace(tmp, path)
    except OSError:
        return  # read-only bench dir etc. — the cache is an optimization, never a requirement
    _BASELINE_CACHE_MEMO[path] = data


def _benchmark_case(torch_module, task_spec, forward_fn: Callable, backward_fn: Callable, case) -> dict[str, Any]:
    if hasattr(task_spec, "seed_for_case"):
        torch_module.manual_seed(task_spec.seed_for_case(case))
    inputs = _pair_inputs(task_spec, torch_module, case)
    dout = inputs[_cotangent_index(task_spec)]
    forward_args = _forward_args(task_spec, inputs)
    backward_extra_args = _backward_extra_args(task_spec, inputs)

    def forward_only():
        return forward_fn(*forward_args)

    def setup_saved():
        _y, saved = forward_fn(*forward_args)
        return _normalize_saved(saved)

    def backward_from_saved(saved_tensors):
        return backward_fn(dout, saved_tensors, *backward_extra_args)

    def candidate_full_step():
        _y, saved = forward_fn(*forward_args)
        saved_tensors = _normalize_saved(saved)
        return backward_fn(dout, saved_tensors, *backward_extra_args)

    def baseline_full_step():
        return _gradient_oracle(task_spec, torch_module, inputs)

    with torch_module.no_grad():
        _y, saved = forward_fn(*forward_args)
        saved_tensors = _normalize_saved(saved)

    forward_ms = _median_ms(forward_only)
    backward_ms = _median_ms_timed_region(setup_saved, backward_from_saved)
    full_step_ms = _median_ms(candidate_full_step)

    if PERFORMANCE_BASELINE not in ("liger", "pytorch_autograd"):
        raise ValueError(
            f"Unsupported AUTOGRAD_PAIR_PERF_BASELINE={PERFORMANCE_BASELINE!r}; "
            "expected 'pytorch_autograd' or 'liger'"
        )
    cached_baseline = _baseline_cache_get(torch_module, task_spec, case)
    if cached_baseline is not None:
        baseline_ms, baseline_full_ms = cached_baseline
    elif PERFORMANCE_BASELINE == "liger":
        liger_fwd, liger_bwd = _liger_baseline_fns(task_spec)

        def liger_setup_saved():
            _, liger_saved = liger_fwd(*forward_args)
            return _normalize_saved(liger_saved)

        def liger_backward_from_saved(liger_saved_tensors):
            return liger_bwd(dout, liger_saved_tensors, *backward_extra_args)

        def liger_full_step():
            _, liger_saved = liger_fwd(*forward_args)
            liger_saved_tensors = _normalize_saved(liger_saved)
            return liger_bwd(dout, liger_saved_tensors, *backward_extra_args)

        # Backward-only excludes forward from the timed region (same convention as
        # candidate's backward_ms above), full-step times forward+backward together —
        # so both baseline numbers are directly comparable to the candidate's.
        baseline_ms = _median_ms_timed_region(liger_setup_saved, liger_backward_from_saved)
        baseline_full_ms = _median_ms(liger_full_step)
        _baseline_cache_put(torch_module, task_spec, case, baseline_ms, baseline_full_ms)
    else:  # pytorch_autograd
        # PyTorch autograd has no clean backward-only split (forward must be redone to
        # get a fresh graph), so the same full fwd+bwd number is used for both.
        baseline_ms = _median_ms(baseline_full_step)
        baseline_full_ms = baseline_ms
        _baseline_cache_put(torch_module, task_spec, case, baseline_ms, baseline_full_ms)
    saved_byte_count = _saved_bytes(saved_tensors)
    input_byte_count = _input_bytes(_memory_inputs(task_spec, inputs))
    return {
        **_case_report(task_spec, case),
        "forward_ms": forward_ms,
        "backward_from_saved_ms": backward_ms,
        "forward_backward_full_step_ms": full_step_ms,
        "raw_forward_backward_full_step_ms": full_step_ms,
        "baseline_backward_ms": baseline_ms,
        "baseline_full_step_ms": baseline_full_ms,
        "baseline_raw_full_step_ms": baseline_full_ms,
        "speedup_vs_baseline_backward": baseline_ms / max(backward_ms, 1e-9),
        "speedup_vs_baseline_full_step": baseline_full_ms / max(full_step_ms, 1e-9),
        "speedup_vs_baseline_raw_full_step": baseline_full_ms / max(full_step_ms, 1e-9),
        "saved_bytes": saved_byte_count,
        "input_bytes": input_byte_count,
        "saved_memory_ratio": saved_byte_count / max(input_byte_count, 1),
        "saved_tensors": [
            {"shape": list(t.shape), "dtype": str(t.dtype), "bytes": _tensor_bytes(t)}
            for t in saved_tensors
        ],
    }


def _run_benchmarks(torch_module, task_spec, forward_fn: Callable, backward_fn: Callable) -> dict[str, Any]:
    cases = []
    totals = {
        "forward_ms": 0.0,
        "backward_from_saved_ms": 0.0,
        "forward_backward_full_step_ms": 0.0,
        "raw_forward_backward_full_step_ms": 0.0,
        "baseline_backward_ms": 0.0,
        "baseline_full_step_ms": 0.0,
        "baseline_raw_full_step_ms": 0.0,
        "saved_bytes": 0.0,
        "input_bytes": 0.0,
    }
    for case in _selected_benchmark_cases(task_spec):
        report = _benchmark_case(torch_module, task_spec, forward_fn, backward_fn, case)
        report["case_weight"] = _case_weight(task_spec, case)
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
    totals["saved_memory_ratio"] = totals["saved_bytes"] / max(totals["input_bytes"], 1e-9)
    backward_speedups = [
        float(case["speedup_vs_baseline_backward"])
        for case in cases
        if float(case["speedup_vs_baseline_backward"]) > 0.0
    ]
    full_step_speedups = [
        float(case["speedup_vs_baseline_full_step"])
        for case in cases
        if float(case["speedup_vs_baseline_full_step"]) > 0.0
    ]
    min_speedups = [
        min(float(case["speedup_vs_baseline_backward"]), float(case["speedup_vs_baseline_full_step"]))
        for case in cases
        if float(case["speedup_vs_baseline_backward"]) > 0.0
        and float(case["speedup_vs_baseline_full_step"]) > 0.0
    ]
    totals["geomean_speedup_vs_baseline_backward"] = math.exp(
        sum(math.log(speedup) for speedup in backward_speedups) / max(len(backward_speedups), 1)
    )
    totals["geomean_speedup_vs_baseline_full_step"] = math.exp(
        sum(math.log(speedup) for speedup in full_step_speedups) / max(len(full_step_speedups), 1)
    )
    totals["geomean_min_speedup_per_case"] = math.exp(
        sum(math.log(speedup) for speedup in min_speedups) / max(len(min_speedups), 1)
    )
    totals["worst_case_min_speedup"] = min(min_speedups) if min_speedups else 0.0
    weighted_pairs = [
        (
            float(case.get("case_weight", 1.0)),
            min(float(case["speedup_vs_baseline_backward"]), float(case["speedup_vs_baseline_full_step"])),
        )
        for case in cases
        if float(case["speedup_vs_baseline_backward"]) > 0.0
        and float(case["speedup_vs_baseline_full_step"]) > 0.0
    ]
    if weighted_pairs:
        weight_sum = sum(w for w, _ in weighted_pairs)
        totals["weighted_geomean_min_speedup_per_case"] = math.exp(
            sum(w * math.log(s) for w, s in weighted_pairs) / max(weight_sum, 1e-9)
        )
    else:
        totals["weighted_geomean_min_speedup_per_case"] = 0.0
    return {"aggregate": totals, "cases": cases}


def _score_from_aggregate(aggregate: dict[str, float]) -> tuple[float, dict[str, float]]:
    backward_speedup = float(aggregate["speedup_vs_baseline_backward"])
    full_step_speedup = float(aggregate["speedup_vs_baseline_full_step"])
    saved_memory_ratio = float(aggregate["saved_memory_ratio"])
    min_speedup = min(backward_speedup, full_step_speedup)
    geomean_min_speedup = float(aggregate.get("geomean_min_speedup_per_case", min_speedup))
    weighted_geomean_min_speedup = float(
        aggregate.get("weighted_geomean_min_speedup_per_case", geomean_min_speedup)
    )
    worst_case_min_speedup = float(aggregate.get("worst_case_min_speedup", min_speedup))
    memory_penalty_factor = 1.0 + MEMORY_PENALTY_WEIGHT * saved_memory_ratio

    if SCORE_MODE == "speed_memory":
        score = (0.5 * backward_speedup + 0.5 * full_step_speedup) / memory_penalty_factor
    elif SCORE_MODE == "speed_memory_min":
        score = min_speedup / memory_penalty_factor
    elif SCORE_MODE == "speed_memory_min_geomean":
        score = geomean_min_speedup / memory_penalty_factor
    elif SCORE_MODE == "speed_memory_min_weighted_geomean":
        # Weighted geomean of per-case min-speedup, with a worst-case guard that only
        # bites when some case is actually slower than the baseline (min-speedup < 1).
        worst_case_guard = min(1.0, worst_case_min_speedup) if worst_case_min_speedup > 0.0 else 0.0
        score = weighted_geomean_min_speedup * worst_case_guard / memory_penalty_factor
    else:
        score = backward_speedup

    return score, {
        "backward_speedup": backward_speedup,
        "full_step_speedup": full_step_speedup,
        "min_speedup": min_speedup,
        "geomean_min_speedup": geomean_min_speedup,
        "weighted_geomean_min_speedup": weighted_geomean_min_speedup,
        "worst_case_min_speedup": worst_case_min_speedup,
        "saved_memory_ratio": saved_memory_ratio,
        "memory_penalty_factor": memory_penalty_factor,
        "score_mode_speed_memory": 1.0 if SCORE_MODE == "speed_memory" else 0.0,
        "score_mode_speed_memory_min": 1.0 if SCORE_MODE == "speed_memory_min" else 0.0,
        "score_mode_speed_memory_min_geomean": 1.0 if SCORE_MODE == "speed_memory_min_geomean" else 0.0,
        "score_mode_speed_memory_min_weighted_geomean": (
            1.0 if SCORE_MODE == "speed_memory_min_weighted_geomean" else 0.0
        ),
    }


def evaluate_autograd_pair_program(program_path: str, task_spec, run_benchmarks: bool = True) -> EvaluationResult:
    _restore_std_fds()  # heal a redirect leaked by a previous timed-out evaluation
    torch_module, runtime_error = check_runtime()
    if runtime_error:
        return _failure(task_spec, COMPILE_ERROR_SCORE, "RuntimeUnavailable", runtime_error)

    try:
        candidate_module = _load_module(program_path, "autograd_pair_candidate")
        forward_fn = validate_api(candidate_module, task_spec.AUTOGRAD_PAIR_FORWARD_FN_NAME)
        backward_fn = validate_api(candidate_module, task_spec.AUTOGRAD_PAIR_BACKWARD_FN_NAME)
    except Exception as exc:
        return _failure(
            task_spec,
            COMPILE_ERROR_SCORE,
            "ImportOrApiError",
            str(exc),
            {"traceback": traceback.format_exc(limit=8)},
        )

    correctness = None
    native_output_path = None
    try:
        with _capture_native_output() as captured_path:
            native_output_path = captured_path
            correctness = _run_correctness(
                torch_module,
                task_spec,
                forward_fn,
                backward_fn,
                task_spec.CORRECTNESS_CASES,
            )
            if correctness["passed"] == correctness["total"] and run_benchmarks:
                benchmark = _run_benchmarks(torch_module, task_spec, forward_fn, backward_fn)
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
        return _result({"combined_score": COMPILE_ERROR_SCORE, "correct": 0.0}, artifacts)

    native_output = _native_output_tail(native_output_path)
    if correctness["passed"] != correctness["total"]:
        metrics = {
            "combined_score": CORRECTNESS_ERROR_SCORE + float(correctness["partial_correctness"]),
            "correct": 0.0,
            "partial_correctness": float(correctness["partial_correctness"]),
        }
        metrics.update({f"{name}_correct": float(correctness[f"{name}_correctness"]) for name in task_spec.OUTPUT_NAMES})
        artifacts = {"correctness": correctness}
        if native_output and native_output.get("bytes", 0) > 0:
            artifacts["native_output"] = native_output
        return _result(metrics, artifacts)

    if not run_benchmarks:
        # Correctness-only mode (used to VET THE SEED). The seed only has to be a *correct* starting
        # point; making it also clear the full benchmark grid — including the pathological largest
        # shapes whose cross-row reduction needs row-tiling — is evolve's job (the `large`
        # specialist), not the seed's. Requiring it here rejects correct seeds for a problem they
        # are not meant to solve yet.
        #
        # But every benchmark shape still gets ONE untimed fw+bwd pass: a seed whose kernel
        # DEADLOCKS on a benchmark-only shape would otherwise wedge evolve (openevolve's eval
        # timeout abandons the thread, and the orphaned CUDA kernel spins on the GPU forever).
        # Here the hang happens inside the seed gate's own subprocess, whose timeout-kill tears
        # the kernel down and rejects the seed. Slow-but-finite is fine; hanging is not.
        smoke_failure = _smoke_benchmark_shapes(torch_module, task_spec, forward_fn, backward_fn)
        if smoke_failure is not None:
            return _result(
                {"combined_score": CORRECTNESS_ERROR_SCORE, "correct": 0.0, "partial_correctness": 1.0},
                {"correctness": correctness, "failure": smoke_failure})
        metrics = {"combined_score": 1.0, "correct": 1.0, "partial_correctness": 1.0}
        metrics.update({f"{name}_correct": 1.0 for name in task_spec.OUTPUT_NAMES})
        return _result(metrics, {"correctness": correctness})

    aggregate = benchmark["aggregate"]
    combined_score, score_details = _score_from_aggregate(aggregate)
    metrics = {
        "combined_score": float(combined_score),
        "correct": 1.0,
        "partial_correctness": 1.0,
        "speedup": float(aggregate["speedup_vs_baseline_backward"]),
        "full_step_speedup": float(aggregate["speedup_vs_baseline_full_step"]),
        "min_speedup": float(score_details["min_speedup"]),
        "geomean_min_speedup": float(score_details["geomean_min_speedup"]),
        "weighted_geomean_min_speedup": float(score_details["weighted_geomean_min_speedup"]),
        "worst_case_min_speedup": float(score_details["worst_case_min_speedup"]),
        "forward_ms": float(aggregate["forward_ms"]),
        "backward_from_saved_ms": float(aggregate["backward_from_saved_ms"]),
        "forward_backward_full_step_ms": float(aggregate["forward_backward_full_step_ms"]),
        "raw_forward_backward_full_step_ms": float(aggregate["raw_forward_backward_full_step_ms"]),
        "baseline_latency_ms": float(aggregate["baseline_backward_ms"]),
        "baseline_full_step_ms": float(aggregate["baseline_full_step_ms"]),
        "baseline_raw_full_step_ms": float(aggregate["baseline_raw_full_step_ms"]),
        "saved_bytes": float(aggregate["saved_bytes"]),
        "input_bytes": float(aggregate["input_bytes"]),
        "saved_memory_ratio": float(score_details["saved_memory_ratio"]),
        "memory_penalty_factor": float(score_details["memory_penalty_factor"]),
        "score_mode_speed_memory": float(score_details["score_mode_speed_memory"]),
        "score_mode_speed_memory_min": float(score_details["score_mode_speed_memory_min"]),
        "score_mode_speed_memory_min_geomean": float(score_details["score_mode_speed_memory_min_geomean"]),
    }
    metrics.update({f"{name}_correct": 1.0 for name in task_spec.OUTPUT_NAMES})
    benchmark["score_mode"] = SCORE_MODE
    benchmark["performance_baseline"] = PERFORMANCE_BASELINE
    benchmark["suite"] = SUITE
    benchmark["memory_penalty_weight"] = MEMORY_PENALTY_WEIGHT
    benchmark["warmup"] = BENCHMARK_WARMUP
    benchmark["reps"] = BENCHMARK_REPS
    return _result(metrics, {"correctness": correctness, "benchmark": benchmark})


def main(argv: list[str], task_spec, run_benchmarks: bool = True) -> int:
    if len(argv) != 2:
        print(f"Usage: {argv[0]} PROGRAM_PATH")
        return 2

    result = evaluate_autograd_pair_program(argv[1], task_spec, run_benchmarks=run_benchmarks)
    payload = {"metrics": result.metrics, "artifacts": result.artifacts}
    # evaluate_isolated() reads the result from this side-channel file — stdout can carry
    # arbitrary candidate/Triton noise, so it is not a reliable transport for the JSON.
    result_path = os.environ.get("AUTOGRAD_PAIR_RESULT_JSON")
    if result_path:
        with open(result_path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh)
    print(json.dumps(payload, indent=2))
    return 0 if result.metrics.get("correct", 0.0) == 1.0 else 1


EVAL_ISOLATION_TIMEOUT = int(os.environ.get("AUTOGRAD_PAIR_EVAL_ISOLATION_TIMEOUT", "850"))


def evaluate_isolated(evaluator_file: str, program_path: str, task_spec, timeout: int | None = None) -> EvaluationResult:
    """Run `evaluator_file PROGRAM_PATH` in a fresh subprocess; SIGKILL it on timeout.

    openevolve's own evaluation timeout merely abandons the worker THREAD — a deadlocked Triton
    kernel keeps spinning on the GPU forever and wedges every later evaluation on the node.
    Killing a subprocess destroys its CUDA context, which tears the kernel down with it. (The
    scaffolded config's max_tasks_per_child=1 was meant to provide this isolation, but Python
    3.10's ProcessPoolExecutor ignores it.) The default timeout sits just under the scaffolded
    openevolve evaluator timeout (900s) so the kill happens HERE, where it works.
    """
    timeout = timeout or EVAL_ISOLATION_TIMEOUT
    fd, result_path = tempfile.mkstemp(prefix="autograd_pair_result_", suffix=".json")
    os.close(fd)
    env = {**os.environ, "AUTOGRAD_PAIR_RESULT_JSON": result_path}
    try:
        try:
            proc = subprocess.run(
                [sys.executable, evaluator_file, program_path],
                capture_output=True, text=True, timeout=timeout, env=env)
        except subprocess.TimeoutExpired:
            return _failure(
                task_spec, CORRECTNESS_ERROR_SCORE, "EvaluationTimeoutKilled",
                f"evaluation exceeded {timeout}s and was killed (deadlocked/pathological kernel?)")
        try:
            with open(result_path, encoding="utf-8") as fh:
                payload = json.load(fh)
        except Exception:
            return _failure(
                task_spec, COMPILE_ERROR_SCORE, "IsolatedEvalCrashed",
                f"evaluator subprocess exited rc={proc.returncode} without a result",
                {"stdout_tail": proc.stdout[-4000:], "stderr_tail": proc.stderr[-4000:]})
        return EvaluationResult(metrics=payload["metrics"], artifacts=payload.get("artifacts", {}))
    finally:
        try:
            os.unlink(result_path)
        except OSError:
            pass
