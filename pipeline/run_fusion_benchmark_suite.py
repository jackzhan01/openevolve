"""Batch Pipeline A over the non-LayerNorm backward benchmarks.

The runner is intentionally manifest-driven: each benchmark declares its forward
reference, public backward API, verifier tolerances, and OpenEvolve config in one
place.  It runs the direct AtenIR fusion agent, gates the seed with the benchmark
evaluator, and optionally launches a short OpenEvolve run.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class BenchSpec:
    name: str
    forward: str
    public_api: str
    api_signature: str
    return_contract: str
    op: str
    example_input: str
    evaluator: str
    config: str
    save_best_to: str
    scalars: tuple[str, ...] = ()
    backward_input_indices: tuple[int, ...] | None = None
    atol: float = 2e-5
    rtol: float = 2e-5
    fp16_atol: float = 5e-2
    fp16_rtol: float = 5e-2


BENCHMARKS: dict[str, BenchSpec] = {
    "rmsnorm": BenchSpec(
        name="rmsnorm",
        forward="benchmark.triton_rmsnorm_backward_bench.forward_ref:rmsnorm_forward_ref",
        public_api="rmsnorm_backward_triton",
        api_signature="def rmsnorm_backward_triton(dy, x, weight, eps=1e-5):",
        return_contract="return dx, dweight",
        op="rmsnorm",
        example_input="[(8,64) f32, (64) f32]",
        evaluator="benchmark/triton_rmsnorm_backward_bench/evaluator.py",
        config="benchmark/triton_rmsnorm_backward_bench/config.yaml",
        save_best_to="benchmark/triton_rmsnorm_backward_bench/evolved_best_pipeline_A.py",
        scalars=("1e-5",),
    ),
    "matmul": BenchSpec(
        name="matmul",
        forward="benchmark.triton_matmul_backward_bench.forward_ref:matmul_forward_ref",
        public_api="matmul_backward_triton",
        api_signature="def matmul_backward_triton(dc, a, b):",
        return_contract="return da, db",
        op="matmul",
        example_input="[(64,64) f32, (64,64) f32]",
        evaluator="benchmark/triton_matmul_backward_bench/evaluator.py",
        config="benchmark/triton_matmul_backward_bench/config.yaml",
        save_best_to="benchmark/triton_matmul_backward_bench/evolved_best_pipeline_A.py",
        atol=8e-2,
        rtol=2e-2,
        fp16_atol=1e-1,
        fp16_rtol=2e-2,
    ),
    "linear": BenchSpec(
        name="linear",
        forward="benchmark.triton_linear_backward_bench.forward_ref:linear_forward_ref",
        public_api="linear_backward_triton",
        api_signature="def linear_backward_triton(dy, x, weight):",
        return_contract=(
            "return dx, dweight, dbias  # bias is not an input to this API; "
            "infer dbias shape from dy/weight"
        ),
        op="linear",
        example_input="[(64,64) f32, (64,64) f32, (64) f32]",
        evaluator="benchmark/triton_linear_backward_bench/evaluator.py",
        config="benchmark/triton_linear_backward_bench/config.yaml",
        save_best_to="benchmark/triton_linear_backward_bench/evolved_best_pipeline_A.py",
        backward_input_indices=(0, 1),
        atol=8e-2,
        rtol=2e-2,
        fp16_atol=1e-1,
        fp16_rtol=2e-2,
    ),
}


def _repo_env() -> dict[str, str]:
    env = os.environ.copy()
    paths = [str(REPO_ROOT), str(REPO_ROOT / "benchmark"), str(REPO_ROOT / "examples")]
    if env.get("PYTHONPATH"):
        paths.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(paths)
    return env


def _redact_command(cmd: list[str]) -> list[str]:
    redacted = list(cmd)
    for index, value in enumerate(redacted[:-1]):
        if value == "--api-key":
            redacted[index + 1] = "<redacted>"
    return redacted


def _run(
    cmd: list[str],
    *,
    cwd: Path,
    log_path: Path,
    env: dict[str, str],
    continue_on_error: bool,
    capture_output: bool = False,
) -> dict[str, Any]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    start = time.time()
    stdout_lines: list[str] = []
    stderr_lines: list[str] = []
    write_lock = threading.Lock()
    redacted_cmd = _redact_command(cmd)

    with log_path.open("w", encoding="utf-8") as log_file:
        log_file.write("COMMAND:\n" + " ".join(redacted_cmd) + "\n\nOUTPUT:\n")
        log_file.flush()
        process = subprocess.Popen(
            cmd,
            cwd=str(cwd),
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=1,
        )

        def _stream(stream, label: str, sink: list[str], terminal) -> None:
            assert stream is not None
            for line in stream:
                sink.append(line)
                terminal.write(line)
                terminal.flush()
                with write_lock:
                    log_file.write(f"[{label}] {line}")
                    log_file.flush()

        stdout_thread = threading.Thread(
            target=_stream,
            args=(process.stdout, "stdout", stdout_lines, sys.stdout),
            daemon=True,
        )
        stderr_thread = threading.Thread(
            target=_stream,
            args=(process.stderr, "stderr", stderr_lines, sys.stderr),
            daemon=True,
        )
        stdout_thread.start()
        stderr_thread.start()
        returncode = process.wait()
        stdout_thread.join()
        stderr_thread.join()

    result: dict[str, Any] = {
        "cmd": redacted_cmd,
        "returncode": returncode,
        "elapsed_sec": time.time() - start,
        "log": str(log_path),
    }
    if capture_output:
        result["stdout"] = "".join(stdout_lines)
        result["stderr"] = "".join(stderr_lines)
    if returncode != 0 and not continue_on_error:
        raise RuntimeError(f"command failed ({returncode}); see {log_path}")
    return result


def _load_json_from_stdout(stdout: str) -> dict[str, Any] | None:
    try:
        return json.loads(stdout)
    except json.JSONDecodeError:
        start = stdout.find("{")
        end = stdout.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return None
        try:
            return json.loads(stdout[start : end + 1])
        except json.JSONDecodeError:
            return None


def _classify_log_failure(log_path: str | None) -> str | None:
    if not log_path:
        return None
    path = Path(log_path)
    if not path.exists():
        return None
    text = path.read_text(encoding="utf-8", errors="replace")
    if "AuthenticationError" in text or "invalid_api_key" in text:
        return "LLMAuthenticationError"
    if "RateLimitError" in text:
        return "LLMRateLimitError"
    if "CUDA is not available" in text:
        return "RuntimeUnavailable"
    if "AtenIR extraction failed" in text:
        return "AtenIRExtractionError"
    if "Fusion synthesis failed" in text:
        return "FusionVerificationFailed"
    if "Traceback" in text:
        return "UnhandledException"
    return None


def _fusion_command(args: argparse.Namespace, spec: BenchSpec, fusion_dir: Path) -> list[str]:
    cmd = [
        args.python,
        "-m",
        "pipeline.run_fusion_agent",
        "--forward",
        spec.forward,
        "--example-input",
        spec.example_input,
        "--public-api",
        spec.public_api,
        "--api-signature",
        spec.api_signature,
        "--return-contract",
        spec.return_contract,
        "--op",
        spec.op,
        "--mode",
        args.mode,
        "--output-dir",
        str(fusion_dir),
        "--api-base",
        args.api_base,
        "--model",
        args.model,
        "--max-attempts",
        str(args.fusion_max_attempts),
        "--max-tokens",
        str(args.max_tokens),
        "--temperature",
        str(args.temperature),
        "--timeout",
        str(args.timeout),
        "--atol",
        str(spec.atol),
        "--rtol",
        str(spec.rtol),
        "--fp16-atol",
        str(spec.fp16_atol),
        "--fp16-rtol",
        str(spec.fp16_rtol),
    ]
    if args.api_key:
        cmd.extend(["--api-key", args.api_key])
    for dtype in args.dtype:
        cmd.extend(["--dtype", dtype])
    for scalar in spec.scalars:
        cmd.extend(["--scalar", scalar])
    if spec.backward_input_indices is not None:
        indices = ",".join(str(index) for index in spec.backward_input_indices)
        cmd.extend(["--backward-input-indices", indices])
    if args.dry_run_agents:
        cmd.append("--dry-run")
    return cmd


def _run_case(args: argparse.Namespace, spec: BenchSpec, env: dict[str, str]) -> dict[str, Any]:
    case_dir = args.output_dir / spec.name
    fusion_dir = case_dir / "fusion"
    evolve_dir = case_dir / f"openevolve_{args.iterations}"
    seed_path = fusion_dir / "best" / "initial_program_from_atenir.py"
    case_summary: dict[str, Any] = {
        "case": spec.name,
        "fusion_dir": str(fusion_dir),
        "seed_path": str(seed_path),
    }

    if args.reuse_existing_seeds and seed_path.exists():
        case_summary["fusion"] = {"skipped": True, "reason": "existing seed reused"}
    else:
        case_summary["fusion"] = _run(
            _fusion_command(args, spec, fusion_dir),
            cwd=REPO_ROOT,
            log_path=case_dir / "run_fusion_agent.log",
            env=env,
            continue_on_error=True,
        )
    fusion = case_summary.get("fusion", {})
    if isinstance(fusion, dict) and fusion.get("returncode", 0) != 0:
        case_summary["fusion_failure_type"] = _classify_log_failure(fusion.get("log"))

    case_summary["seed_exists"] = seed_path.exists()
    if args.dry_run_agents or not seed_path.exists():
        case_summary["evaluator"] = {"skipped": True, "reason": "no seed or dry-run"}
        case_summary["openevolve"] = {"skipped": True, "reason": "no evaluator pass"}
        return case_summary

    evaluator_result = _run(
        [args.python, spec.evaluator, str(seed_path)],
        cwd=REPO_ROOT,
        log_path=case_dir / "evaluator.log",
        env=env,
        continue_on_error=True,
        capture_output=True,
    )
    evaluator_json = _load_json_from_stdout(str(evaluator_result.get("stdout", "")))
    if evaluator_json is not None:
        (case_dir / "evaluator_result.json").write_text(
            json.dumps(evaluator_json, indent=2, sort_keys=True),
            encoding="utf-8",
        )
    evaluator_result["parsed_json"] = evaluator_json
    case_summary["evaluator"] = evaluator_result

    metrics = (evaluator_json or {}).get("metrics", {})
    evaluator_passed = bool(metrics.get("correct") == 1.0)
    case_summary["evaluator_passed"] = evaluator_passed
    if args.no_evolve or not evaluator_passed:
        reason = "disabled by --no-evolve" if args.no_evolve else "evaluator did not pass"
        case_summary["openevolve"] = {"skipped": True, "reason": reason}
        return case_summary

    evolve_cmd = [
        args.openevolve_command,
        str(seed_path),
        spec.evaluator,
        "--config",
        spec.config,
        "--api-base",
        args.api_base,
        "--primary-model",
        args.model,
        "--secondary-model",
        args.model,
        "--iterations",
        str(args.iterations),
        "--output",
        str(evolve_dir),
        "--save-best-to",
        spec.save_best_to,
    ]
    case_summary["openevolve"] = _run(
        evolve_cmd,
        cwd=REPO_ROOT,
        log_path=case_dir / "openevolve.log",
        env=env,
        continue_on_error=True,
    )
    return case_summary


def _write_markdown_summary(summary: dict[str, Any], path: Path) -> None:
    lines = ["# Pipeline A Benchmark Suite Summary", ""]
    for item in summary["cases"]:
        evaluator = item.get("evaluator", {})
        parsed = evaluator.get("parsed_json") or {}
        metrics = parsed.get("metrics", {})
        lines.append(f"## {item['case']}")
        lines.append(f"- seed_exists: `{item.get('seed_exists')}`")
        fusion = item.get("fusion", {})
        if fusion:
            lines.append(f"- fusion_returncode: `{fusion.get('returncode')}`")
            if item.get("fusion_failure_type"):
                lines.append(f"- fusion_failure_type: `{item.get('fusion_failure_type')}`")
            if fusion.get("log"):
                lines.append(f"- fusion_log: `{fusion.get('log')}`")
        lines.append(f"- evaluator_passed: `{item.get('evaluator_passed', False)}`")
        if metrics:
            lines.append(f"- combined_score: `{metrics.get('combined_score')}`")
            lines.append(f"- speedup: `{metrics.get('speedup')}`")
        openevolve = item.get("openevolve", {})
        if openevolve.get("skipped"):
            lines.append(f"- openevolve: skipped ({openevolve.get('reason')})")
        else:
            lines.append(f"- openevolve_returncode: `{openevolve.get('returncode')}`")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch Pipeline A over non-LayerNorm benchmarks")
    parser.add_argument("--case", action="append", choices=sorted(BENCHMARKS), default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("/u/wzhan/tmp/pipeline_A_suite"))
    parser.add_argument("--mode", default="dynamic", choices=["static", "dynamic", "nontile"])
    parser.add_argument("--api-base", default="https://api.openai.com/v1")
    parser.add_argument("--model", default="gpt-5.5")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--fusion-max-attempts", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=16000)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--dtype", action="append", default=None)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--no-evolve", action="store_true")
    parser.add_argument("--dry-run-agents", action="store_true")
    parser.add_argument("--reuse-existing-seeds", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--openevolve-command", default="openevolve-run")
    args = parser.parse_args(argv)
    args.output_dir = args.output_dir.resolve()
    args.dtype = args.dtype or ["float32", "float16"]
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    selected = args.case or ["rmsnorm", "matmul", "linear"]
    env = _repo_env()
    summary: dict[str, Any] = {
        "output_dir": str(args.output_dir),
        "cases": [],
    }

    for name in selected:
        try:
            summary["cases"].append(_run_case(args, BENCHMARKS[name], env))
        except Exception as exc:
            if not args.continue_on_error:
                raise
            summary["cases"].append(
                {
                    "case": name,
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                }
            )

    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    _write_markdown_summary(summary, args.output_dir / "summary.md")
    print(f"Wrote {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
