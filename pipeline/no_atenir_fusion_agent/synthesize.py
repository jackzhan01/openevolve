"""No-AtenIR forward-reference-only autograd-pair synthesis ablation."""

from __future__ import annotations

import importlib
import inspect
import json
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pipeline.no_atenir_fusion_agent.prompts import (
    SYSTEM_MESSAGE,
    render_codegen_prompt,
    render_plan_prompt,
    render_repair_prompt,
)
from pipeline.shared.llm_client import generate_with_openai_compatible_api_metadata


REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class NoAtenIRFusionConfig:
    forward: str
    output_dir: Path
    api_base: str
    model: str
    api_key: str | None
    max_attempts: int
    max_tokens: int
    temperature: float | None
    timeout: int
    python: str
    dry_run: bool = False
    # Evaluator used to verify generated programs. Defaults to LayerNorm for
    # back-compat; point at another benchmark's evaluator_autograd_pair.py to run
    # the ablation on a different operator.
    evaluator: str = "benchmark/triton_layernorm_backward_bench/evaluator_autograd_pair.py"
    # Operator contract for the prompts. None -> LayerNorm (back-compatible).
    op_spec: Any = None


def _strip_code_fence(text: str) -> str:
    match = re.search(r"```(?:python)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip() + "\n"
    return text.strip() + "\n"


def _load_callable(spec: str):
    module_name, fn_name = spec.split(":", 1) if ":" in spec else spec.rsplit(".", 1)
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    return getattr(importlib.import_module(module_name), fn_name)


def _forward_source(forward: str) -> str:
    try:
        return inspect.getsource(_load_callable(forward)).strip()
    except Exception as exc:
        return f"# Could not inspect forward source for {forward}: {type(exc).__name__}: {exc}"


def _verify_program(config: NoAtenIRFusionConfig, program_path: Path) -> dict:
    cmd = [
        config.python,
        config.evaluator,
        str(program_path),
    ]
    completed = subprocess.run(
        cmd,
        cwd=str(REPO_ROOT),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    try:
        report = json.loads(completed.stdout)
    except json.JSONDecodeError:
        report = {
            "metrics": {"correct": 0.0},
            "artifacts": {},
            "stdout": completed.stdout,
            "stderr": completed.stderr,
            "returncode": completed.returncode,
        }
    if completed.stderr:
        report["stderr"] = completed.stderr
    return report


def _passed(report: dict) -> bool:
    return float(report.get("metrics", {}).get("correct", 0.0)) == 1.0


def _merge_usage(total: dict[str, int], metadata: dict[str, Any]) -> None:
    usage = metadata.get("usage") or {}
    for key, value in usage.items():
        if isinstance(value, int):
            total[key] = total.get(key, 0) + value


def _write_cost_summary(output_dir: Path, calls: list[dict[str, Any]], attempts: int, passed: bool) -> None:
    usage_totals: dict[str, int] = {}
    for call in calls:
        _merge_usage(usage_totals, call)
    summary = {
        "agent": "no_atenir_fusion_agent",
        "contract": "autograd_pair",
        "passed": passed,
        "attempts": attempts,
        "llm_call_count": len(calls),
        "prompt_chars_total": sum(int(call.get("prompt_chars", 0)) for call in calls),
        "response_chars_total": sum(int(call.get("response_chars", 0)) for call in calls),
        "usage_totals": usage_totals,
        "calls": calls,
    }
    (output_dir / "cost_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _call_llm(
    *,
    config: NoAtenIRFusionConfig,
    prompt: str,
    call_name: str,
    calls: list[dict[str, Any]],
) -> str:
    start = time.time()
    content, metadata = generate_with_openai_compatible_api_metadata(
        prompt=prompt,
        system_message=SYSTEM_MESSAGE,
        model=config.model,
        api_base=config.api_base,
        api_key=config.api_key,
        max_tokens=config.max_tokens,
        temperature=config.temperature,
        timeout=config.timeout,
    )
    metadata = dict(metadata)
    metadata["call_name"] = call_name
    metadata["elapsed_sec"] = time.time() - start
    calls.append(metadata)
    return content


def synthesize_no_atenir_fusion(config: NoAtenIRFusionConfig) -> int:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    forward_source = _forward_source(config.forward)
    (config.output_dir / "forward_source.py").write_text(forward_source + "\n", encoding="utf-8")

    calls: list[dict[str, Any]] = []
    plan_prompt = render_plan_prompt(
        forward=config.forward,
        forward_source=forward_source,
        spec=config.op_spec,
    )
    (config.output_dir / "plan_prompt.md").write_text(plan_prompt, encoding="utf-8")

    if config.dry_run:
        code_prompt = render_codegen_prompt(
            forward=config.forward,
            forward_source=forward_source,
            plan="{PLAN_FROM_LLM}",
            spec=config.op_spec,
        )
        dry_dir = config.output_dir / "attempt_001"
        dry_dir.mkdir(parents=True, exist_ok=True)
        (dry_dir / "codegen_prompt.md").write_text(code_prompt, encoding="utf-8")
        _write_cost_summary(config.output_dir, calls, attempts=0, passed=False)
        print(f"dry-run wrote {config.output_dir / 'plan_prompt.md'}")
        print(f"dry-run wrote {dry_dir / 'codegen_prompt.md'}")
        return 0

    print("No-AtenIR: plan synthesis")
    plan = _call_llm(config=config, prompt=plan_prompt, call_name="plan", calls=calls)
    (config.output_dir / "plan.md").write_text(plan, encoding="utf-8")

    prompt = render_codegen_prompt(
        forward=config.forward,
        forward_source=forward_source,
        plan=plan,
        spec=config.op_spec,
    )
    previous_code = ""

    for attempt in range(1, config.max_attempts + 1):
        attempt_dir = config.output_dir / f"attempt_{attempt:03d}"
        attempt_dir.mkdir(parents=True, exist_ok=True)
        (attempt_dir / "codegen_prompt.md").write_text(prompt, encoding="utf-8")
        print(f"No-AtenIR: codegen attempt {attempt}/{config.max_attempts}")
        response = _call_llm(
            config=config,
            prompt=prompt,
            call_name=f"codegen_attempt_{attempt}",
            calls=calls,
        )
        code = _strip_code_fence(response)
        program_path = attempt_dir / "program.py"
        program_path.write_text(code, encoding="utf-8")
        (attempt_dir / "response.txt").write_text(response, encoding="utf-8")

        report = _verify_program(config, program_path)
        (attempt_dir / "verification_report.json").write_text(
            json.dumps(report, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        if _passed(report):
            best_dir = config.output_dir / "best"
            best_dir.mkdir(parents=True, exist_ok=True)
            best_path = best_dir / "initial_program_no_atenir_autograd_pair.py"
            best_path.write_text(code, encoding="utf-8")
            (best_dir / "verification_report.json").write_text(
                json.dumps(report, indent=2, sort_keys=True),
                encoding="utf-8",
            )
            _write_cost_summary(config.output_dir, calls, attempts=attempt, passed=True)
            print(f"No-AtenIR synthesis passed. Best program: {best_path}")
            return 0

        repair_prompt = render_repair_prompt(
            forward=config.forward,
            forward_source=forward_source,
            plan=plan,
            previous_code=code or previous_code,
            verifier_report=json.dumps(report, indent=2, sort_keys=True),
            spec=config.op_spec,
        )
        (attempt_dir / "repair_prompt.md").write_text(repair_prompt, encoding="utf-8")
        prompt = repair_prompt
        previous_code = code
        _write_cost_summary(config.output_dir, calls, attempts=attempt, passed=False)
        print(f"attempt {attempt} failed; wrote repair prompt")

    _write_cost_summary(config.output_dir, calls, attempts=config.max_attempts, passed=False)
    print(f"No-AtenIR synthesis failed after {config.max_attempts} attempts")
    return 1
