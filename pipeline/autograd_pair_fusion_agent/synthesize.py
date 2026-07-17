"""Autograd-pair saved-tensor fusion synthesis pipeline."""

from __future__ import annotations

import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

from pipeline.fusion_agent.graph_summary import summarize_graph_file
from pipeline.shared.llm_client import generate_with_openai_compatible_api
from pipeline.autograd_pair_fusion_agent.prompts import (
    DEFAULT_PAIR_API,
    DEFAULT_TASK_CONTEXT,
    SYSTEM_MESSAGE,
    render_codegen_prompt,
    render_plan_prompt,
    render_repair_prompt,
)


@dataclass(frozen=True)
class AutogradPairConfig:
    forward: str
    example_input: str
    output_dir: Path
    evaluator: str
    api_base: str
    model: str
    api_key: str | None
    max_attempts: int
    max_tokens: int
    temperature: float | None
    timeout: int
    dtypes: tuple[str, ...]
    python: str
    eval_timeout: int = 120     # kill a hanging candidate kernel (legit seed eval ~25s, so 120s is ample)
    lowering_context: str | None = None
    pair_api: str | None = None
    task_context: str | None = None
    dry_run: bool = False


def _strip_code_fence(text: str) -> str:
    match = re.search(r"```(?:python)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip() + "\n"
    return text.strip() + "\n"


def _extract_graph(config: AutogradPairConfig) -> Path:
    graph_path = config.output_dir / "atenir_graph.json"
    cmd = [
        config.python,
        "-m",
        "atenir.extract",
        "--fn",
        config.forward,
        "--example-input",
        config.example_input,
        "--out",
        str(graph_path),
    ]
    completed = subprocess.run(
        cmd,
        cwd=str(Path(__file__).resolve().parents[2]),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    (config.output_dir / "extract_stdout.txt").write_text(completed.stdout, encoding="utf-8")
    (config.output_dir / "extract_stderr.txt").write_text(completed.stderr, encoding="utf-8")
    if completed.returncode != 0:
        raise RuntimeError(f"AtenIR extraction failed:\n{completed.stderr}")
    return graph_path


def _verify_program(config: AutogradPairConfig, program_path: Path) -> dict:
    cmd = [
        config.python,
        config.evaluator,
        str(program_path),
    ]
    try:
        completed = subprocess.run(
            cmd,
            cwd=str(Path(__file__).resolve().parents[2]),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=config.eval_timeout,
        )
    except subprocess.TimeoutExpired as e:
        # An LLM-generated kernel can deadlock / spin on the GPU. Without a timeout the whole
        # pipeline hangs forever on that one candidate; kill it, mark it failed, and let the next
        # attempt proceed. (subprocess.run SIGKILLs the child on timeout, freeing its CUDA context.)
        # The child's partial stderr carries the evaluator's [smoke]/progress lines — its LAST
        # line names the case that was running when the hang started, which is exactly what the
        # repair prompt needs the LLM to know.
        partial_err = (e.stderr or "") if isinstance(e.stderr, str) else ""
        return {
            "metrics": {"correct": 0.0},
            "artifacts": {},
            "stdout": (e.stdout or "") if isinstance(e.stdout, str) else "",
            "stderr": f"[eval_timeout] candidate killed after {config.eval_timeout}s "
                      f"(likely a hanging/deadlocked kernel). Progress before the kill "
                      f"(the LAST line is where it hung):\n{partial_err[-2000:]}",
            "returncode": -9,
        }
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


def synthesize_autograd_pair(config: AutogradPairConfig) -> int:
    config.output_dir.mkdir(parents=True, exist_ok=True)

    print("Extract: AtenIR reference backward graph")
    graph_path = _extract_graph(config)
    graph_summary = summarize_graph_file(graph_path)
    (config.output_dir / "graph_summary.md").write_text(graph_summary, encoding="utf-8")

    lowering_context = config.lowering_context or ""
    if lowering_context:
        (config.output_dir / "lowering_context.md").write_text(lowering_context, encoding="utf-8")
    pair_api = config.pair_api or DEFAULT_PAIR_API
    task_context = config.task_context or DEFAULT_TASK_CONTEXT
    if pair_api:
        (config.output_dir / "pair_api.py").write_text(pair_api, encoding="utf-8")
    if task_context:
        (config.output_dir / "task_context.md").write_text(task_context, encoding="utf-8")

    plan_prompt = render_plan_prompt(
        forward=config.forward,
        graph_summary=graph_summary,
        lowering_context=lowering_context,
        pair_api=pair_api,
        task_context=task_context,
    )
    (config.output_dir / "autograd_pair_plan_prompt.md").write_text(plan_prompt, encoding="utf-8")

    if config.dry_run:
        dry_dir = config.output_dir / "attempt_001"
        dry_dir.mkdir(parents=True, exist_ok=True)
        (dry_dir / "codegen_prompt.md").write_text(
            render_codegen_prompt(
                graph_summary=graph_summary,
                pair_plan="{AUTOGRAD_PAIR_PLAN_FROM_LLM}",
                lowering_context=lowering_context,
                pair_api=pair_api,
                task_context=task_context,
            ),
            encoding="utf-8",
        )
        print(f"dry-run wrote {config.output_dir}")
        return 0

    print("Autograd-pair: plan synthesis")
    pair_plan = generate_with_openai_compatible_api(
        prompt=plan_prompt,
        system_message=SYSTEM_MESSAGE,
        model=config.model,
        api_base=config.api_base,
        api_key=config.api_key,
        max_tokens=config.max_tokens,
        temperature=config.temperature,
        timeout=config.timeout,
    )
    (config.output_dir / "autograd_pair_plan.md").write_text(pair_plan, encoding="utf-8")

    prompt = render_codegen_prompt(
        graph_summary=graph_summary,
        pair_plan=pair_plan,
        lowering_context=lowering_context,
        pair_api=pair_api,
        task_context=task_context,
    )
    previous_code = ""
    for attempt in range(1, config.max_attempts + 1):
        attempt_dir = config.output_dir / f"attempt_{attempt:03d}"
        attempt_dir.mkdir(parents=True, exist_ok=True)
        (attempt_dir / "codegen_prompt.md").write_text(prompt, encoding="utf-8")
        print(f"Autograd-pair: codegen attempt {attempt}/{config.max_attempts}")
        response = generate_with_openai_compatible_api(
            prompt=prompt,
            system_message=SYSTEM_MESSAGE,
            model=config.model,
            api_base=config.api_base,
            api_key=config.api_key,
            max_tokens=config.max_tokens,
            temperature=config.temperature,
            timeout=config.timeout,
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
            best_path = best_dir / "initial_program_autograd_pair.py"
            best_path.write_text(code, encoding="utf-8")
            (best_dir / "verification_report.json").write_text(
                json.dumps(report, indent=2, sort_keys=True),
                encoding="utf-8",
            )
            print(f"Autograd-pair synthesis passed. Best program: {best_path}")
            return 0

        repair_prompt = render_repair_prompt(
            graph_summary=graph_summary,
            pair_plan=pair_plan,
            previous_code=code or previous_code,
            verifier_report=json.dumps(report, indent=2, sort_keys=True),
            lowering_context=lowering_context,
            pair_api=pair_api,
            task_context=task_context,
        )
        (attempt_dir / "repair_prompt.md").write_text(repair_prompt, encoding="utf-8")
        prompt = repair_prompt
        previous_code = code
        print(f"attempt {attempt} failed; wrote repair prompt")

    print(f"Autograd-pair synthesis failed after {config.max_attempts} attempts")
    return 1
