"""Phase 2 — LLM synthesizer for the Liger strong-baseline wrapper, with a closed-loop gate.

The single hardest, most bug-prone file in a benchmark case is
`strong_baselines/liger_<op>.py`: it must wrap Liger's *raw* forward/backward ops (not the
autograd `Function`) into this benchmark's saved-tensor autograd-pair contract, matching the
pure-PyTorch math exactly, while dodging Liger's gotchas (non-tensor forward state that must be
re-derived, `in_place=True` mutating the cotangent, casting-mode/offset drift, the lazy
`torch.distributed.tensor` import). Doing it by hand cost two closed-loop bugs on rmsnorm.

This module automates that. It reuses:
  - `generate_with_openai_compatible_api` (shared LLM client, gpt-5.5),
  - the fusion agent's attempt→verify→repair loop shape,
  - `verify_liger_baseline.py` as the correctness gate (forward + all grads vs the PyTorch
    oracles over the correctness cases + a couple of the largest benchmark shapes).

It expects `task_spec.py` to already exist in the bench dir (from the Phase-1 scaffold + the
Phase-3 task-spec body) with `make_liger_autograd_pair_fns()` wired to import this wrapper.
The synthesizer only writes `strong_baselines/liger_<op>.py` and iterates until the gate passes.
"""

from __future__ import annotations

import argparse
import importlib.util
import re
import subprocess
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path

from pipeline.shared.llm_client import generate_with_openai_compatible_api


REPO_ROOT = Path(__file__).resolve().parents[2]
_VERIFY_SCRIPT = REPO_ROOT / "benchmark" / "triton_backward_bench_common" / "verify_liger_baseline.py"


# ZERO-SHOT: the prompt carries no example wrapper.
#
# This used to paste a known-correct wrapper in as a few-shot. That is a liability with no upside:
# the Liger wrapper is not the product — it is an adapter that brings Liger in as a perf baseline —
# and its correctness is already guaranteed by an independent gate (verify_liger_baseline, PyTorch
# oracle). An example buys nothing the gate doesn't already enforce, while coupling every op's
# synthesis to whatever other op happened to be in the registry, and (when the example was the SAME
# op) letting the model copy the answer outright. The rules below state the contract and the sharp
# edges explicitly; that is what the model actually needs.


@dataclass(frozen=True)
class LigerWrapperConfig:
    bench_dir: Path             # benchmark/triton_<op>_backward_bench
    op: str                     # "geglu"
    liger_source_paths: tuple[Path, ...]  # Liger op module(s) to show the LLM
    # LLM
    api_base: str
    model: str
    api_key: str | None
    max_attempts: int
    max_tokens: int
    temperature: float | None
    timeout: int
    python: str
    verify_extra_large: int = 2
    dry_run: bool = False


# --------------------------------------------------------------------------- contract

@dataclass(frozen=True)
class _Contract:
    api: str
    task_context: str
    output_names: tuple[str, ...]
    forward_fn: str
    backward_fn: str


def _load_contract(bench_dir: Path) -> _Contract:
    """Read the autograd-pair contract constants out of the case's task_spec.py.

    task_spec's top-level imports are lightweight (no torch), so this is cheap and needs no GPU.
    """
    spec = importlib.util.spec_from_file_location(
        f"ts_{uuid.uuid4().hex}", str(bench_dir / "task_spec.py")
    )
    ts = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ts)
    return _Contract(
        api=ts.AUTOGRAD_PAIR_API,
        task_context=getattr(ts, "AUTOGRAD_PAIR_TASK_CONTEXT", ""),
        output_names=tuple(ts.OUTPUT_NAMES),
        forward_fn=ts.AUTOGRAD_PAIR_FORWARD_FN_NAME,
        backward_fn=ts.AUTOGRAD_PAIR_BACKWARD_FN_NAME,
    )


# --------------------------------------------------------------------------- prompts

SYSTEM_MESSAGE = (
    "You are an expert at wrapping the Liger-Kernel library's raw Triton ops into a strict "
    "saved-tensor autograd-pair contract for a microbenchmark. You call Liger's raw "
    "`*_forward` / `*_backward` functions directly (NEVER the autograd `Function`, and NEVER "
    "PyTorch reference math), and you make the wrapper's forward+backward numerically match a "
    "given pure-PyTorch reference exactly. You know Liger's sharp edges cold: forward ops return "
    "non-tensor state (BLOCK_SIZE / num_warps / casting-mode) that backward needs but that a "
    "tensor-only `saved_tensors` cannot carry, so it must be RE-DERIVED in backward the same way "
    "the forward computed it; several backward ops default to `in_place=True` and overwrite the "
    "incoming cotangent (fatal in a timing loop that reuses dY); casting_mode/offset must match "
    "the reference or fp16 drifts; and some ops lazily touch `torch.distributed.tensor` so it "
    "must be imported. You output ONE self-contained Python module and nothing else."
)


def _rules(op: str, contract: _Contract) -> str:
    names = ", ".join(contract.output_names)
    return f"""HARD REQUIREMENTS for `strong_baselines/liger_{op}.py`:

1. Define exactly these public names:
     - `liger_available() -> bool`  (True iff the needed liger module imports)
     - `make_liger_{op}_autograd_pair_fns() -> (forward_with_saved, backward_from_saved)`

2. The returned pair MUST match this benchmark's contract:

{contract.api}

   - `forward_with_saved(*forward_args)` returns `(y, saved_tensors)`.
   - `saved_tensors` MUST be a FLAT `tuple` of `torch.Tensor` ONLY — no ints, strings, None,
     dtypes, or nested containers. Any non-tensor state the Liger backward needs (BLOCK_SIZE,
     num_warps, casting-mode, etc.) must be RE-DERIVED inside `backward_from_saved`, exactly the
     way the Liger forward derived it (e.g. `calculate_settings(n_cols)`), NOT saved.
   - `backward_from_saved(dout, saved_tensors, *extras)` returns the gradients in this order:
     ({names}).

3. Numerically match the pure-PyTorch reference math below (same convention: casting_mode/offset
   chosen so the wrapped forward equals the reference forward):

{contract.task_context}

4. Call Liger's RAW ops (the module-level `*_forward` / `*_backward`), not the autograd Function
   and not a reimplementation. Make inputs `.contiguous()`. If any liger op is known to lazily
   reference `torch.distributed.tensor`, import it at module top.
   CRITICAL — no mutation of reused state. The timing harness calls forward ONCE and then calls
   `backward_from_saved` MANY times on the SAME `dout` and the SAME `saved_tensors`, so backward
   must be pure w.r.t. both. If a liger backward op has an `in_place`/`inplace` flag, force it
   False. If a liger backward op mutates its inputs in place WITHOUT a flag (e.g. it stores the
   gradients back into the saved activation tensors, or overwrites `dout`), you MUST `.clone()`
   those tensors inside `backward_from_saved` before passing them in, so that repeated backward
   calls on the same saved state all return identical, correct gradients.

5. Output ONLY the complete module as one ```python code block. No prose."""


def render_codegen_prompt(op: str, contract: _Contract, liger_source: str) -> str:
    return f"""Write the Liger strong-baseline wrapper `strong_baselines/liger_{op}.py`.

{_rules(op, contract)}

=== Liger source you must wrap (call these raw ops) ===
```python
{liger_source}
```

Now produce `liger_{op}.py`."""


def render_repair_prompt(op: str, contract: _Contract, previous_code: str, gate_output: str) -> str:
    return f"""Your previous `liger_{op}.py` FAILED the correctness gate. Fix it.

The gate runs `make_liger_{op}_autograd_pair_fns()` and compares the wrapped forward and every
gradient against the pure-PyTorch oracle over the correctness cases plus the largest benchmark
shapes. Below is the gate output. `FAIL` on a column means that output mismatched; a traceback
means the wrapper crashed or imported wrong. `abs`/`rel` are the max abs/rel error per shape.

=== gate output ===
{gate_output}

=== your previous code ===
```python
{previous_code}
```

Common causes: saved non-tensor state (contract violation), NOT re-deriving BLOCK_SIZE/num_warps
so backward used wrong settings, in_place backward mutating dout across the repeated calls, a
casting_mode/offset that doesn't match the reference (fp16 drift), or calling the autograd
Function / a reimplementation instead of the raw liger ops. If a large-row reduction output (e.g.
a weight gradient) is only slightly off, that is fp reduction-order noise and the case's task_spec
already loosens its atol — do NOT change the math to chase it; re-check the structural issues.

{_rules(op, contract)}

Output ONLY the corrected complete module as one ```python code block."""


# --------------------------------------------------------------------------- gate

def _strip_code_fence(text: str) -> str:
    match = re.search(r"```(?:python)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip() + "\n"
    return text.strip() + "\n"


# Gate failures that are OUR harness's fault, not the LLM's wrapper. The repair loop tells the LLM
# "your code failed the gate" — when the gate never even reached its code, that is a lie, and every
# attempt burns an LLM call fixing a problem that does not exist. Abort loudly and point at us.
# (`make_liger_autograd_pair_fns` missing = task_spec.py was generated without the Liger hook.)
_HARNESS_BUG_MARKERS = (
    "has no attribute 'make_liger_autograd_pair_fns'",
    "no attribute 'make_liger_autograd_pair_fns'",
    "verify_liger_baseline.py\", line",   # traceback inside the gate script itself
)


def _harness_bug(gate_output: str, wrapper_path: Path) -> str | None:
    """Return an explanation if the gate died on OUR scaffolding rather than the wrapper."""
    if "make_liger_autograd_pair_fns" in gate_output and "has no attribute" in gate_output:
        return (
            f"the case's task_spec.py does not define make_liger_autograd_pair_fns(), so the gate "
            f"could not reach the wrapper at all. Regenerate task_spec with a Liger perf baseline "
            f"(synthesize_task_spec emits the hook only when the baseline is 'liger')."
        )
    return None


def _run_gate(config: LigerWrapperConfig) -> tuple[bool, str]:
    """Run verify_liger_baseline.py against the case. Returns (passed, combined_output)."""
    cmd = [
        config.python,
        str(_VERIFY_SCRIPT),
        "--bench",
        str(config.bench_dir),
        "--extra-large",
        str(config.verify_extra_large),
    ]
    completed = subprocess.run(
        cmd, cwd=str(REPO_ROOT), text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
    )
    out = completed.stdout or ""
    if completed.stderr:
        out += "\n=== stderr ===\n" + completed.stderr
    passed = completed.returncode == 0 and "ALL PASS" in (completed.stdout or "")
    return passed, out.strip()


# --------------------------------------------------------------------------- driver

def synthesize_liger_wrapper(config: LigerWrapperConfig) -> int:
    bench_dir = config.bench_dir.resolve()
    contract = _load_contract(bench_dir)
    liger_source = "\n\n# ---- next liger source file ----\n\n".join(
        p.read_text(encoding="utf-8") for p in config.liger_source_paths
    )

    work = bench_dir / ".liger_wrapper_synth"
    work.mkdir(parents=True, exist_ok=True)
    wrapper_path = bench_dir / "strong_baselines" / f"liger_{config.op}.py"
    wrapper_path.parent.mkdir(parents=True, exist_ok=True)

    prompt = render_codegen_prompt(config.op, contract, liger_source)
    (work / "codegen_prompt.md").write_text(prompt, encoding="utf-8")

    if config.dry_run:
        print(f"dry-run: wrote prompt to {work / 'codegen_prompt.md'}")
        return 0

    previous_code = ""
    for attempt in range(1, config.max_attempts + 1):
        attempt_dir = work / f"attempt_{attempt:03d}"
        attempt_dir.mkdir(parents=True, exist_ok=True)
        (attempt_dir / "prompt.md").write_text(prompt, encoding="utf-8")
        print(f"Liger-wrapper: codegen attempt {attempt}/{config.max_attempts}")

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
        (attempt_dir / "response.txt").write_text(response, encoding="utf-8")
        wrapper_path.write_text(code, encoding="utf-8")

        passed, gate_output = _run_gate(config)
        (attempt_dir / "gate_output.txt").write_text(gate_output, encoding="utf-8")
        (attempt_dir / f"liger_{config.op}.py").write_text(code, encoding="utf-8")

        if passed:
            print(f"Liger-wrapper synthesis passed on attempt {attempt}: {wrapper_path}")
            return 0

        why = _harness_bug(gate_output, wrapper_path)
        if why is not None:
            print(f"\nABORT — this is a HARNESS bug, not an LLM bug: {why}\n"
                  f"Not feeding this back to the LLM as 'your code is wrong'. "
                  f"Gate output in {attempt_dir / 'gate_output.txt'}")
            return 2

        prompt = render_repair_prompt(config.op, contract, code or previous_code, gate_output)
        previous_code = code
        print(f"attempt {attempt} failed the gate; wrote repair prompt")

    print(f"Liger-wrapper synthesis FAILED after {config.max_attempts} attempts (see {work})")
    return 1


# --------------------------------------------------------------------------- CLI

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Phase 2: synthesize + verify a Liger baseline wrapper")
    ap.add_argument("--bench", required=True, help="benchmark/triton_<op>_backward_bench dir")
    ap.add_argument("--op", required=True, help="op name, e.g. geglu")
    ap.add_argument(
        "--liger-source", action="append", required=True,
        help="path to a Liger op source file to wrap (repeatable)",
    )
    ap.add_argument("--api-base", default="https://api.openai.com/v1")
    ap.add_argument("--model", default="gpt-5.5")
    ap.add_argument("--api-key", default=None)
    ap.add_argument("--max-attempts", type=int, default=4)
    ap.add_argument("--max-tokens", type=int, default=16000)
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--timeout", type=int, default=180)
    ap.add_argument("--extra-large", type=int, default=2)
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv or sys.argv[1:])

    return synthesize_liger_wrapper(
        LigerWrapperConfig(
            bench_dir=Path(args.bench),
            op=args.op,
            liger_source_paths=tuple(Path(p) for p in args.liger_source),
            api_base=args.api_base,
            model=args.model,
            api_key=args.api_key,
            max_attempts=args.max_attempts,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            timeout=args.timeout,
            python=args.python,
            verify_extra_large=args.extra_large,
            dry_run=args.dry_run,
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())
