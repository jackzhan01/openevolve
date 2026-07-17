"""Phase 3, Part 1 — synthesize `task_spec.py` + `forward_ref.py` from a PyTorch forward.

This removes the last big piece of hand-work: given a PyTorch forward function, produce the
benchmark case's contract, oracles, input generator and tolerances — so the Phase-4 orchestrator
can take it from there (scaffold -> wrapper -> seed -> evolve -> dispatch -> report).

DESIGN — who does what (deliberate split):

  * The LLM ONLY supplies what it is actually good at, as structured JSON:
      - `make_inputs` body      (needs SEMANTICS: log-probs? probabilities? class labels in [0,V)?)
      - differentiability       (which float inputs are DATA vs a non-differentiable float target,
                                 e.g. kl_div's y_true is float but must not be differentiated)
      - config params           (non-tensor args like eps)
      - math description        (prompt text for fusion/evolve)
      - benchmark shapes/dtypes (typical LLM-scale sizes)

  * The CODE derives everything that is mechanical, and does NOT trust the LLM for it:
      - the contract (cotangent index, forward-input indices, extra/memory indices, OUTPUT_NAMES)
      - `torch_oracle`   -> ONE generic template: clone inputs, requires_grad_, forward,
                            out.backward(cotangent), return grads. (All 7 hand-written specs were
                            literally this.)
      - `autograd_pair_forward_oracle` -> the forward itself
      - tolerances by dtype

    Rationale: if the LLM wrote BOTH the oracle and the candidate, it could get them consistently
    wrong and the correctness check would pass anyway. Deriving the oracle in code kills that.

  * SELF-CHECK closed loop (the gate for this stage): build a trivial autograd-pair program from
    the forward itself (forward + PyTorch autograd as the backward) and run it through the real
    evaluator. It MUST come out 100% correct. If the evaluator errors -> the contract is wrong;
    if `correct < 1` -> the oracle/inputs are wrong. Either way, feed the failure back and repair.

The regime fields (regime_feature / REGIME_SPLIT / SMALL_CASES / LARGE_CASES) come from the SAME
LLM call that designs the benchmark shapes. That coupling is the point: the axis is a MATH FACT
about the backward (which dim does it reduce along), and the shapes must be swept ALONG that axis
or the two specialists get trained on suites that differ in every dimension at once. Code then
checks the balance (>= 4 shapes on each side of the split) before accepting the spec.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import re
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path

from pipeline.shared.llm_client import generate_with_openai_compatible_api

REPO_ROOT = Path(__file__).resolve().parents[2]

_DTYPE_TOL = {
    "float32": (2e-5, 2e-5),
    "float16": (5e-2, 5e-2),
    "bfloat16": (8e-2, 8e-2),
}

_AXIS_EXPR = {
    "rows": "case.rows",
    "cols": "case.cols",
    "numel": "case.rows * case.cols",
}


def _axis_value(axis: str, rows: int, cols: int) -> float:
    return float(rows if axis == "rows" else cols if axis == "cols" else rows * cols)


def _check_shape_balance(spec: dict, min_per_side: int = 4) -> str | None:
    """The whole point of designing shapes AROUND the regime axis: both specialists must be
    trainable. Returns an error message if either side of the split has too few shapes."""
    axis = spec.get("regime_axis", "numel")
    split = float(spec.get("regime_split", 1_000_000))
    shapes = spec.get("benchmark_shapes", [])
    small = [s for s in shapes if _axis_value(axis, s[0], s[1]) < split]
    large = [s for s in shapes if _axis_value(axis, s[0], s[1]) >= split]
    if axis == "numel" and spec.get("backward_structure") == "element_wise":
        return None  # single-regime op: the split is a placeholder, imbalance is fine
    if len(small) < min_per_side or len(large) < min_per_side:
        return (f"benchmark_shapes are not usable for training two specialists on axis `{axis}` "
                f"with regime_split={split:g}: only {len(small)} shape(s) below the split and "
                f"{len(large)} above (need >= {min_per_side} on EACH side). Sweep the `{axis}` axis "
                f"much more widely (log-spaced) and/or move regime_split so both sides are populated.")
    return None


@dataclass
class TaskSpecConfig:
    op: str
    forward: str                 # "module.path:fn_name" — a PyTorch forward
    bench_dir: Path
    api_base: str = "https://api.openai.com/v1"
    model: str = "gpt-5.5"
    api_key: str | None = None
    max_attempts: int = 4
    max_tokens: int = 16000
    temperature: float | None = 0.2
    timeout: int = 240
    perf_baseline: str | None = None   # auto-detect liger if None


# --------------------------------------------------------------------------- LLM layer

SYSTEM_MESSAGE = (
    "You are an expert at turning a PyTorch operator forward into a GPU-benchmark specification. "
    "You are asked ONLY for the semantic parts that cannot be derived mechanically: how to generate "
    "VALID random inputs (respecting each input's meaning — log-probabilities, probabilities that sum "
    "to 1, integer class labels in range, plain activations), which inputs are differentiable DATA vs "
    "non-differentiable targets/labels, which args are scalar config (eps, etc.), a concise statement "
    "of the forward and backward math, the BACKWARD's reduction structure (which decides the shape "
    "regime), and a set of benchmark shapes DESIGNED AROUND that regime axis. "
    "You reply with ONE JSON object and nothing else."
)


def _render_prompt(cfg: TaskSpecConfig, fwd_src: str, sig: str) -> str:
    return f"""Operator: `{cfg.op}`. Here is its PyTorch forward.

```python
{fwd_src}
```

Signature: `{sig}`

Return ONE JSON object with exactly these keys:

{{
  "tensor_args":  ["names of the forward args that are TENSORS, in signature order"],
  "config_args":  ["names of the forward args that are NON-tensor scalars/config (e.g. eps); [] if none"],
  "differentiable_args": ["subset of tensor_args that the backward computes gradients FOR"],
  "make_inputs_body": "python code producing the tensor args as VALID random inputs, in signature order",
  "task_context": "concise text: the forward math, then the backward math (what gradients, w.r.t. what)",
  "correctness_hint": "one-line formula summary",
  "backward_structure": "element_wise" | "row_reduction" | "cross_row_reduction" | "mixed",
  "regime_axis": "numel" | "cols" | "rows",
  "regime_reason": "one sentence: which reduction in the BACKWARD forces the kernel structure to change",
  "regime_split": <number on that axis: the rule-of-thumb small/large cut>,
  "benchmark_shapes": [[rows, cols], ...],
  "dtypes": ["float32", "float16", "bfloat16"]  // dtypes to SUPPORT (correctness/tolerances); perf timing itself runs bf16-only
}}

RULES:
- `differentiable_args` matters: a float tensor is NOT necessarily differentiated. E.g. a KL-divergence
  target `y_true` is float probabilities but is a TARGET (no gradient); a cross-entropy `target` is int64
  labels (no gradient). Only list args whose gradient the backward must return.
- `make_inputs_body` MUST respect each input's semantics, e.g.:
    * log-probability input  -> `torch_module.log_softmax(torch_module.randn(...), dim=-1)`
    * probability target     -> `torch_module.softmax(torch_module.randn(...), dim=-1)`
    * class labels           -> `torch_module.randint(0, cols, (rows,), dtype=torch_module.int64)`
    * plain activations      -> `torch_module.randn(...)`
  Use ONLY `torch_module` (not `torch`), device="cuda", and the case's `rows`/`cols`/`dtype`.
  Assign each tensor arg to a variable NAMED EXACTLY as in tensor_args. Do NOT return anything.
  Config args are bound by the harness from the forward's own default values — assign one yourself
  ONLY if the forward declares it with no default.

- REGIME (this decides how the two specialist kernels get trained, so get it right). Judge the
  BACKWARD only — a reduction in the FORWARD does not create a backward regime:
    * "element_wise"        = backward computes each output element with NO reduction
                              -> regime_axis = "numel"  (there is no structural knob; a single kernel wins
                                 at all shapes). Example: SwiGLU, GeGLU, and KL-divergence (its forward
                                 sums over the vocab, but its backward is just -y_true/BT — element-wise).
    * "row_reduction"       = each row reduces along the last dim (softmax, cross-entropy)
                              -> regime_axis = "cols"  (single-pass fused row vs streaming multi-block row)
    * "cross_row_reduction" = a gradient is summed ACROSS rows (e.g. dweight = sum over rows)
                              -> regime_axis = "rows"  (direct atomics vs split multi-group reduction)
    * "mixed"               = both; pick the axis of the CROSS-ROW reduction (it dominates the structure).

- `benchmark_shapes` MUST BE DESIGNED AROUND `regime_axis` — this is the whole point:
    * SWEEP the regime axis widely (log-spaced, small -> LLM scale) while holding the other dim at a
      few realistic values. E.g. axis=rows -> sweep rows 16,64,...,131072 at cols=1024 (+ a couple of
      wider-cols shapes); axis=cols -> sweep cols 512,...,128256 at rows=4096.
    * There MUST be at least 4 shapes on EACH side of `regime_split`, otherwise the small/large
      specialists cannot be trained. Choose `regime_split` so that holds.
    * Include 1-2 non-power-of-two shapes. Keep any single tensor under ~2GB in fp32.
    * ~12-16 shapes total.

- Output ONLY the JSON object.
"""


def _render_repair(cfg: TaskSpecConfig, prev: str, failure: str) -> str:
    return f"""Your previous specification for `{cfg.op}` FAILED the self-check.

The self-check builds a TRIVIAL autograd-pair implementation straight from the forward
(forward + PyTorch autograd as the backward) and runs it through the real evaluator. It must come
out 100% correct. It did not:

=== self-check output ===
{failure}

=== your previous JSON ===
{prev}

Likely causes: `make_inputs_body` produces invalid inputs for this op's semantics (e.g. feeding raw
randn where log-probabilities or normalized probabilities are required, or labels out of range);
`differentiable_args` lists an arg that has no gradient (a target/label) or omits one that does;
`tensor_args`/`config_args` misclassify an argument.

Return the corrected JSON object only."""


def _extract_json(text: str) -> dict:
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    raw = m.group(1) if m else text[text.find("{"): text.rfind("}") + 1]
    return json.loads(raw)


# --------------------------------------------------------------------------- code generation

_FORWARD_REF = '''"""PyTorch reference forward for {op} (used for AtenIR extraction and as the oracle)."""

import torch

{fwd_src}
'''


def _task_spec_source(cfg: TaskSpecConfig, spec: dict, out_shape_is_scalar: bool,
                      param_defaults: dict | None = None,
                      reduced_outputs: tuple[str, ...] = ()) -> str:
    """Assemble task_spec.py. Contract + oracles are CODE-derived templates, not LLM text."""
    op = cfg.op
    tensor_args = spec["tensor_args"]
    config_args = spec.get("config_args", [])
    diff_args = spec["differentiable_args"]
    out_names = tuple(f"d{a}" for a in diff_args)

    # regime: decided in the SAME LLM call as the shapes, so the shapes are designed around the axis
    backward_structure = spec.get("backward_structure", "unknown")
    _axis = spec.get("regime_axis", "numel")
    regime_expr = _AXIS_EXPR.get(_axis, "case.rows * case.cols")
    regime_reason = spec.get("regime_reason", "").replace("\n", " ")
    regime_split = int(float(spec.get("regime_split", 1_000_000)))

    # contract indices: make_inputs returns (cotangent, *tensor_args, *config_args)
    n_t = len(tensor_args)
    fwd_idx = tuple(range(1, 1 + n_t + len(config_args)))
    extra_idx = tuple(range(1 + n_t, 1 + n_t + len(config_args)))
    mem_idx = tuple(1 + tensor_args.index(a) for a in diff_args)

    all_args = tensor_args + config_args
    fwd_sig = ", ".join(all_args)
    diff_idx_in_tensors = [tensor_args.index(a) for a in diff_args]

    cot = "dout"
    api = (f"def {op}_forward_with_saved({fwd_sig}):\n    return out, saved_tensors\n\n"
           f"def {op}_backward_from_saved({cot}, saved_tensors"
           + ("".join(f", {c}" for c in config_args)) + "):\n    return "
           + ", ".join(out_names) + "\n")

    tol_lines = ",\n".join(
        f'    "{d}": ({_DTYPE_TOL[d][0]}, {_DTYPE_TOL[d][1]})' for d in spec.get("dtypes", list(_DTYPE_TOL))
    )
    shapes = ",\n".join(f"    ({r}, {c})" for r, c in spec["benchmark_shapes"])
    mk_body = "\n".join("    " + ln for ln in spec["make_inputs_body"].strip().splitlines())
    # Bind the non-tensor config args from the FORWARD'S OWN DEFAULTS. The LLM is only asked to
    # produce the tensor args, so relying on it to also assign `eps` is a contract it never agreed
    # to; emitting these after mk_body also makes the value deterministic if it assigned one anyway.
    defaults = param_defaults or {}
    cfg_lines = "\n".join(f"    {c} = {defaults[c]!r}" for c in config_args if c in defaults)
    if cfg_lines:
        mk_body = mk_body + "\n" + cfg_lines
    cot_expr = ("torch_module.rand((), device=\"cuda\", dtype=dtype) + 0.5" if out_shape_is_scalar
                else "torch_module.randn_like(_out_probe)")

    # When the perf baseline is Liger, the evaluator core and verify_liger_baseline both call
    # `task_spec.make_liger_autograd_pair_fns()`. It is CODE-derived (never LLM text), and it must be
    # emitted HERE: scaffold only writes it into the spliceable `regime_hooks.py`, which the
    # orchestrator deletes because task_spec already owns the regime hooks. Without this the Liger
    # gate dies with AttributeError and the repair loop blames the LLM for our own missing hook.
    liger_pair_hook = ""
    if _detect_baseline(cfg.op, cfg.perf_baseline)[0] == "liger":
        pkg = cfg.bench_dir.name
        liger_pair_hook = f'''

def make_liger_autograd_pair_fns():
    """(forward_with_saved, backward_from_saved) from Liger's hand-optimized kernel — the perf
    baseline when AUTOGRAD_PAIR_PERF_BASELINE=liger."""
    try:
        from benchmark.{pkg}.strong_baselines.liger_{op} import (
            make_liger_{op}_autograd_pair_fns,
        )
    except ImportError:
        from strong_baselines.liger_{op} import make_liger_{op}_autograd_pair_fns
    return make_liger_{op}_autograd_pair_fns()

'''

    return f'''"""Task spec for the {op} Triton backward benchmark.

AUTO-GENERATED by Phase-3 (synthesize_task_spec). The contract and the oracles are code-derived
templates; the input semantics / math text / shapes came from the spec synthesizer and were
validated by the self-check (a trivial forward+autograd pair must score 100% correct).
"""

from dataclasses import dataclass

try:
    from benchmark.triton_{op}_backward_bench.forward_ref import {op}_forward_ref
except ImportError:  # pragma: no cover
    from forward_ref import {op}_forward_ref


CANDIDATE_FN_NAME = "{op}_backward_triton"
OUTPUT_NAMES = {out_names!r}

AUTOGRAD_PAIR_FORWARD_FN_NAME = "{op}_forward_with_saved"
AUTOGRAD_PAIR_BACKWARD_FN_NAME = "{op}_backward_from_saved"
# make_inputs returns (cotangent, {", ".join(all_args)})
AUTOGRAD_PAIR_COTANGENT_INDEX = 0
AUTOGRAD_PAIR_FORWARD_INPUT_INDICES = {fwd_idx!r}
AUTOGRAD_PAIR_BACKWARD_EXTRA_INPUT_INDICES = {extra_idx!r}
AUTOGRAD_PAIR_MEMORY_INPUT_INDICES = {mem_idx!r}

AUTOGRAD_PAIR_API = """{api}"""

AUTOGRAD_PAIR_TASK_CONTEXT = """{spec["task_context"]}"""


@dataclass(frozen=True)
class TestCase:
    rows: int
    cols: int
    dtype_name: str
    atol_value: float
    rtol_value: float


_DTYPE_TOLERANCES = {{
{tol_lines}
}}

_BENCHMARK_SHAPES = [
{shapes}
]

CORRECTNESS_CASES = [
    TestCase(8, 64, "float32", 2e-5, 2e-5),
    TestCase(17, 128, "float32", 2e-5, 2e-5),
    TestCase(32, 256, "float16", 5e-2, 5e-2),
    TestCase(64, 512, "float16", 5e-2, 5e-2),
    TestCase(32, 256, "bfloat16", 8e-2, 8e-2),
    TestCase(64, 512, "bfloat16", 8e-2, 8e-2),
]

# Timing runs on bf16 only: these kernels are memory-bound, so fp16 duplicates bf16's byte
# width (same perf point twice) and fp32 is not a training dtype — their value is numeric
# correctness, which CORRECTNESS_CASES already covers on all dtypes. (TritonBench times
# bf16-only for the same reason.)
BENCHMARK_DTYPES = ["bfloat16"]


def _make_benchmark_cases() -> list[TestCase]:
    cases: list[TestCase] = []
    for rows, cols in _BENCHMARK_SHAPES:
        for dtype_name in BENCHMARK_DTYPES:
            a, r = _DTYPE_TOLERANCES.get(dtype_name, (8e-2, 8e-2))
            cases.append(TestCase(rows, cols, dtype_name, a, r))
    return cases


BENCHMARK_CASES = _make_benchmark_cases()


# --- Shape-regime metadata ---------------------------------------------------------------------
# Axis derived from the BACKWARD's reduction structure ({backward_structure}):
#   {regime_reason}
# The split only partitions the TRAINING suites; the deployment threshold is measured later from
# real data by shape_dispatch_report.
REGIME_SPLIT = {regime_split}


def regime_feature(case: "TestCase") -> float:
    return float({regime_expr})


def case_weight(case: "TestCase") -> float:
    import math

    dist = abs(math.log2(max(regime_feature(case), 1.0)) - math.log2(REGIME_SPLIT))
    return float(max(1.0, round(dist)))


SMALL_CASES = [c for c in BENCHMARK_CASES if regime_feature(c) < REGIME_SPLIT]
LARGE_CASES = [c for c in BENCHMARK_CASES if regime_feature(c) >= REGIME_SPLIT]

{liger_pair_hook}
def _dtype(torch_module, dtype_name: str):
    if dtype_name == "float32":
        return torch_module.float32
    if dtype_name == "float16":
        return torch_module.float16
    if dtype_name in ("bfloat16", "bf16"):
        return torch_module.bfloat16
    raise ValueError(f"Unsupported dtype: {{dtype_name}}")


def seed_for_case(case: TestCase) -> int:
    return case.rows * 100003 + case.cols


def case_metadata(case: TestCase):
    return {{"shape": [case.rows, case.cols], "dtype": case.dtype_name}}


def make_inputs(torch_module, case: TestCase):
    """Return (cotangent, {", ".join(all_args)})."""
    dtype = _dtype(torch_module, case.dtype_name)
    rows, cols = case.rows, case.cols
{mk_body}
    _out_probe = {op}_forward_ref({fwd_sig})
    dout = {cot_expr}
    return dout, {", ".join(all_args)}


def torch_oracle(torch_module, dout, {fwd_sig}):
    """Generic autograd oracle: clone -> requires_grad -> forward -> backward(dout) -> grads."""
    _tensors = [{", ".join(tensor_args)}]
    _ins = []
    for _i, _t in enumerate(_tensors):
        if _i in {diff_idx_in_tensors!r}:
            _ins.append(_t.detach().clone().float().requires_grad_(True))
        else:
            _ins.append(_t.detach().clone())
    _out = {op}_forward_ref(*_ins{"".join(f", {c}" for c in config_args)})
    _out.backward(dout.float() if dout.dim() == 0 else dout.float())
    _grads = tuple(_ins[_i].grad.to(_tensors[_i].dtype) for _i in {diff_idx_in_tensors!r})
    return _grads if len(_grads) > 1 else _grads[0]


def autograd_pair_forward_oracle(torch_module, {fwd_sig}):
    return {op}_forward_ref({fwd_sig})


# Gradients that are summed over EVERY row (probed from tensor shapes, not from LLM text).
_CROSS_ROW_REDUCED_OUTPUTS = {reduced_outputs!r}

# fp reduction-order noise vs the autograd oracle grows with the row count and dominates the
# near-zero entries of a cross-row-reduced gradient, so the tight per-element atol used for the
# forward and for the per-row grads is unreachable for it at 100k+ rows. Loosen atol for THOSE
# OUTPUTS ONLY; everything else keeps the strict tolerance.
_REDUCED_ATOL = {{"float32": 2e-3, "float16": 2e-1, "bfloat16": 2e-1}}


def atol(case: TestCase, output_name: str) -> float:
    if output_name in _CROSS_ROW_REDUCED_OUTPUTS:
        # Accumulation noise of a rows-long sum random-walks as sqrt(rows): two CORRECT
        # implementations that merely accumulate in a different order drift apart by
        # O(eps * sqrt(rows)). A constant atol therefore over-rejects at 100k+ rows —
        # scale it with sqrt(rows), anchored at 64 rows (where the base values were tuned).
        base = _REDUCED_ATOL.get(case.dtype_name, case.atol_value)
        return base * max(1.0, (case.rows / 64.0) ** 0.5)
    return case.atol_value


def rtol(case: TestCase, output_name: str) -> float:
    return case.rtol_value


def correctness_hint() -> str:
    return {spec["correctness_hint"]!r}
'''


# --------------------------------------------------------------------------- self-check

_TRIVIAL_PAIR = '''"""Trivial autograd-pair built straight from the forward — the Phase-3 self-check.

If this does NOT score 100% correct through the real evaluator, the generated task_spec's contract
or oracle is wrong (this implementation IS the reference semantics).
"""

import torch

from benchmark.triton_{op}_backward_bench.forward_ref import {op}_forward_ref

_DIFF_IDX = {diff_idx!r}


def {op}_forward_with_saved({fwd_sig}):
    out = {op}_forward_ref({fwd_sig})
    return out, ({saved_tuple})


def {op}_backward_from_saved(dout, saved_tensors{extra_sig}):
    _tensors = list(saved_tensors)
    _ins = []
    for _i, _t in enumerate(_tensors):
        if _i in _DIFF_IDX:
            _ins.append(_t.detach().clone().float().requires_grad_(True))
        else:
            _ins.append(_t.detach().clone())
    _out = {op}_forward_ref(*_ins{extra_call})
    _out.backward(dout.float())
    _grads = tuple(_ins[_i].grad.to(_tensors[_i].dtype) for _i in _DIFF_IDX)
    return _grads if len(_grads) > 1 else _grads[0]
'''


def _write_trivial_pair(cfg: TaskSpecConfig, spec: dict, path: Path) -> None:
    tensor_args = spec["tensor_args"]
    config_args = spec.get("config_args", [])
    diff_idx = [tensor_args.index(a) for a in spec["differentiable_args"]]
    fwd_sig = ", ".join(tensor_args + config_args)
    saved_tuple = ", ".join(tensor_args) + ("," if len(tensor_args) == 1 else "")
    extra_sig = "".join(f", {c}" for c in config_args)
    extra_call = "".join(f", {c}" for c in config_args)
    path.write_text(_TRIVIAL_PAIR.format(
        op=cfg.op, diff_idx=diff_idx, fwd_sig=fwd_sig, saved_tuple=saved_tuple,
        extra_sig=extra_sig, extra_call=extra_call), encoding="utf-8")


def _self_check(cfg: TaskSpecConfig, spec: dict) -> tuple[bool, str]:
    """Run the trivial forward+autograd pair through the REAL evaluator. Must be 100% correct."""
    work = cfg.bench_dir / ".taskspec_selfcheck"
    work.mkdir(parents=True, exist_ok=True)
    trivial = work / "trivial_pair.py"
    _write_trivial_pair(cfg, spec, trivial)

    for p in (str(REPO_ROOT), str(cfg.bench_dir)):
        if p not in sys.path:
            sys.path.insert(0, p)
    try:
        from benchmark.triton_backward_bench_common.autograd_pair_evaluator_core import (
            evaluate_autograd_pair_program,
        )
        ts_spec = importlib.util.spec_from_file_location(
            f"ts_{uuid.uuid4().hex}", str(cfg.bench_dir / "task_spec.py"))
        ts = importlib.util.module_from_spec(ts_spec)
        ts_spec.loader.exec_module(ts)
        report = evaluate_autograd_pair_program(str(trivial), ts)
    except Exception as e:  # contract wrong -> evaluator blows up
        import traceback
        return False, f"evaluator raised (contract likely wrong):\n{traceback.format_exc()[-1500:]}"

    # the evaluator returns an EvaluationResult (dataclass with .metrics/.artifacts), and older
    # paths may return a plain dict — accept both.
    if hasattr(report, "metrics"):
        metrics = dict(report.metrics or {})
        artifacts = getattr(report, "artifacts", None)
    elif isinstance(report, dict):
        metrics = dict(report.get("metrics", report) or {})
        artifacts = report.get("artifacts")
    else:
        return False, f"unexpected evaluator return type: {type(report)!r}"

    correct = float(metrics.get("correct", 0.0))
    if correct == 1.0:
        return True, "self-check PASSED (trivial forward+autograd pair is 100% correct)"
    interesting = {k: v for k, v in metrics.items() if "correct" in k or "error" in k}
    return False, (f"self-check FAILED: correct={correct}\n"
                   f"metrics={json.dumps(interesting, indent=1, default=str)}\n"
                   f"artifacts={str(artifacts)[:800]}")


# --------------------------------------------------------------------------- driver

def _resolve_forward(forward: str):
    """Load the user's forward from EITHER a file path OR a module path.

    Accepts, in order of preference:
      - `/path/to/forward.py:fn`   — a plain file path + function name
      - `/path/to/forward.py`      — a file with exactly one top-level function (auto-detected)
      - `package.module:fn`        — an importable module path (the original contract)

    File paths are the user-facing form (drop a .py file, no need to build an importable package);
    the module form is kept because the pipeline itself points `--forward` at the generated
    forward_ref.py during the seed stage.
    """
    path_part, _, attr_name = forward.partition(":")
    p = Path(path_part)
    if p.suffix == ".py" and p.is_file():
        spec = importlib.util.spec_from_file_location(f"user_fwd_{uuid.uuid4().hex}", str(p))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        if attr_name:
            return getattr(mod, attr_name)
        import inspect
        fns = [v for k, v in vars(mod).items()
               if inspect.isfunction(v) and v.__module__ == mod.__name__ and not k.startswith("_")]
        if len(fns) == 1:
            return fns[0]
        names = ", ".join(f.__name__ for f in fns) or "(none)"
        raise SystemExit(
            f"error: {p} has {len(fns)} top-level functions ({names}); name the one to use as "
            f"'{path_part}:<fn_name>'")
    # A path that LOOKS like a file but isn't one — usually a not-yet-created path or (on a cluster)
    # a file on a disk this node can't see (e.g. a login-node /tmp). Say that plainly instead of
    # letting importlib emit a cryptic ModuleNotFoundError.
    if path_part.endswith(".py") or "/" in path_part:
        raise SystemExit(
            f"error: forward file not found: {path_part} (cwd={Path.cwd()}). If this is a GPU-node "
            f"run, put the forward on a SHARED filesystem — a compute node cannot read the login "
            f"node's /tmp.")
    # fall back to module import path
    mod = importlib.import_module(path_part)
    return getattr(mod, attr_name)


def _load_forward_source(forward: str) -> tuple[str, str, str, dict]:
    """Return (source, signature, def_name, param_defaults).

    `def_name` is the function's own `__name__`, which is what the copied source actually defines.
    The attribute we were pointed at may be an ALIAS of it (`foo_ref = rmsnorm_forward_ref`), so
    aliasing against the attribute name would emit a reference to a name the source never defines.

    `param_defaults` lets CODE bind the non-tensor config args (eps, ...) from the forward's own
    defaults instead of hoping the LLM assigns them in make_inputs_body.
    """
    fn = _resolve_forward(forward)
    import inspect
    sig = inspect.signature(fn)
    defaults = {n: p.default for n, p in sig.parameters.items()
                if p.default is not inspect.Parameter.empty}
    return inspect.getsource(fn), f"{fn.__name__}{sig}", fn.__name__, defaults


def _detect_baseline(op: str, override: str | None) -> tuple[str, str]:
    if override:
        return override, ("Liger's hand-optimized" if override == "liger" else "PyTorch autograd's")
    try:
        importlib.import_module(f"liger_kernel.ops.{op}")
        return "liger", "Liger's hand-optimized"
    except Exception:
        return "pytorch_autograd", "PyTorch autograd's"


def synthesize_task_spec(cfg: TaskSpecConfig) -> int:
    cfg.bench_dir.mkdir(parents=True, exist_ok=True)
    (cfg.bench_dir / "__init__.py").touch(exist_ok=True)
    fwd_src, sig, def_name, param_defaults = _load_forward_source(cfg.forward)

    baseline, baseline_title = _detect_baseline(cfg.op, cfg.perf_baseline)
    print(f"[task_spec] perf baseline = {baseline} ({baseline_title})")

    # forward_ref.py (aliased to the canonical <op>_forward_ref). Alias against the DEF's own name:
    # that is the only name the copied source is guaranteed to bind.
    ref_src = fwd_src if def_name == f"{cfg.op}_forward_ref" else (
        fwd_src + f"\n\n{cfg.op}_forward_ref = {def_name}\n")
    (cfg.bench_dir / "forward_ref.py").write_text(
        _FORWARD_REF.format(op=cfg.op, fwd_src=ref_src), encoding="utf-8")

    work = cfg.bench_dir / ".taskspec_synth"
    work.mkdir(parents=True, exist_ok=True)

    prompt = _render_prompt(cfg, fwd_src, sig)
    prev_json = ""
    for attempt in range(1, cfg.max_attempts + 1):
        adir = work / f"attempt_{attempt:03d}"
        adir.mkdir(exist_ok=True)
        (adir / "prompt.md").write_text(prompt, encoding="utf-8")
        print(f"[task_spec] spec synthesis attempt {attempt}/{cfg.max_attempts}")
        resp = generate_with_openai_compatible_api(
            prompt=prompt, system_message=SYSTEM_MESSAGE, model=cfg.model,
            api_base=cfg.api_base, api_key=cfg.api_key, max_tokens=cfg.max_tokens,
            temperature=cfg.temperature, timeout=cfg.timeout)
        (adir / "response.txt").write_text(resp, encoding="utf-8")
        try:
            spec = _extract_json(resp)
        except Exception as e:
            prompt = _render_repair(cfg, resp[:2000], f"could not parse JSON: {e}")
            continue
        prev_json = json.dumps(spec, indent=1)
        (adir / "spec.json").write_text(prev_json, encoding="utf-8")

        # Shapes must be usable for training BOTH specialists on the chosen axis. Checking this
        # here (rather than discovering it at evolve time) is why the regime is decided in the SAME
        # call as the shapes.
        bal = _check_shape_balance(spec)
        if bal:
            print(f"[task_spec] shape/regime balance FAILED: {bal.splitlines()[0]}")
            prompt = _render_repair(cfg, prev_json, bal)
            continue

        # probe the forward once to learn the output shape (scalar loss vs same-shape output)
        out_scalar = _probe_output_is_scalar(cfg, spec, param_defaults)
        # ...and which grads are cross-row reductions, so their atol can be loosened (shape-derived)
        reduced = _probe_reduced_outputs(cfg, spec, param_defaults)
        if reduced:
            print(f"[task_spec] cross-row-reduced grads (loosened atol): {', '.join(reduced)}")
        (cfg.bench_dir / "task_spec.py").write_text(
            _task_spec_source(cfg, spec, out_scalar, param_defaults, reduced), encoding="utf-8")

        ok, msg = _self_check(cfg, spec)
        (adir / "selfcheck.txt").write_text(msg, encoding="utf-8")
        print(f"[task_spec] {msg.splitlines()[0]}")

        # Distinguish "the LLM's spec is wrong" (repairable) from "our own codegen is broken"
        # (NOT repairable — the LLM would be asked to fix a JSON that was never the problem).
        # A failure raised from forward_ref.py / the generated task_spec's own import block is ours.
        if not ok and _is_generator_bug(msg):
            print(f"[task_spec] ABORT: this failure is in GENERATED scaffolding, not in the spec — "
                  f"repairing the JSON cannot fix it. See {adir/'selfcheck.txt'}")
            return 2

        if ok:
            (cfg.bench_dir / ".taskspec_spec.json").write_text(prev_json, encoding="utf-8")
            axis = spec.get("regime_axis", "numel")
            print(f"[task_spec] wrote {cfg.bench_dir/'task_spec.py'} and forward_ref.py")
            print(f"[task_spec] regime: backward is {spec.get('backward_structure')} -> axis=`{axis}`, "
                  f"split={spec.get('regime_split')} ({spec.get('regime_reason','')})")
            return 0
        prompt = _render_repair(cfg, prev_json, msg)

    print(f"[task_spec] FAILED after {cfg.max_attempts} attempts (see {work})")
    return 1


_GENERATOR_BUG_MARKERS = (
    "forward_ref.py",          # the alias/import block we emit
    "NameError",               # a name our template referenced but never bound
    "IndentationError",
    "SyntaxError",             # our f-string template produced invalid Python
)


def _is_generator_bug(msg: str) -> bool:
    """True if the self-check failure comes from OUR generated scaffolding rather than the spec.

    The repair loop feeds failures back to the LLM as "your JSON is wrong". When the real fault is
    in the code WE generate around the JSON, that feedback is a lie and every attempt burns an LLM
    call chasing a problem that does not exist. Bail out loudly instead.
    """
    return any(m in msg for m in _GENERATOR_BUG_MARKERS)


def _probe_output_is_scalar(cfg: TaskSpecConfig, spec: dict, param_defaults: dict | None = None) -> bool:
    """Call the forward once on a tiny case to see whether it returns a scalar (loss) or a tensor."""
    import torch
    defaults = param_defaults or {}
    ns: dict = {"torch_module": torch, "rows": 8, "cols": 64,
                "dtype": torch.float32, "case": None}
    try:
        exec(spec["make_inputs_body"], {"torch_module": torch}, ns)
        fn = _resolve_forward(cfg.forward)
        args = [ns[a] for a in spec["tensor_args"]] + [
            defaults.get(c, ns.get(c)) for c in spec.get("config_args", [])]
        out = fn(*args)
        return out.dim() == 0
    except Exception as e:
        print(f"[task_spec] output probe failed ({e}); assuming non-scalar output")
        return False


def _probe_reduced_outputs(cfg: TaskSpecConfig, spec: dict,
                           param_defaults: dict | None = None) -> tuple[str, ...]:
    """Which gradients are CROSS-ROW REDUCTIONS — derived from tensor shapes, never from LLM text.

    A differentiable arg that is broadcast across rows (e.g. rmsnorm's `weight`, shape (cols,), vs
    an output of shape (rows, cols)) has a gradient that sums over EVERY row. Against the autograd
    oracle, fp reduction-order noise in that sum grows with the row count and dominates the
    near-zero entries, so the per-element atol used for the forward and for dx is physically
    unreachable for it at 100k+ rows — a correct kernel "fails" on arithmetic it cannot control.
    The handwritten cases all loosen atol for exactly these outputs; the generator must too, or the
    gate rejects correct wrappers and the repair loop blames the LLM for our tolerance.

    Mechanical test: grad(arg) accumulates `out.numel() / arg.numel()` terms. Reduced iff > 1.
    We do NOT ask the LLM which outputs are reductions — a spec that could loosen its own tolerance
    could hide a wrong answer behind a wide atol.
    """
    import torch
    defaults = param_defaults or {}
    ns: dict = {"torch_module": torch, "rows": 8, "cols": 64,
                "dtype": torch.float32, "case": None}
    try:
        exec(spec["make_inputs_body"], {"torch_module": torch}, ns)
        fn = _resolve_forward(cfg.forward)
        args = [ns[a] for a in spec["tensor_args"]] + [
            defaults.get(c, ns.get(c)) for c in spec.get("config_args", [])]
        out = fn(*args)
        reduced = []
        for a in spec["differentiable_args"]:
            t = ns[a]
            if not isinstance(t, torch.Tensor) or t.numel() == 0:
                continue
            if out.numel() // max(t.numel(), 1) > 1:
                reduced.append(f"d{a}")
        return tuple(reduced)
    except Exception as e:
        print(f"[task_spec] reduction probe failed ({e}); assuming no reduced outputs")
        return ()


def main(argv=None):
    ap = argparse.ArgumentParser(description="Phase-3 Part-1: synthesize task_spec + forward_ref")
    ap.add_argument("--op", required=True)
    ap.add_argument("--forward", required=True, help="module.path:fn_name (a PyTorch forward)")
    ap.add_argument("--bench-dir", default=None)
    ap.add_argument("--api-base", default="https://api.openai.com/v1")
    ap.add_argument("--model", default="gpt-5.5")
    ap.add_argument("--api-key", default=None)
    ap.add_argument("--max-attempts", type=int, default=4)
    ap.add_argument("--perf-baseline", default=None, choices=[None, "liger", "pytorch_autograd"])
    args = ap.parse_args(argv or sys.argv[1:])

    bench = Path(args.bench_dir) if args.bench_dir else (
        REPO_ROOT / "benchmark" / f"triton_{args.op}_backward_bench")
    return synthesize_task_spec(TaskSpecConfig(
        op=args.op, forward=args.forward, bench_dir=bench, api_base=args.api_base,
        model=args.model, api_key=args.api_key, max_attempts=args.max_attempts,
        perf_baseline=args.perf_baseline))


if __name__ == "__main__":
    raise SystemExit(main())
