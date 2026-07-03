"""Prompt templates for autograd-pair saved-tensor fusion synthesis."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class OperatorSpec:
    """Everything that varies between operators in the autograd-pair pipeline.

    Pass an instance to all render_* functions so the prompts describe the
    right function names, tensor shapes, and semantics for the target benchmark.
    """

    # Public function signatures the LLM must implement.
    forward_fn_name: str       # e.g. "layernorm_forward_with_saved"
    forward_args: str          # e.g. "x, weight, bias, eps=1e-5"
    backward_fn_name: str      # e.g. "layernorm_backward_from_saved"
    backward_args: str         # e.g. "dy, saved_tensors, eps=1e-5"
    backward_returns: str      # e.g. "dx, dweight, dbias"

    # One-liner each: what the forward computes, what the backward returns.
    # Shown verbatim in the "Hard constraints" block.
    forward_semantics: str
    backward_semantics: str

    # Inputs with no gradient. The AtenIR graph may include their grads as
    # outputs; tell the LLM to discard them.
    no_grad_inputs: tuple[str, ...] = ()

    # Optional freeform text appended after the Triton pitfalls block.
    extra_constraints: str = ""


LAYERNORM_SPEC = OperatorSpec(
    forward_fn_name="layernorm_forward_with_saved",
    forward_args="x, weight, bias, eps=1e-5",
    backward_fn_name="layernorm_backward_from_saved",
    backward_args="dy, saved_tensors, eps=1e-5",
    backward_returns="dx, dweight, dbias",
    forward_semantics=(
        "Do not call PyTorch autograd or PyTorch reference LayerNorm in the generated math. "
        "Forward must produce the same `y` as row-wise LayerNorm over the last dimension."
    ),
    backward_semantics=(
        "Backward must consume only `dy`, `saved_tensors`, and `eps`. "
        "Return `dx` with `x` dtype, `dweight` with `weight` dtype, and `dbias` with `bias` dtype."
    ),
)


def load_op_spec(path: str | Path) -> OperatorSpec:
    """Load an :class:`OperatorSpec` from a JSON file.

    Shared by all three autograd-pair pipelines (A's fusion agent, B's
    handwritten dispatch seed wrapper, and C's no-AtenIR ablation) so a single
    ``<op>_spec.json`` describes one operator everywhere.
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return OperatorSpec(
        forward_fn_name=data["forward_fn_name"],
        forward_args=data["forward_args"],
        backward_fn_name=data["backward_fn_name"],
        backward_args=data["backward_args"],
        backward_returns=data["backward_returns"],
        forward_semantics=data["forward_semantics"],
        backward_semantics=data["backward_semantics"],
        no_grad_inputs=tuple(data.get("no_grad_inputs", [])),
        extra_constraints=data.get("extra_constraints", ""),
    )


# ── Signature derivation helpers ────────────────────────────────────────────
# These parse the (string) signatures on an OperatorSpec into structured pieces
# that the deterministic Pipeline B wrapper and the Pipeline C prompts both
# need.  They rely on two conventions the autograd-pair specs already follow:
#   * the only keyword/default arg is ``eps`` (everything else is a tensor input);
#   * each gradient return is ``"d" + <forward input name>`` (dx<-x, dweight<-weight,
#     dbias<-bias, da<-a, db<-b, dlinear_weight<-linear_weight, ...).


def _arg_names(arg_str: str) -> list[str]:
    return [a.split("=", 1)[0].strip() for a in arg_str.split(",") if a.strip()]


def forward_input_names(spec: OperatorSpec) -> list[str]:
    """Ordered tensor inputs of the forward (eps excluded), matching the forward
    reference signature and therefore the AtenIR graph's ``placeholders[1:]``."""
    return [a for a in _arg_names(spec.forward_args) if a != "eps"]


def forward_has_eps(spec: OperatorSpec) -> bool:
    return "eps" in _arg_names(spec.forward_args)


def grad_output_name(spec: OperatorSpec) -> str:
    """Name of the upstream gradient (e.g. ``dy``/``dc``/``do``/``dout``); the
    first positional arg of the backward, matching the graph's ``placeholders[0]``."""
    return _arg_names(spec.backward_args)[0]


def backward_return_names(spec: OperatorSpec) -> list[str]:
    return [r.strip() for r in spec.backward_returns.split(",") if r.strip()]


def grad_reorder(spec: OperatorSpec) -> list[int]:
    """Permutation mapping ``run_graph_program`` outputs (in forward-input order)
    to ``backward_returns`` order.

    ``run_graph`` returns gradients in forward-input order (it differentiates the
    forward inputs in signature order); the evaluator expects them in
    ``backward_returns`` order.  For most ops these coincide (identity), but fused
    ops such as LayerNorm->Linear list ``dlinear_weight`` before ``dweight``.
    """
    inputs = forward_input_names(spec)
    reorder: list[int] = []
    for ret in backward_return_names(spec):
        src = ret[1:] if ret.startswith("d") else ret
        if src not in inputs and src.startswith("_"):
            # "d_pair_bias" style: separator underscore after the leading d.
            src = src[1:]
        if src not in inputs:
            raise ValueError(
                f"backward return {ret!r} does not map to a forward input by the "
                f"'d'+name convention (forward inputs: {inputs}); add an explicit "
                "mapping for this operator."
            )
        reorder.append(inputs.index(src))
    return reorder


def render_dispatch_autograd_pair_wrapper(forward: str, spec: OperatorSpec) -> str:
    """Render Pipeline B's autograd-pair seed wrapper for an arbitrary operator.

    Pure string construction (no torch/triton), so it can be unit-tested without
    a GPU.  ``run_graph_program`` (emitted by the dispatch codegen) returns
    gradients in forward-input order; the evaluator expects ``backward_returns``
    order, so the reorder from :func:`grad_reorder` is applied here.
    """
    inputs = forward_input_names(spec)
    has_eps = forward_has_eps(spec)
    grad_out = grad_output_name(spec)
    reorder = grad_reorder(spec)

    fwd_call = ", ".join(inputs + (["eps"] if has_eps else []))
    saved = ", ".join(f"{n}.contiguous()" for n in inputs)
    saved_tuple = f"({saved},)" if len(inputs) == 1 else f"({saved})"
    unpack = f"{', '.join(inputs)} = saved_tensors[:{len(inputs)}]"
    run_args = ",\n        ".join(
        [f"{grad_out}.contiguous()"] + [f"{n}.contiguous()" for n in inputs]
    )
    eps_marker = "    _ = eps\n" if has_eps else ""

    if reorder == list(range(len(reorder))):
        backward_body = (
            f"{eps_marker}"
            f"    {unpack}\n"
            f"    return run_graph_program(\n        {run_args},\n    )"
        )
    else:
        ret = ", ".join(f"_grads[{i}]" for i in reorder)
        backward_body = (
            f"{eps_marker}"
            f"    {unpack}\n"
            f"    _grads = run_graph_program(\n        {run_args},\n    )\n"
            f"    return ({ret})"
        )

    backward_call = ", ".join([grad_out, "saved_tensors"] + (["eps"] if has_eps else []))

    return f'''

_FORWARD_SPEC = {forward!r}


def _load_forward_callable():
    module_name, fn_name = (
        _FORWARD_SPEC.split(":", 1)
        if ":" in _FORWARD_SPEC
        else _FORWARD_SPEC.rsplit(".", 1)
    )
    module = __import__(module_name, fromlist=[fn_name])
    return getattr(module, fn_name)


def _forward_with_saved_impl({spec.forward_args}):
    # Conservative seed: only save original forward inputs. OpenEvolve may
    # replace this with saved intermediates.
    y = _load_forward_callable()({fwd_call})
    return y, {saved_tuple}


def _backward_from_saved_impl({spec.backward_args}):
{backward_body}


def {spec.forward_fn_name}({spec.forward_args}):
    return _forward_with_saved_impl({fwd_call})


def {spec.backward_fn_name}({spec.backward_args}):
    return _backward_from_saved_impl({backward_call})
'''


SYSTEM_MESSAGE = """You are a Triton compiler engineer.
You synthesize a forward/backward autograd pair.  The forward may save tensors
that the backward reuses.  Correctness is required; efficiency should balance
backward latency, forward+backward latency, and saved-tensor memory."""

_SAVED_TENSOR_GUIDANCE = """\
Saved-tensor guidance:
- The saved tensor tuple is part of the evolvable program state.  The initial
  seed may save only original inputs; OpenEvolve may add, remove, or reorder
  saved tensors as long as the forward and backward agree.
- You may save forward intermediates if doing so improves the forward+backward
  tradeoff, but the prompt does not prescribe which intermediates to save.
- Prefer compact saved state such as small per-row/per-block statistics over
  full activation-sized tensors when the backward can cheaply reconstruct the
  larger intermediate.
- Avoid saving tensors with the same shape as a large activation unless the
  latency benefit clearly outweighs the memory cost.
- Do not save excessive large intermediates unless they clearly improve
  forward+backward latency.  The evaluator reports saved memory.
- It is acceptable to save original inputs if the backward needs them."""

_TRITON_PITFALLS = """\
Triton pitfalls:
- `tl.arange` bounds must be compile-time constants. Use `BLOCK_*: tl.constexpr`.
- Do not read Python globals inside `@triton.jit`; pass dimensions, strides, and
  scalar constants as arguments or meta-parameters.
- Use fp32 accumulation for reductions.
- Avoid global atomic contention when a partial-buffer reduction is better."""


def render_pair_rules(spec: OperatorSpec) -> str:
    no_grad_lines = ""
    if spec.no_grad_inputs:
        names = ", ".join(f"`{n}`" for n in spec.no_grad_inputs)
        no_grad_lines = (
            f"- The following inputs carry NO gradient and must not appear in the "
            f"backward output: {names}. "
            f"The AtenIR graph may include their gradients as outputs — discard them.\n"
        )
    extra = f"\n{spec.extra_constraints}" if spec.extra_constraints else ""
    return f"""\
## Autograd-pair rules

Public API to implement:

```python
def {spec.forward_fn_name}({spec.forward_args}):
    return y, saved_tensors

def {spec.backward_fn_name}({spec.backward_args}):
    return {spec.backward_returns}
```

Hard constraints:
- Return only Python source, no Markdown.
- Include imports for `torch`, `triton`, and `triton.language as tl`.
- Include an `EVOLVE-BLOCK` around generated Triton kernels and launch helpers.
- `saved_tensors` must be a tensor or tuple/list of tensors, because the evaluator
  stores them via `ctx.save_for_backward`.
- {spec.forward_semantics}
- {spec.backward_semantics}
{no_grad_lines}
{_SAVED_TENSOR_GUIDANCE}

{_TRITON_PITFALLS}
{extra}"""


def render_plan_prompt(
    *,
    forward: str,
    graph_summary: str,
    lowering_context: str = "",
    spec: OperatorSpec = LAYERNORM_SPEC,
) -> str:
    lowering_section = (
        f"\nAdditional lowering context:\n\n```text\n{lowering_context}\n```\n"
        if lowering_context else ""
    )
    return f"""# Autograd-Pair Planning

Forward reference:

```text
{forward}
```

The AtenIR backward graph below describes the reference backward semantics, but
the generated implementation is allowed to change the forward/backward contract
by saving forward intermediates.

{graph_summary}
{lowering_section}

Return Markdown with:

1. Initial saved tensor contract and which parts should remain evolvable.
2. Triton kernels for forward and backward.
3. Backward formula and reduction strategy.
4. Expected memory overhead of saved tensors and why it is worth the latency tradeoff.
"""


def render_codegen_prompt(
    *,
    graph_summary: str,
    pair_plan: str,
    lowering_context: str = "",
    spec: OperatorSpec = LAYERNORM_SPEC,
) -> str:
    lowering_section = (
        f"\nAdditional lowering context:\n\n```text\n{lowering_context}\n```\n"
        if lowering_context else ""
    )
    return f"""# Autograd-Pair Codegen

Generate a complete Python module for an autograd pair for the provided forward reference.

{render_pair_rules(spec)}

## Plan

```markdown
{pair_plan}
```

## AtenIR backward graph summary

{graph_summary}
{lowering_section}"""


def render_repair_prompt(
    *,
    graph_summary: str,
    pair_plan: str,
    previous_code: str,
    verifier_report: str,
    lowering_context: str = "",
    spec: OperatorSpec = LAYERNORM_SPEC,
) -> str:
    lowering_section = (
        f"\nAdditional lowering context:\n\n```text\n{lowering_context}\n```\n"
        if lowering_context else ""
    )
    return f"""# Autograd-Pair Repair

The generated autograd-pair program failed verification.

Verifier report:

```json
{verifier_report}
```

{render_pair_rules(spec)}

## Plan

```markdown
{pair_plan}
```

## AtenIR backward graph summary

{graph_summary}
{lowering_section}

## Previous code

```python
{previous_code}
```

Return only the repaired Python source. No Markdown.
"""
