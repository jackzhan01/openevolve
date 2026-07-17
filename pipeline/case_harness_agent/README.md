# `autodiff` — one PyTorch forward → a dispatched Triton forward+backward + a benchmark report

`autodiff` is the product interface over this repo's kernel-synthesis pipeline. You hand it a
single PyTorch forward — an ordinary `def f(...)` built from torch ops, in a `.py` file — and it
returns a shape-dispatched fused forward+backward kernel and a performance report measured against a
strong baseline (Liger where one exists, else PyTorch autograd), printing every stage and its live
progress as it runs.

`--forward` accepts `path/to/file.py` (auto-detects the single top-level function),
`path/to/file.py:fn_name` (when the file has several), or an importable `module.path:fn`. The
examples below point at an existing forward in the repo so they run as-is — for your own operator,
just drop a `.py` file with the forward function and point `--forward` at it.

From the Python API the forward can also be **the function object itself** — it is snapshotted to
`<bench_dir>/user_forward.py` at the entry (the record of exactly which forward was run) and the
file-based pipeline runs unchanged. The callable must be a self-contained named `def` over torch
ops: lambdas, closures, and globals beyond `math`/`torch`/`F` are rejected up front with a message
saying to use the file form instead.

```python
from pipeline.case_harness_agent.autodiff import autodiff

# file form
result = autodiff(
    "benchmark/triton_layernorm_backward_bench/forward_ref.py:layernorm_forward_ref",
    op="layernorm", bench_dir="benchmark/triton_layernorm_auto_backward_bench",
)

# callable form — same contract, minus creating a file yourself
def my_layernorm(x, weight, bias, eps=1e-5):
    mean = x.mean(dim=-1, keepdim=True)
    var = ((x - mean) ** 2).mean(dim=-1, keepdim=True)
    return (x - mean) / torch.sqrt(var + eps) * weight + bias

result = autodiff(my_layernorm, op="layernorm", bench_dir="benchmark/triton_layernorm_auto_backward_bench")
print(result.program)   # <op>_final_dispatched.py  — the deployed fw+bwd
print(result.report)    # RESULTS_<op>_vs_<baseline>.md
print(result.metrics)   # <op>_dispatch_report.json (measured baseline + geomeans)
print(result.baseline)  # "liger" | "pytorch_autograd"
```

Shell form (needs a GPU and `OPENAI_API_KEY`):

```bash
python -m pipeline.case_harness_agent.autodiff \
  --forward benchmark/triton_layernorm_backward_bench/forward_ref.py:layernorm_forward_ref \
  --op layernorm --bench-dir benchmark/triton_layernorm_auto_backward_bench
```

## What it does — the stages

```
taskspec → seed → evolve → dispatch → report
```

| stage | what happens | truth is checked by |
| --- | --- | --- |
| **taskspec** | one LLM call turns the forward into a benchmark spec (input semantics, differentiable args, math, the backward's reduction axis → shape regime, benchmark shapes). Contract + oracles + tolerances are **code-derived, never trusted from the LLM**. | self-check: a trivial `forward + autograd` pair must score `correct == 1.0` in the real evaluator |
| **seed** | the fusion agent writes the first *correct* fused backward kernel | correctness vs the code-derived oracle |
| **evolve** | OpenEvolve evolves three groups: `full` (generalist), `small` / `large` (regime specialists) | the evaluator's correctness gate before any timing |
| **dispatch** | measure every program on the whole shape grid, pick the real regime threshold, emit the dispatched program | real measurement |
| **report** | render `RESULTS_<op>_vs_<baseline>.md` | — |

Between taskspec and seed the harness auto-scaffolds the evaluators/configs. Only when the baseline
is Liger does it also synthesize `strong_baselines/liger_<op>.py` (zero-shot), gated by
`verify_liger_baseline.py` (wrapped forward + every gradient vs the PyTorch oracle); the general path
compares against PyTorch autograd and needs no wrapper.

The one invariant that lets this be automated: **every gate's notion of "correct" is independent
of the LLM call that wrote the code.** taskspec's gate is a code-derived oracle; the wrapper's gate
is a PyTorch oracle; dispatch is real measurement. Never let the model both write an answer and
define the standard it's judged by.

## Where the code lives

Product (the `autodiff` chain):

```
pipeline/case_harness_agent/
  autodiff.py                 # the product entry — file-in / file-out, returns artifact paths
  orchestrate.py              # the six-stage driver (stage banners, streaming _run, evolve modes)
  synthesize_task_spec.py     # taskspec: forward → spec + code-derived contract/oracles/tolerances
  scaffold.py                 # generates evaluators + configs for a case
  synthesize_liger_wrapper.py # zero-shot Liger baseline wrapper + closed-loop gate (Liger baseline only)
pipeline/autograd_pair_fusion_agent/   # seed: the fusion agent that writes the first kernel
pipeline/shared/llm_client.py          # the shared OpenAI-compatible client
openevolve-run.py                      # evolve
benchmark/triton_backward_bench_common/
  autograd_pair_evaluator_core.py      # the correctness+timing evaluator (the real gate)
  verify_liger_baseline.py             # the Liger-wrapper gate
  shape_dispatch_report.py             # dispatch: grid measurement → threshold → program
```

The other `pipeline/*/` packages and top-level `pipeline/run_*.py` scripts are earlier experiments
(handwritten dispatch, no-atenir fusion, lowering agents, …) and are **not** on the `autodiff` path.

## Input and output are files

- **Input** `--forward` is a path to a `.py` file. `path.py` auto-detects the single top-level
  function; `path.py:fn` names one when the file has several. (A `module.path:fn` import path also
  still works — the pipeline uses it internally.)
- **Output** is written under the bench dir (default `benchmark/triton_<op>_backward_bench`, or
  `--bench-dir`): `<op>_final_dispatched.py`, `RESULTS_<op>_vs_<baseline>.md`,
  `<op>_dispatch_report.json`.

`op` names the operator (used for Liger detection and the report title); `bench_dir` is just where
output goes. They may differ — e.g. `--op layernorm --bench-dir …/triton_layernorm_auto_backward_bench`
to avoid clobbering a hand-written case.

## Baseline: never silently downgraded

`--perf-baseline {auto,liger,pytorch_autograd}` (default `auto`). `auto` probes for a Liger kernel
and falls back to PyTorch autograd. **`liger` HARD-FAILS if Liger can't be resolved** rather than
quietly measuring against the far weaker autograd baseline. Liger's module name often differs from
`op` (op `rmsnorm` is Liger's `rms_norm`, `layernorm` is `layer_norm`), so pass the real sources
with `--liger-source` (repeatable). The only trustworthy record of what was measured against is the
`"baseline"` field in `<op>_dispatch_report.json`.

## GPUs: `--gpus 1` or `--gpus 3`

- `--gpus 1` (default): the three evolve groups run **sequentially** on one GPU. Portable — any
  machine with a single GPU can run it.
- `--gpus 3`: the three groups run **in parallel**, one per GPU (`CUDA_VISIBLE_DEVICES=0,1,2`) on a
  single node with ≥3 visible GPUs. Same three-way parallelism as three nodes, but no cross-node
  scheduling — which fits `autodiff` being one process.

**The pipeline never allocates nodes / calls `srun`.** It assumes it is already on a GPU with the
env and `OPENAI_API_KEY` ready. Getting onto a node and setting that up is the caller's environment
concern (whatever scheduler they use), deliberately kept out of the pipeline so it stays portable.

## Watching progress

`orchestrate` prints a timestamped banner per stage and streams each sub-stage's output live. Evolve
per-iteration progress (`Iteration N … completed in Xs`) streams from OpenEvolve's console handler;
in `--gpus 3` mode a single status line shows all three groups' current iteration plus current/best
score, updated in place (carriage-return) as they advance.
