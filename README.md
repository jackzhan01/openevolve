# AtenIR-to-Triton OpenEvolve Fork

This fork uses OpenEvolve as the search engine for Triton backward kernels, but
its main research code is organized around AtenIR graph extraction, agentic
Triton generation, unified correctness testing, and benchmark evaluation.

The current codebase has four main pieces:

```text
benchmark/                         # Benchmark tasks and reusable evaluator code
pipeline/                          # AtenIR-to-Triton synthesis pipelines
atenir/                            # AtenIR extraction, graph execution, primitive dispatch
tests/atenir_correctness/          # Unified correctness harness and regression tests
```

## `autodiff` — the one-call interface

The whole pipeline is packaged behind a single interface: give it **one PyTorch forward** — a path
to a `.py` file, or **the function object itself** — and it returns a shape-dispatched fused
**forward+backward** kernel and a **performance report** vs a strong baseline (PyTorch autograd, or
Liger for the ops it covers), printing each stage's live progress as it runs. It assumes it is
already on a GPU — allocating the node / setting up the env is the caller's concern, never the
pipeline's.

```bash
# needs a GPU and OPENAI_API_KEY; here the forward is an existing reference file in the repo
python -m pipeline.case_harness_agent.autodiff \
  --forward benchmark/triton_layernorm_backward_bench/forward_ref.py:layernorm_forward_ref \
  --op layernorm --bench-dir benchmark/triton_layernorm_auto_backward_bench \
  --iterations 10 --gpus 3
```

From Python, the forward can be passed as a plain callable — it is snapshotted to
`<bench_dir>/user_forward.py` at the entry and the file-based pipeline runs unchanged:

```python
from pipeline.case_harness_agent.autodiff import autodiff

def my_layernorm(x, weight, bias, eps=1e-5):
    mean = x.mean(dim=-1, keepdim=True)
    var = ((x - mean) ** 2).mean(dim=-1, keepdim=True)
    return (x - mean) / torch.sqrt(var + eps) * weight + bias

result = autodiff(my_layernorm, op="layernorm", iterations=10)
```

From the forward alone it runs five stages, with every gate's notion of "correct" kept independent
of the LLM that wrote the code:

- **taskspec** — one LLM call turns the forward into a benchmark spec (input semantics, which args
  are differentiable, the backward's reduction axis → shape regime, benchmark shapes); the contract
  and correctness oracles are derived in code, not by the LLM.
- **seed** — the fusion agent writes the first *correct* fused backward kernel.
- **evolve** — OpenEvolve evolves a generalist plus small/large shape-regime specialists,
  sequentially on one GPU (`--gpus 1`) or one group per GPU (`--gpus 3`).
- **dispatch** — measures every program across the full shape grid, picks the regime threshold, and
  emits the deployed forward+backward.
- **report** — renders `RESULTS_<op>_vs_<baseline>.md`.

Full interface docs — file map, baseline handling, `--gpus 1|3`, live progress — are in
[`pipeline/case_harness_agent/README.md`](pipeline/case_harness_agent/README.md).

## Liger benchmark suite

`benchmark/liger_suite/` builds benchmark cases for [Liger](https://github.com/linkedin/Liger-Kernel)
operators **automatically**: each op needs only a hand-written naive PyTorch forward
(`liger_suite/forwards/*.py`, ~10 lines each, semantics pinned against Liger's source), and the
suite driver runs the pipeline's `taskspec + prep` stages to produce everything else — the
code-derived task spec and oracles, the evaluator scaffolding, and a **gated** Liger baseline
wrapper (verified against the PyTorch oracle on every forward output and gradient).

```bash
# needs a GPU and OPENAI_API_KEY; one op failing does not block the rest
python -m benchmark.liger_suite.run_suite            # all manifest ops
python -m benchmark.liger_suite.run_suite --only dyt # a subset
```

Current coverage: 7 auto-built cases (`dyt`, `relu_squared`, `sparsemax`, `tvd`, `jsd`,
`poly_norm`, `fused_add_rms_norm`) on top of the 7 hand-written ones (swiglu, geglu, rmsnorm,
layernorm, softmax, cross_entropy, kl_div) — 14 Liger ops total. Ops whose inputs do not fit the
current 2D rows×cols case contract (rope / attention / fused_linear families) are out of scope
until the case schema is generalized.

Every auto-synthesized baseline wrapper was manually reviewed for benchmark fairness (no
host-device syncs, no extra tensor passes in the timed regions, saved-tensor sets matching
Liger's own `autograd.Function`) — verdicts and the known semantic caveats are recorded in
[`benchmark/liger_suite/WRAPPER_REVIEW.md`](benchmark/liger_suite/WRAPPER_REVIEW.md).

## Benchmark

`benchmark/` is a suite of Triton **backward-kernel benchmark cases** built around
[Liger](https://github.com/linkedin/Liger-Kernel) operators — currently 14 ops (swiglu, geglu,
rmsnorm, layernorm, softmax, cross_entropy, kl_div, dyt, relu_squared, sparsemax, tvd, jsd,
poly_norm, fused_add_rms_norm), with the goal of covering all Liger operators. Every case pairs a
PyTorch ground truth with a fairness-reviewed Liger baseline, so a candidate kernel can be checked
for correctness and timed against a strong target with one command.

Anatomy of a case (`benchmark/triton_dyt_backward_bench/` as the example):

```text
forward_ref.py                  # the PyTorch reference forward — the case's ground truth
task_spec.py                    # the contract: shapes/dtypes/tolerances, correctness oracles,
                                #   benchmark suites (full + small/large shape regimes)
evaluator_autograd_pair*.py     # evaluators: hard correctness gate first, then timing vs the
config_autograd_pair*.yaml      #   baseline; one per scoring mode, with its OpenEvolve config
strong_baselines/liger_dyt.py   # the gated Liger baseline (verified against the PyTorch oracle
                                #   on every forward output and gradient)
```

A candidate program exposes the pair API named in `task_spec.py` —
`<op>_forward_with_saved(*inputs) -> (y, saved)` and
`<op>_backward_from_saved(dout, saved, *extras) -> grads` — and is evaluated with:

```bash
# hard correctness gate, then benchmark timing vs the Liger baseline (needs a GPU)
python benchmark/triton_dyt_backward_bench/evaluator_autograd_pair_speed_memory_min_liger.py \
  path/to/candidate_program.py
```

or end-to-end (seed + evolve + dispatch + report) through `autodiff` with `--forward` pointed at
the case's `forward_ref.py`. The shared gate/timing machinery lives in
`benchmark/triton_backward_bench_common/`; baseline fairness verdicts and known semantic caveats
are recorded in [`benchmark/liger_suite/WRAPPER_REVIEW.md`](benchmark/liger_suite/WRAPPER_REVIEW.md).

## Pipelines

`pipeline/` contains the three AtenIR-to-Triton synthesis paths:

- `pipeline/fusion_agent/`: Pipeline A, direct AtenIR graph fusion into a Triton backward seed.
- `pipeline/primitive_atenir_lowering_agent/`: Pipeline B, per-op AtenIR lowering into verified Triton kernels.
- `pipeline/kernel_fusion_agent/`: Pipeline C, kernel-aware fusion using verified per-op lowering context.

Stable top-level entry points are provided so callers do not need to import the
internal package layout directly:

```bash
python -m pipeline.run_fusion_agent --help
python -m pipeline.run_lowering_agent --help
python -m pipeline.run_kernel_fusion_agent --help
python -m pipeline.run_layernorm_pipeline_comparison --help
```

The comparison runner executes the three LayerNorm pipelines under a shared
model/budget, evaluates the generated seeds with the benchmark evaluator, and can
optionally launch OpenEvolve for each seed:

```bash
python -m pipeline.run_layernorm_pipeline_comparison   --output-dir ~/tmp/layernorm_pipeline_comparison   --model gpt-5.5   --lowering-model gpt-4o   --reuse-existing-seeds
```

## Unified Tester And Evaluator

There are two related correctness layers:

- `tests/atenir_correctness/run_correctness.py` is a generic backward-kernel
  correctness runner. It compares generated backward functions against PyTorch
  autograd over a forward reference.
- `benchmark/triton_backward_bench_common/evaluator_core.py` is the benchmark
  evaluator core used by OpenEvolve. It imports a candidate program, checks the
  public API, runs correctness cases, and only benchmarks candidates that pass.

LayerNorm examples use the same semantic contract: a candidate must expose
`layernorm_backward_triton(dy, x, weight, bias, eps)` and return
`(dx, dweight, dbias)`.

## AtenIR

`atenir/` is the reusable graph layer:

- `atenir.extract` extracts autograd graphs from PyTorch forward functions.
- `atenir.compose` executes serialized AtenIR graphs.
- `atenir.primitive_triton` provides primitive Triton dispatch used by the
  lowering and correctness workflows.

The LayerNorm benchmark also contains a frozen AtenIR graph and primitive-kernel
artifacts under `benchmark/triton_layernorm_backward_bench/atenir/` as the first
worked benchmark instance.

## OpenEvolve Base

The original OpenEvolve framework remains in `openevolve/` and is used for code
evolution, population management, LLM calls, and evaluator orchestration. This
fork keeps that engine, but the top-level workflow is centered on AtenIR-based
Triton backward generation and verification.
