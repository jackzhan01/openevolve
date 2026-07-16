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

The whole pipeline is packaged behind a single interface: give it **one PyTorch forward in a file**
and it returns a shape-dispatched fused **forward+backward** kernel and a **performance report** vs a
strong baseline (PyTorch autograd, or Liger for the ops it covers), printing each stage's live
progress as it runs. It assumes it is already on a GPU — allocating the node / setting up the env is
the caller's concern, never the pipeline's.

```bash
# needs a GPU and OPENAI_API_KEY; here the forward is an existing reference file in the repo
python -m pipeline.case_harness_agent.autodiff \
  --forward benchmark/triton_layernorm_backward_bench/forward_ref.py:layernorm_forward_ref \
  --op layernorm --bench-dir benchmark/triton_layernorm_auto_backward_bench \
  --iterations 10 --gpus 3
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

## Benchmark

`benchmark/` contains public tasks and shared benchmark infrastructure. The
first benchmark is LayerNorm backward:

```text
benchmark/triton_layernorm_backward_bench/
benchmark/triton_backward_bench_common/
```

The LayerNorm benchmark defines the forward reference, PyTorch-autograd oracle,
Triton baseline, OpenEvolve config, and evaluator wrapper. The shared common
package contains `evaluator_core.py`, which is the hard correctness gate used by
OpenEvolve before timing any generated candidate.

Useful commands:

```bash
python benchmark/triton_layernorm_backward_bench/test_correctness.py
python benchmark/triton_layernorm_backward_bench/evaluator.py   benchmark/triton_layernorm_backward_bench/initial_program.py

python -m benchmark.triton_backward_bench_common.builder_cli emit-spec   benchmark/triton_layernorm_backward_bench
```

To run OpenEvolve on the LayerNorm seed:

```bash
python -m openevolve.cli   benchmark/triton_layernorm_backward_bench/initial_program.py   benchmark/triton_layernorm_backward_bench/evaluator.py   --config benchmark/triton_layernorm_backward_bench/config.yaml   --iterations 10   --output /tmp/openevolve_triton_layernorm_backward_10   --save-best-to benchmark/triton_layernorm_backward_bench/evolved_best_program.py
```

To run the LayerNorm autograd-pair saved-tensor experiment, where the candidate
evolves both the forward saved tensors and the backward-from-saved kernel:

```bash
openevolve-run   benchmark/triton_layernorm_backward_bench/initial_program_autograd_pair.py   benchmark/triton_layernorm_backward_bench/evaluator_autograd_pair_speed_memory.py   --config benchmark/triton_layernorm_backward_bench/config_autograd_pair_speed_memory.yaml   --iterations 10   --output /tmp/openevolve_layernorm_autograd_pair_speed_memory_10   --save-best-to benchmark/triton_layernorm_backward_bench/evolved_best_autograd_pair_speed_memory.py
```

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
