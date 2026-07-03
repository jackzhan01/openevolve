# AtenIR-to-Triton Pipeline Notes

The current research path compares three LayerNorm autograd-pair seed
generation pipelines. All three should ultimately feed the same OpenEvolve
speed-memory evaluator:

```text
layernorm_forward_with_saved(x, weight, bias, eps) -> y, saved_tensors
layernorm_backward_from_saved(dy, saved_tensors, eps) -> dx, dweight, dbias
```

## Current Autograd-Pair Pipelines

### Pipeline A: Main AtenIR Autograd-Pair Fusion Agent

Path:

```text
pipeline/autograd_pair_fusion_agent/
pipeline/run_pipeline_a_autograd_pair.py
pipeline/run_autograd_pair_fusion_agent.py      # compatibility alias
```

Role:

```text
forward reference
-> AtenIR backward graph summary
-> LLM generates autograd-pair seed
```

This is the main pipeline for the current work.

```bash
python -m pipeline.run_pipeline_a_autograd_pair \
  --forward benchmark.triton_layernorm_backward_bench.forward_ref:layernorm_forward_ref \
  --example-input "[(8,64) f32, (64) f32, (64) f32]" \
  --output-dir /u/wzhan/tmp/atenir_layernorm_autograd_pair \
  --model gpt-5.5 \
  --max-attempts 3
```

### Pipeline B: LLM-Free Hand-Written Dispatch Seed

Path:

```text
pipeline/handwritten_dispatch/
pipeline/run_pipeline_b_handwritten_dispatch.py
pipeline/run_handwritten_dispatch.py            # compatibility alias
```

Role:

```text
AtenIR graph
-> hand-written primitive dispatch
-> dispatch-free Triton program
-> manually wrapped autograd-pair seed
```

This pipeline does not use an LLM for seed generation. It emits
`dispatch_program.py` plus `initial_program_autograd_pair.py`.

```bash
python -m pipeline.run_pipeline_b_handwritten_dispatch \
  --forward benchmark.triton_layernorm_backward_bench.forward_ref:layernorm_forward_ref \
  --example-input "[(8,64) f32, (64) f32, (64) f32]" \
  --output-dir /u/wzhan/tmp/handwritten_layernorm_dispatch \
  --dtype float32 \
  --dtype float16 \
  --emit-autograd-pair-seed
```

The autograd-pair wrapper initially saves only original inputs. OpenEvolve can
then change the saved tensor contract.

### Pipeline C: No-AtenIR Baseline

Path:

```text
pipeline/no_atenir_fusion_agent/
pipeline/run_pipeline_c_no_atenir.py
pipeline/run_no_atenir_fusion_agent.py          # compatibility alias
```

Role:

```text
forward source only
-> LLM derives autograd-pair seed
```

Use this as the baseline for measuring how much AtenIR context helps.

```bash
python -m pipeline.run_pipeline_c_no_atenir \
  --output-dir /u/wzhan/tmp/no_atenir_layernorm_ablation \
  --model gpt-5.5 \
  --max-attempts 3
```

It writes prompts, attempts, verifier reports, and `cost_summary.json` for
correctness/cost comparison.

## Common OpenEvolve Step

After any pipeline emits an autograd-pair seed, run:

```bash
openevolve-run \
  /path/to/initial_program_autograd_pair.py \
  benchmark/triton_layernorm_backward_bench/evaluator_autograd_pair_speed_memory.py \
  --config benchmark/triton_layernorm_backward_bench/config_autograd_pair_speed_memory.yaml \
  --iterations 10 \
  --output /u/wzhan/tmp/openevolve_layernorm_<pipeline>_10
```

## Running A/B/C on Other Benchmarks

All three pipelines are now operator-generic. The LayerNorm contract is still
the default (existing commands are unchanged), but each pipeline accepts an
`--op-spec <op>_spec.json` to target another benchmark. Four operators ship with
the full asset set (op-spec + autograd-pair evaluator + speed/memory config +
reference seed):

| Operator | op-spec (`pipeline/autograd_pair_fusion_agent/`) | forward ref | `--example-input` |
|---|---|---|---|
| rmsnorm | `rmsnorm_spec.json` | `benchmark.triton_rmsnorm_backward_bench.forward_ref:rmsnorm_forward_ref` | `"[(8,64) f32, (64) f32]"` |
| matmul | `matmul_spec.json` | `benchmark.triton_matmul_backward_bench.forward_ref:matmul_forward_ref` | `"[(64,64) f32, (64,64) f32]"` |
| linear | `linear_spec.json` | `benchmark.triton_linear_backward_bench.forward_ref:linear_forward_ref` | `"[(64,64) f32, (64,64) f32, (64) f32]"` |
| layernorm_linear | `layernorm_linear_spec.json` | `benchmark.triton_layernorm_linear_backward_bench.forward_ref:layernorm_linear_forward_ref` | `"[(64,128) f32, (128) f32, (128) f32, (128,256) f32]"` |

Example for **rmsnorm** (swap the op-spec / forward / example-input / bench dir
for the others):

```bash
# Pipeline A — AtenIR autograd-pair fusion agent
python -m pipeline.run_pipeline_a_autograd_pair \
  --forward benchmark.triton_rmsnorm_backward_bench.forward_ref:rmsnorm_forward_ref \
  --example-input "[(8,64) f32, (64) f32]" \
  --op-spec pipeline/autograd_pair_fusion_agent/rmsnorm_spec.json \
  --evaluator benchmark/triton_rmsnorm_backward_bench/evaluator_autograd_pair.py \
  --output-dir /tmp/A_rmsnorm --model gpt-5.5 --max-attempts 3

# Pipeline B — LLM-free hand-written dispatch seed
python -m pipeline.run_pipeline_b_handwritten_dispatch \
  --forward benchmark.triton_rmsnorm_backward_bench.forward_ref:rmsnorm_forward_ref \
  --example-input "[(8,64) f32, (64) f32]" \
  --op-spec pipeline/autograd_pair_fusion_agent/rmsnorm_spec.json \
  --output-dir /tmp/B_rmsnorm --dtype float32 --dtype float16 \
  --emit-autograd-pair-seed

# Pipeline C — no-AtenIR baseline
python -m pipeline.run_pipeline_c_no_atenir \
  --forward benchmark.triton_rmsnorm_backward_bench.forward_ref:rmsnorm_forward_ref \
  --op-spec pipeline/autograd_pair_fusion_agent/rmsnorm_spec.json \
  --evaluator benchmark/triton_rmsnorm_backward_bench/evaluator_autograd_pair.py \
  --output-dir /tmp/C_rmsnorm --model gpt-5.5 --max-attempts 3
```

Then run the common OpenEvolve speed/memory step against the matching
`benchmark/triton_rmsnorm_backward_bench/evaluator_autograd_pair_speed_memory.py`
and `config_autograd_pair_speed_memory.yaml`.

> The new bench assets were authored without a CUDA environment available, so
> they are validated by inspection + `py_compile` only. Smoke-test each bench's
> reference seed once on a GPU before a full A/B/C run, e.g.
> `python benchmark/triton_rmsnorm_backward_bench/evaluator_autograd_pair.py \`
> `       benchmark/triton_rmsnorm_backward_bench/initial_program_autograd_pair.py`
> (expect `"correct": 1.0`).

## Development / Historical Pipelines

The following directories are useful development artifacts but are not the main
three-way autograd-pair comparison:

```text
pipeline/fusion_agent/
pipeline/run_fusion_agent.py
```

Older standalone-backward Pipeline A. Generates `layernorm_backward_triton(...)`
rather than the autograd-pair API.

```text
pipeline/primitive_atenir_lowering_agent/
pipeline/run_lowering_agent.py
pipeline/run_lowering_correctness.py
```

Earlier per-op LLM lowering work.

```text
pipeline/kernel_fusion_agent/
pipeline/run_kernel_fusion_agent.py
```

Earlier kernel-aware fusion over verified per-op lowering context. Standalone
backward API.

```text
pipeline/handwritten_fusion_agent/
pipeline/run_handwritten_fusion_agent.py
```

Earlier LLM fusion using hand-written dispatch context. Standalone backward API.

```text
pipeline/run_layernorm_pipeline_comparison.py
pipeline/run_fusion_benchmark_suite.py
```

Experiment runners from earlier standalone-backward comparisons.

Shared LLM and failure-taxonomy helpers live in `pipeline/shared/`.
