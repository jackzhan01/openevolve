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
