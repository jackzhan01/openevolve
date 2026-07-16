# GeGLU evolve — run book

The geglu case skeleton is complete and real-run validated (Liger wrapper gate `ALL PASS`,
fusion-agent seed passes correctness). This is the remaining step: evolve the seed against Liger,
then derive the shape-dispatch result. Same recipe as `triton_rmsnorm_handwritten_backward_bench`.

Expected outcome: **negative control** — geglu is element-wise / bandwidth-bound like swiglu, so a
single regime is expected (dispatch should collapse to one program). The seed's `tanh_z` save vs
recompute tradeoff is the main thing evolution should probe.

## 0. Per-node environment (each srun shell)

```bash
conda activate openev
export LD_LIBRARY_PATH=/opt/rh/gcc-toolset-13/root/usr/lib64:$LD_LIBRARY_PATH
export OPENAI_API_KEY=...        # a key with quota
cd <repo-root>                    # the openevolve checkout
BENCH=benchmark/triton_geglu_backward_bench
SEED=$BENCH/initial_program_autograd_pair.py
```

## 1. Three evolutions, repeat x2 each (6 runs → ~3 A100 nodes)

Baseline is Liger for all (evaluators set `AUTOGRAD_PAIR_PERF_BASELINE=liger`). `full` trains on
the whole suite with the min score; `small`/`large` train their regime suite with the weighted
geomean. Each run writes its own `--output`; keep r1/r2 separate. `--save-best-to` drops the final
best program at a fixed path for the dispatch step.

```bash
# full-trained (min score, whole suite)
python openevolve-run.py $SEED \
  $BENCH/evaluator_autograd_pair_speed_memory_min_liger.py \
  --config $BENCH/config_autograd_pair_speed_memory_min_liger.yaml \
  --iterations 10 --output $BENCH/evolve_full_r1 --save-best-to $BENCH/evolve_full_r1/best_program.py   # and _r2

# small specialist (weighted geomean, small suite)
python openevolve-run.py $SEED \
  $BENCH/evaluator_autograd_pair_weighted_geomean_liger_small.py \
  --config $BENCH/config_autograd_pair_weighted_geomean_liger_small.yaml \
  --iterations 10 --output $BENCH/evolve_small_r1 --save-best-to $BENCH/evolve_small_r1/best_program.py   # and _r2

# large specialist (weighted geomean, large suite)
python openevolve-run.py $SEED \
  $BENCH/evaluator_autograd_pair_weighted_geomean_liger_large.py \
  --config $BENCH/config_autograd_pair_weighted_geomean_liger_large.yaml \
  --iterations 10 --output $BENCH/evolve_large_r1 --save-best-to $BENCH/evolve_large_r1/best_program.py   # and _r2
```

Suggested node split: node1 = full r1+r2, node2 = small r1+r2, node3 = large r1+r2 (each pair runs
serially inside the 1-hour node). Take the better repeat per group (LLM API non-determinism makes
r1/r2 differ even with a fixed seed).

## 2. Dispatch analysis + final deployable program (op-agnostic tool)

Point it at the best program from each group. It measures every program vs Liger, derives the
data-driven threshold on `regime_feature` (numel), and emits the final shape-dispatched program
(backward routes by cotangent shape, keeping `saved_tensors` tensor-only).

```bash
COMMON=benchmark/triton_backward_bench_common
python $COMMON/shape_dispatch_report.py --bench $BENCH \
  --program full=<best_program.py> \
  --program small=<best_program.py> \
  --program large=<best_program.py> \
  --timeout 90 \
  --out-report <scratch>/geglu_dispatch_report.json \
  --out-program <scratch>/geglu_final_dispatched.py
```

Best program per run is at `$BENCH/evolve_*_r1/best_program.py` (via --save-best-to), or `<output>/best/best_program.py` (or the latest
`checkpoints/checkpoint_N/best_program.py` if a run got cut off — resume with
`--checkpoint <ckpt> --iterations K` for K more).

## 3. Write up

Summarize into `RESULTS_geglu_vs_liger.md` (mirror the swiglu/rmsnorm write-ups): bwd-only and
full-step geomean speedup vs Liger for the deployed program, whether dispatch collapsed to one
program (expected yes → negative control), and the reason (element-wise, structure identical across
sizes; Liger already near roofline).
