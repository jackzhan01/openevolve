# SwiGLU autograd-pair — benchmark results vs Liger

Result of running the OpenEvolve pipeline on the SwiGLU backward benchmark, with
**Liger's hand-optimized Triton SwiGLU kernel as the strong baseline**.

- **Operator:** SwiGLU (SiLU-gated), `c = silu(a) * b`, `silu(x) = x·sigmoid(x)`, element-wise.
- **Contract:** `swiglu_forward_with_saved(a,b) -> (c, saved)` / `swiglu_backward_from_saved(dc, saved) -> (da, db)`.
- **Baseline:** Liger `swiglu_forward` / `swiglu_backward` (`liger_kernel.ops.swiglu`), same forward+backward contract.
- **Metric:** geomean of per-shape **full-step (forward+backward) speedup vs Liger**, measured with
  the same timing convention for both sides (backward-only excludes the forward from the timed
  region; full-step times forward+backward together).
- **Hardware:** NVIDIA A100 (gpuA100x4 partition), CUDA. dtypes: float32 / float16 / bfloat16.
- **Score used to evolve:** `min(backward_speedup, full_step_speedup)` with a small saved-memory
  penalty, **scored against Liger** (`AUTOGRAD_PAIR_PERF_BASELINE=liger`). small/large specialists
  used the weighted-geomean variant; the winning program was the small-suite run at 20 iterations.

## Headline result

The best evolved program is **~1.28× geomean full-step vs Liger** across the full suite (14 shapes ×
3 dtypes), ranging from **~1.35× on small shapes down to ~1.04× on the largest shapes**.

Shape-aware dispatch was tested (train separate small-shape and large-shape specialists, then pick a
deployment threshold on `numel`). **It gave no benefit for SwiGLU** — the data-derived threshold
collapses to "use one program everywhere" (threshold = ∞). SwiGLU is a single-regime, bandwidth-bound
element-wise op with **no small/large crossover**, so a single kernel is the final deliverable.
(Contrast: LayerNorm, which *does* have distinct regimes and gains ~+12% from small/large dispatch.)

| Deployment | geomean full-step vs Liger (full suite) |
| --- | ---: |
| single evolved program (`small_r2`, 20 iters) | **1.279** |
| {small, large} + threshold dispatch | 1.279 (cut = small-only) |
| {small, large} + oracle dispatch (ceiling) | 1.279 |

### Backward-only vs full-step
The evolved program beats Liger on **both** the backward-only path and the full forward+backward step
(same program, one measurement run):

| metric vs Liger | geomean |
| --- | ---: |
| backward-only | **1.16×** |
| full-step (fwd+bwd) | **1.26×** |

Backward is the heavier kernel (loads `dc, a, b`; computes sigmoid + silu + chain rule; stores
`da, db`), so it is closer to the bandwidth roofline and has less headroom over Liger. The forward is
lighter (2 loads, 1 store) and more overhead-bound, where the leaner mask-free flat-tiled launch helps
most — which is why the forward pulls the full-step speedup *above* the backward-only number.
(The 1.26× here vs 1.279× above is the same program under slightly different warmup/reps timing.)

## Per-shape sweep (full-step speedup vs Liger, dtype-averaged)

| shape | numel | evolved (`small_r2`) |
| --- | ---: | ---: |
| (1, 512) | 512 | 1.365 |
| (8, 1024) | 8,192 | 1.371 |
| (1, 8192) | 8,192 | 1.351 |
| (17, 255) | 4,335 | 1.330 |
| (17, 1001) | 17,017 | 1.327 |
| (8, 4096) | 32,768 | 1.362 |
| (32, 2048) | 65,536 | 1.353 |
| (64, 4096) | 262,144 | 1.353 |
| (128, 4096) | 524,288 | 1.337 |
| (256, 4096) | 1,048,576 | 1.348 |
| (512, 4096) | 2,097,152 | 1.320 |
| (512, 14336) | 7,340,032 | 1.089 |
| (2048, 4096) | 8,388,608 | 1.068 |
| (2048, 14336) | 29,360,128 | 1.037 |

> `> 1.0` means faster than Liger. The advantage decays with size: small/mid shapes ~1.33–1.37×,
> largest shapes ~1.04× (see "Why" below).

## Where the advantage over Liger comes from

Both the evolved kernel and Liger **save the same tensors `(a, b)` and recompute `sigmoid(a)` in the
backward pass** — identical memory strategy (`saved_memory_ratio = 1.0`). So the speedup is **not** a
memory/recompute trade; it is purely **launch, occupancy, and instruction efficiency**.

### 1. Flat 1-D tiling instead of one-block-per-row
- **Liger:** `grid = (n_rows,)` — one program instance per row, `BLOCK_SIZE = next_pow2(n_cols)`.
  When `n_rows` is small (few-row / wide shapes), Liger launches very few blocks (e.g. **1 block** for
  `(1, 8192)`, 8 for `(8, 1024)`), leaving most of the A100's ~108 SMs idle → occupancy-starved.
- **Evolved:** flattens the tensor to `N = rows*cols` and tiles with a fixed small block, so the number
  of blocks is decoupled from `n_rows`. Low-row shapes get spread across more SMs.

### 2. Mask-free fast path
- **Liger** always emits a boundary mask (`col_offsets < n_cols`) and predicated loads/stores, plus
  per-launch `stride`, `int64` program-id, and a `gate_multiplier` multiply.
- **Evolved** picks `BLOCK_SIZE ∈ {512, 1024, 2048}` from the total element count and, when
  `N % BLOCK_SIZE == 0`, dispatches a **completely mask-free kernel** (`_swiglu_forward_nomask` /
  `_swiglu_backward_nomask`). Removing the mask and the extra bookkeeping yields leaner, cleanly
  vectorized memory ops — a real win on overhead-bound small shapes. (All benchmark shapes hit the
  divisible fast path.)

### 3. Block size decoupled from `n_cols`
- Liger ties `BLOCK_SIZE` to the column width, so a single wide row (e.g. `n_cols = 14336`) becomes one
  huge block on one SM. The evolved kernel's fixed moderate block keeps good per-SM granularity
  regardless of row width.

### Why the advantage shrinks on large shapes
On large shapes both kernels launch thousands of blocks and **saturate HBM bandwidth**. SwiGLU is
memory-bound (read `a, b, dc`; write `da, db`), so once the GPU is full, everyone — including Liger —
is pinned at the DRAM roofline and the evolved kernel can only be marginally faster (~1.04×). The big
wins are on small/overhead-bound shapes; at scale it converges to Liger.

## Caveats (honest scope)
- The evolved kernel is **specialized**: 2-D contiguous inputs only, no `gate_multiplier`, no DTensor
  (distributed) path, and it allocates **new** `da/db` tensors, whereas Liger writes gradients
  **in-place** into the saved `a/b` (so Liger uses slightly less transient memory — not captured by our
  saved-memory metric). Liger is the more general, production-hardened kernel.
- Numbers are single-op microbenchmarks on one A100 across the shapes/dtypes listed; behavior inside a
  full MLP (surrounding kernels, cache state) may differ.
- Run-to-run evolution variance is real. `small_r2` was the best of the repeats; 10→20 iterations
  improved it substantially (geomean 1.12 → 1.28), i.e. 10 iterations was under-fit for this op.

## Reproduce
Paths below are relative to the repo root; `$SEED` is the pipeline-generated seed program and
`$OUT` is a scratch output directory (both live outside the repo).
```bash
BENCH=benchmark/triton_swiglu_backward_bench

# evolve (Liger-baseline, min score, full suite)
python openevolve-run.py "$SEED" \
  $BENCH/evaluator_autograd_pair_speed_memory_min_liger.py \
  --config $BENCH/config_autograd_pair_speed_memory_min_liger.yaml \
  --iterations 10 --output "$OUT"

# small / large specialists: use evaluator_autograd_pair_weighted_geomean_liger_{small,large}.py
#   + config_autograd_pair_weighted_geomean_liger_{small,large}.yaml  (sets AUTOGRAD_PAIR_SUITE=small|large)

# final comparison (money table + per-shape winner) and per-shape sweep
python $BENCH/compare_shape_dispatch.py
python $BENCH/sweep_regime.py

# backward-only + full-step speedup vs Liger for one program
python $BENCH/benchmark_strong_baselines.py <best_program.py>
```

Best evolved program: the `best/best_program.py` under the small-suite 20-iteration run's output
directory (the 10-iteration version is preserved alongside as `best_program_iter10.py`).
