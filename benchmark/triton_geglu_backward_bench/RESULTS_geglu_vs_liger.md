# GeGLU autograd-pair — benchmark results vs Liger

Result of running the full OpenEvolve pipeline (fusion-agent seed → evolve → dispatch) on the GeGLU
backward benchmark, with **Liger's hand-optimized Triton GeGLU kernel as the strong baseline**.
This is a *negative* shape-regime case (like SwiGLU, contrast RMSNorm): a single program wins
everywhere, so shape-dispatch collapses to one program.

- **Operator:** GeGLU (GELU-gated activation), tanh approximation to match Liger:
  `c = gelu_tanh(a) * b`, `gelu_tanh(x) = 0.5·x·(1 + tanh(√(2/π)·(x + 0.044715·x³)))`.
- **Contract:** `geglu_forward_with_saved(a, b) -> (c, saved)` /
  `geglu_backward_from_saved(dc, saved) -> (da, db)`.
- **Baseline:** Liger raw `geglu_forward` / `geglu_backward`, wrapped to the same contract
  (`strong_baselines/liger_geglu.py`, auto-synthesized by the case-harness agent — it correctly
  clones the saved `a`/`b` because Liger's `geglu_backward` writes the grads back in place).
- **Metric:** geomean of per-shape full-step (fwd+bwd) and backward-only speedup vs Liger, same
  timing convention on both sides.
- **Hardware:** NVIDIA A100 40GB. dtypes: float32 / float16 / bfloat16.
- **Regime axis:** `regime_feature = numel (rows·cols)`. GeGLU is element-wise and
  bandwidth-bound, so the optimal kernel structure is the same at all sizes → no regime.

## Final conclusion (deployed program)

The regime collapsed to a **single program** (the "small" specialist, which is best on every shape
including the largest). Measured end-to-end vs Liger over the full suite (numel 512 → 29M,
fp32/fp16/bf16, **0 failures**):

| metric vs Liger | geomean speedup |
| --- | ---: |
| **full-step (fwd+bwd)** | **1.44×** |
| **backward-only** | **1.59×** |

**Why:** the evolved winner beats Liger by two element-wise, bandwidth-oriented moves:
1. **Drops the saved intermediate.** The fusion-agent seed saved `(a, b, tanh_z)` and reused
   `tanh_z` in backward to skip recomputing tanh. Evolution **removed `tanh_z`** (saves only
   `(a, b)`) and recomputes tanh in backward — because GeGLU is bandwidth-bound, not writing +
   re-reading a full fp32 tensor beats saving the tanh recompute. (`saved_memory_ratio` 1.5 → 1.0.)
2. **Flat 1-D tiling + mask-free fast path.** A single flattened grid with an `EVEN` branch that
   drops the bounds mask when `numel` is a multiple of the block size — same trick that helped
   SwiGLU. Block size is decoupled from `n_cols`.

Because the kernel structure is identical at every size (unlike RMSNorm's cross-row `dweight`
reduction), one program dominates and dispatch collapses — a negative control, as expected.

## Headline — GeGLU has NO regime; dispatch collapses to one program

| Deployment | geomean full-step vs Liger (full suite) |
| --- | ---: |
| full-trained single program (`full_r2`) | 1.346 |
| **{small, large} + threshold dispatch** | **1.439** (cut = small-only) |
| {small, large} + oracle dispatch (ceiling) | 1.441 |
| regime collapsed to a single program? | **Yes** |

The "small" specialist (trained with a weighted geomean on the small suite) turned out best on the
*whole* suite, so the data-derived threshold is "small-only" — i.e. deploy the small specialist
everywhere. Dispatch's ceiling (oracle 1.441) barely exceeds the single small program (1.439),
confirming no useful regime split (contrast RMSNorm, where dispatch beat the single program by
+13.6%).

## Per-shape sweep (full-step speedup vs Liger, dtype-avg)

| shape | numel | small_r1 | large_r1 | full_r2 |
| --- | ---: | ---: | ---: | ---: |
| (1, 512) | 512 | **1.42** | 1.40 | 1.37 |
| (1, 8192) | 8192 | **1.45** | 1.40 | 1.39 |
| (8, 4096) | 32768 | **1.44** | 1.40 | 1.39 |
| (128, 4096) | 524288 | **1.45** | 1.40 | 1.39 |
| (256, 4096) | 1048576 | **1.43** | 1.39 | 1.38 |
| (512, 14336) | 7340032 | **1.50** | 1.27 | 1.26 |
| (2048, 4096) | 8388608 | **1.43** | 1.21 | 1.21 |
| (2048, 14336) | 29360128 | **1.52** | 1.24 | 1.24 |

The small specialist dominates at **every** numel — there is no crossover where the large
specialist wins, so no threshold helps. (Its advantage even *widens* at the largest shapes, where
dropping the `tanh_z` traffic matters most.) This is exactly the single-regime signature.

## Comparison to SwiGLU / RMSNorm

| op | element-wise? | regime? | deployed vs Liger (full-step) | dispatch verdict |
| --- | --- | --- | ---: | --- |
| SwiGLU | yes | no | ~1.28× | collapsed to 1 program |
| **GeGLU** | **yes** | **no** | **1.44×** | **collapsed to 1 program** |
| RMSNorm | no (reduction) | yes | 1.50× | {small,large} dispatch, +13.6% |

GeGLU lands higher than SwiGLU because its backward is heavier (the `da` term is a cubic-ish
polynomial + tanh), so Liger's recompute-in-backward leaves more room, and the "drop the saved
intermediate + flat tiling" win compounds. Same *pipeline* correctly produced a single program for
both element-wise ops and a real dispatcher for the reduction op.

## Caveats
- The evolved kernel is specialized (2-D contiguous, tanh-approx GELU, gate·up form); Liger's is
  the more general kernel.
- Single-op microbenchmark on one A100 40GB; run-to-run evolution variance is real (r1 vs r2 differ
  even with a fixed seed due to LLM API non-determinism — all three groups' r1 edged out r2; the
  dispatch tool measured all 6 and took the better repeat per regime).

## Reproduce
See `RUN_evolve.md` (env setup, the 3 evolve commands, dispatch analysis). The seed came from the
fusion agent (Pipeline A); the Liger wrapper from the case-harness agent's Phase-2 synthesizer.
Best programs are in `evolve_{full,small,large}_{r1,r2}/best_program.py`; the deployed program is
`geglu_final_dispatched.py` (the collapsed single winner) and `geglu_dispatch_report.json` has the
full numbers.
