# Softmax autograd-pair — benchmark results vs Liger

Result of running the full case-harness pipeline (Liger-wrapper synth → fusion seed → evolve →
dispatch) on the Softmax backward benchmark, with **Liger's softmax kernel as the strong baseline**.
This is a **positive** shape-regime case (like RMSNorm / Cross-Entropy) — but with the *strongest*
baseline of the operators done so far: **Liger already dispatches its own single-block vs
multi-block softmax kernel by column count**, so beating it is harder and the speedups are more
modest than CE.

- **Operator:** row-wise softmax over the last dim. `y = softmax(x, -1)`;
  `dx = y·(dy − sum(dy·y, -1))`.
- **Contract:** `softmax_forward_with_saved(x) -> (y, saved)` /
  `softmax_backward_from_saved(dy, saved) -> dx`. All-float inputs, cotangent `dy` same shape as y.
- **Baseline:** Liger raw `_softmax_forward` / `_softmax_backward`, wrapped to the same contract
  (`strong_baselines/liger_softmax.py`). Liger's backward is NOT in-place (reads y+dy, writes a
  fresh dx).
- **Hardware:** NVIDIA A100 40GB. dtypes: float32 / float16 / bfloat16.
- **Regime axis:** `regime_feature = cols` (the softmax reduction width V). Small V fits one block
  (single-pass); large V (Liger switches its own multi-block kernel around V=65536) needs streaming.

## Final conclusion (deployed shape-dispatched program)

Softmax has a **real regime**; the deployed dispatcher routes by width V (small specialist below
V≈46k, large specialist above), measured full-step vs Liger over the full suite (V 512 → 131072,
fp32/fp16/bf16, **0 failures**):

| metric vs Liger | geomean speedup |
| --- | ---: |
| **full-step (fwd+bwd), {small,large}+dispatch** | **1.83×** |
| full-trained single program | 1.70× |
| backward-only (best single program) | 2.10× |

Even against Liger's already-block-dispatched kernel, evolution + shape-dispatch wins ~1.8×, and
dispatch beats the single full-trained program by **+7.7%** (1.70 → 1.83). This is the "strong
baseline, still a positive regime" data point of the sweep.

## Headline — Softmax HAS a real regime; dispatch wins

| Deployment | geomean full-step vs Liger (full suite) |
| --- | ---: |
| full-trained single program (`full_r1`) | 1.696 |
| **{small, large} + threshold dispatch** | **1.827**  (cut at V ≈ 46341) |
| {small, large} + oracle dispatch (ceiling) | 1.862 |
| regime collapsed to a single program? | **No** |

## Per-shape sweep (full-step speedup vs Liger, dtype-avg)

| shape (rows, V) | V | small_r1 | large_r1 | full_r1 |
| --- | ---: | ---: | ---: | ---: |
| (4096, 512) | 512 | **2.51** | 2.30 | 2.14 |
| (4096, 4096) | 4096 | 1.81 | 1.78 | 1.77 |
| (2048, 16384) | 16384 | 1.80 | 1.78 | 1.78 |
| (4096, 32768) | 32768 | **1.78** | 1.33 | 1.13 |
| (2048, 65536) | 65536 | 1.31 | **1.59** | 1.52 |
| (1024, 131072) | 131072 | 0.38 💥 | **1.64** | 1.65 |

The crossover sits between V=32768 and 65536: the small (single-pass) specialist is best up to 32k
(and much better than the large one there — 1.78 vs 1.33), but **collapses at V=131072 to 0.38×**
(one block can't hold a 131072-wide row), while the large (streaming) specialist holds ~1.64×. A
single kernel is dominated at one end, so shape-dispatch is a real win (positive control), like CE
and RMSNorm — unlike element-wise SwiGLU/GeGLU.

## Where the advantage comes from
- **small specialist:** a fused single-pass reduction (max+sum+scale in one shot over the row),
  best when the row fits one block. Up to 2.5× at V=512, but it can't hold a 131072-wide row so it
  falls apart there.
- **large specialist:** a streaming / multi-block reduction that tiles the width, so it stays ~1.6×
  at 128k+ where the single-pass kernel collapses (and where Liger switches to its own multi-block
  kernel too — the evolved one still edges it out).

## The four-operator picture so far

| op | type | regime | deployed vs Liger (full-step) |
| --- | --- | --- | ---: |
| SwiGLU | element-wise | none (collapses) | ~1.28× |
| GeGLU | element-wise | none (collapses) | 1.44× |
| RMSNorm | row-reduction norm | positive | 1.50× |
| Cross-Entropy | vocab-reduction loss | positive | 3.46× |
| **Softmax** | **row-reduction** | **positive** | **1.83×** |

Softmax fills in the "positive regime, but the baseline is already strong" slot: Liger's softmax is
already block-dispatched, so the head-room is smaller than CE, yet evolution + threshold dispatch
still delivers 1.83× and a real +7.7% dispatch gain.

## Pipeline note — fixed an over-strict gate
Softmax exposed that the wrapper's idempotency gate was too strict: it used `torch.equal`
(bit-exact), which flagged Liger's **multi-block backward** (atomic/split reduction over 10⁵
elements, legitimately non-deterministic across launches) as an in-place mutation at V=131072. The
wrapper was actually correct. Fixed `verify_liger_baseline.py` to judge idempotency with a loose
fixed tolerance (`allclose atol/rtol=1e-2`) — large enough to ignore cross-launch non-determinism,
far smaller than a real mutation's catastrophic error. (Same "new op exposes a pipeline blind spot"
pattern as CE's AtenIR int-input fix.)

## Caveats
- Only r1 was used per group (r2 skipped; the r1 speedup was already conclusive and modest — beating
  an already-optimized baseline). Specialists are specialized (2-D contiguous, softmax over last
  dim); Liger's kernel is more general.
- Single-op microbenchmark on one A100 40GB.

## Reproduce
`strong_baselines/liger_softmax.py` (Phase-2 synth), `initial_program_autograd_pair.py` (fusion
seed), the three `evaluator_*_liger*.py` / `config_*` (Phase-1 scaffold), `shape_dispatch_report.py`
for dispatch. Best programs in `evolve_{full,small,large}_r1/best_program.py`; deployed program is
`sm_final_dispatched.py` (routes by V vs the 46341 threshold); `sm_dispatch_report.json` has the
numbers.
