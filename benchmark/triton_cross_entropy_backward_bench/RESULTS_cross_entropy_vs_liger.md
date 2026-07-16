# Cross-Entropy autograd-pair — benchmark results vs Liger

Result of running the full case-harness pipeline (Liger-wrapper synth → fusion seed → evolve →
dispatch) on the Cross-Entropy backward benchmark, with **Liger's cross_entropy kernel as the
strong baseline**. This is a **positive** shape-regime case (like RMSNorm/LayerNorm; contrast the
element-wise SwiGLU/GeGLU): the optimal kernel changes with vocab size, so shape-dispatch wins.

- **Operator:** mean-reduced hard-label Cross-Entropy. `loss = mean_i(logsumexp(logits_i) −
  logits_i[target_i])`; `dlogits = dloss·(softmax(logits) − onehot(target))/N`.
- **Contract:** `cross_entropy_forward_with_saved(logits, target) -> (loss, saved)` /
  `cross_entropy_backward_from_saved(dloss, saved) -> dlogits`. Cotangent `dloss` is a **scalar**;
  `target` is an int64 label tensor (non-differentiable, carried in saved).
- **Baseline:** Liger raw `cross_entropy_forward` / `cross_entropy_backward`, wrapped to the same
  contract (`strong_baselines/liger_cross_entropy.py`, auto-synthesized by the Phase-2 agent — it
  clones the logits because Liger stores the gradient in-place in logits and scales it in-place in
  backward).
- **Hardware:** NVIDIA A100 40GB. dtypes: float32 / float16 / bfloat16.
- **Regime axis:** `regime_feature = V (vocab / cols)`. CE backward is embarrassingly parallel
  across rows with a per-row reduction over V, so the kernel structure changes with V (fits one
  block vs needs streaming), not with row count.

## Final conclusion (deployed shape-dispatched program)

CE has a **real regime**; the deployed dispatcher routes by vocab size V (small specialist below
V≈40k, large specialist above), measured full-step vs Liger over the full suite (V 512 → 128k,
fp32/fp16/bf16, **0 failures**):

| metric vs Liger | geomean speedup |
| --- | ---: |
| **full-step (fwd+bwd), {small,large}+dispatch** | **3.46×** |
| full-trained single program | 3.13× |
| backward-only (best single program) | 2.53× |

**This is the largest speedup of the four operators done so far** (SwiGLU ~1.28×, GeGLU 1.44×,
RMSNorm 1.50×, **CE 3.46×**). Two reasons: (1) CE backward is compute-heavy (a full softmax over V
plus the onehot subtraction), so there is real room to out-schedule Liger; (2) Liger's CE kernel
computes the whole gradient inside the forward and stores it in-place in the (BT×V) logits — a lot
of memory traffic — whereas the evolved kernels save only a per-row `lse` (logsumexp) and recompute
`softmax = exp(logits − lse)` in backward, trading a big write/read of the V-wide gradient for cheap
recompute.

## Headline — CE HAS a real regime; dispatch wins

| Deployment | geomean full-step vs Liger (full suite) |
| --- | ---: |
| full-trained single program (`full_r1`) | 3.128 |
| **{small, large} + threshold dispatch** | **3.461**  (cut at V ≈ 40103) |
| {small, large} + oracle dispatch (ceiling) | 3.464 |
| regime collapsed to a single program? | **No** |

Dispatch beats the single full-trained program by **+10.6%** (3.128 → 3.461). The data-derived
threshold is V ≈ 40103 (between vocab 32000 and 50257).

## Per-shape sweep (full-step speedup vs Liger, dtype-avg)

| shape (BT, V) | V | small_r1 | large_r1 | full_r1 |
| --- | ---: | ---: | ---: | ---: |
| (8192, 512) | 512 | **5.19** | 3.80 | 4.64 |
| (4096, 4096) | 4096 | **4.06** | 3.33 | 4.04 |
| (4096, 16384) | 16384 | **3.32** | 3.09 | 3.27 |
| (2048, 32000) | 32000 | 3.07 | 3.09 | **3.15** |
| (777, 50257) | 50257 | 2.68 | **3.51** | 3.50 |
| (4096, 65536) | 65536 | 1.46 | **2.92** | 2.88 |
| (4096, 128256) | 128256 | 0.36 💥 | **2.97** | 0.84 |

The crossover is sharp: the small specialist dominates small vocabularies (up to **5.2×** at
V=512) but **collapses at V=128256 to 0.36×** (a single-pass softmax-grad can't hold a 128k row —
slower than Liger), while the large specialist holds **~2.97×** there. A single kernel is dominated
at one end (the full-trained program is only 0.84× at 128k), so shape-dispatch is a real win —
exactly the positive-regime signature (contrast SwiGLU/GeGLU, element-wise, which collapsed to one
program).

## Where the advantage comes from — and why dispatch is needed

- **small specialist:** a fused single-pass kernel — each row loads its V logits once, does the
  softmax-grad in one shot with modest warps. Minimal launch/占用 overhead → up to 5.2× on small V.
  But one block can't hold a 128k-wide row, so it degrades (1.46× at 65k) then loses to Liger
  (0.36× at 128k).
- **large specialist:** a streaming / multi-block online-softmax reduction that tiles the vocab
  dim and uses more warps, so it fills the GPU at 128k vocab (2.97×) where the single-pass kernel
  falls over.

Because the optimal structure itself changes with V (single-pass vs streaming), a single program
is dominated at one end — so shape-dispatch is a real win here.

## Pipeline note — a real bug fixed along the way

CE is the first case with a **non-differentiable int input** (the target labels). AtenIR's
`extract_autograd` assumed every forward input is a float that can `requires_grad_`, so it crashed
on the int64 target. Fixed generically in `atenir/extract.py`: only floating-point inputs require
grad and are differentiated; int/index inputs pass through. This unblocks CE and any future
label-carrying loss (kl_div, FLCE, …), with zero effect on the all-float ops.

## Caveats
- Only r1 was used per group (the r2 repeats were stopped early — the r1 speedup was already
  conclusive). The specialists are specialized (2-D contiguous, mean reduction, no label smoothing
  / z-loss / class weight / ignore-index masking); Liger's kernel is more general.
- Single-op microbenchmark on one A100 40GB. `dlogits` has many near-zero entries (softmax of
  non-target classes), so relative error is naturally large there and correctness is judged with an
  atol-dominated tolerance.

## Reproduce
See the case-harness flow: `strong_baselines/liger_cross_entropy.py` (Phase-2 synth),
`initial_program_autograd_pair.py` (fusion seed), the three
`evaluator_autograd_pair_*_liger*.py` / `config_*` (Phase-1 scaffold), and
`shape_dispatch_report.py` for the dispatch analysis. Best programs are in
`evolve_{full,small,large}_r1/best_program.py`; the deployed program is `ce_final_dispatched.py`
(routes by V vs the 40103 threshold) and `ce_dispatch_report.json` has the full numbers.
