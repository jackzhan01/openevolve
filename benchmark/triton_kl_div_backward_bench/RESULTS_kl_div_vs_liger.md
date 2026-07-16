# KL-Divergence autograd-pair — benchmark results vs Liger

Result of running the full case-harness pipeline (Liger-wrapper synth → fusion seed → evolve →
dispatch) on the KL-divergence backward benchmark, with **Liger's kl_div kernel as the strong
baseline**. This is the **"nominal positive, effectively negative"** data point: there is a
technical crossover (so the tool reports regime≠collapsed), but shape-dispatch buys almost nothing
(+1.5%) because the backward is element-wise and a single program already wins everywhere at scale.

- **Operator:** batchmean KL divergence, non-log target. `loss = sum_ij y_true·(log y_true −
  y_pred)/BT`; backward only wrt the log-prob input: `d_input = dloss·(−y_true)/BT`.
- **Contract:** `kl_div_forward_with_saved(y_pred, y_true) -> (loss, saved)` /
  `kl_div_backward_from_saved(dloss, saved) -> d_input`. Scalar cotangent; y_true is the (float)
  probability target, threaded through saved and NOT differentiated.
- **Baseline:** Liger raw `kldiv_forward_triton` / `kldiv_backward_triton`, wrapped to the same
  contract (`strong_baselines/liger_kl_div.py`, Phase-2 synth, passed gate on attempt 3).
- **Hardware:** NVIDIA A100 40GB. dtypes: float32 / float16 / bfloat16.
- **Regime axis:** `regime_feature = cols (V)`. The forward has a per-row reduction over V, but the
  BACKWARD is a pure element-wise scale of −y_true (no reduction) — which is why the regime is weak.

## Final conclusion (deployed program)

| metric vs Liger | geomean speedup |
| --- | ---: |
| **full-step (fwd+bwd), full-trained single program** | **2.24×** |
| {small,large} + threshold dispatch | 2.28× (only +1.5%) |
| **backward-only** | **3.42×** |

**Deploy the single full-trained program (2.24× full-step).** Shape-dispatch's +1.5% (2.28×) is not
worth a two-program deployment here. The backward is far faster than Liger (3.4×), but the forward's
over-V reduction dilutes the full-step speedup to ~2.28×.

## Headline — KL-div: a crossover exists, but dispatch is not worth it

| Deployment | geomean full-step vs Liger (full suite) |
| --- | ---: |
| full-trained single program (`full_r1`) | 2.245 |
| {small, large} + threshold dispatch | 2.279  (cut at V ≈ 11585) |
| {small, large} + oracle dispatch (ceiling) | 2.281 |
| regime collapsed to a single program? | No (but the gain is only +1.5%) |

The tool reports regime≠collapsed (there *is* a crossover), but the full-trained single program is
already within 1.5% of the dispatch ceiling — so operationally this behaves like the element-wise
negative controls, not like CE/Softmax where dispatch bought +8–11%.

## Per-shape sweep (full-step speedup vs Liger, dtype-avg)

| shape (BT, V) | V | small_r1 | large_r1 | full_r1 |
| --- | ---: | ---: | ---: | ---: |
| (4096, 512) | 512 | **2.51** | 2.48 | 2.45 |
| (4096, 4096) | 4096 | **2.55** | 2.32 | 2.31 |
| (4096, 16384) | 16384 | 2.13 | **2.24** | 2.24 |
| (2048, 32000) | 32000 | 1.46 | **2.26** | 2.26 |
| (777, 50257) | 50257 | 0.51 | 1.93 | **1.99** |
| (4096, 128256) | 128256 | 0.33 💥 | 1.98 | **1.98** |

The small (single-pass) specialist edges the others only at small V (2.5 vs 2.24), and **collapses
at large V** (0.33× at 128k). But the full-trained program already matches the large specialist at
large V (both ~1.98) — so unlike CE/Softmax, there is no regime where a specialist meaningfully
beats the single full-trained program. Hence dispatch ≈ single.

## Why the weak regime (the interesting bit)
KL-div's **backward is embarrassingly element-wise**: `d_input = −y_true/BT`, no reduction, no V-
dependent structure. So the backward kernel has no real regime — a single tiling wins at all V. The
*forward* does have an over-V reduction (a potential regime), but it is the minority of full-step
cost here and the full-trained program handles it fine. Net: a technical crossover (from the small
specialist overfitting a small-V single-pass forward and dying at large V) but no operational value
to dispatching. Backward-only is a strong 3.4× — the win is real, it just isn't regime-structured.

## Operator spectrum so far (deployed full-step vs Liger)

| op | backward structure | regime | deploy | speedup |
| --- | --- | --- | --- | ---: |
| SwiGLU / GeGLU | element-wise | none | single | 1.28× / 1.44× |
| **KL-div** | **element-wise** (fwd has reduction) | **nominal, +1.5%** | **single** | **2.24× (bwd 3.4×)** |
| RMSNorm | row reduction (cross-row dweight) | positive | dispatch | 1.50× |
| Softmax | row reduction | positive | dispatch | 1.83× |
| Cross-Entropy | softmax-grad over V | positive | dispatch | 3.46× |

KL-div slots between the two groups: its *backward* is element-wise like SwiGLU/GeGLU (so dispatch
doesn't pay), but its absolute speedup is high (2.24× / bwd 3.4×) because Liger's kl_div backward
leaves more on the table than its softmax/rmsnorm kernels.

## Caveats
- Nodes hit the 1-hour limit at iter 7-8; the r1 best is from **checkpoint_5** (evolution had
  already saturated by iter 5 — iter5≈iter8 metrics — so this is not a material loss). r2 not run.
- Specialists specialized (2-D contiguous, batchmean, non-log target); Liger's kernel is more
  general.

## Reproduce
`strong_baselines/liger_kl_div.py` (Phase-2 synth), `initial_program_autograd_pair.py` (fusion
seed), the three `evaluator_*_liger*.py` / `config_*` (Phase-1 scaffold), `shape_dispatch_report.py`
for dispatch. Best programs in `evolve_{full,small,large}_r1/checkpoints/checkpoint_5/best_program.py`;
deployed program is `kl_final_dispatched.py`; `kl_dispatch_report.json` has the numbers.
