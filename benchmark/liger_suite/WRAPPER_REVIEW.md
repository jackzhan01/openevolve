# Liger baseline wrapper — fairness review

Manual review of all 7 auto-synthesized `strong_baselines/liger_<op>.py` wrappers
(2026-07-16). Checklist: no host-device syncs in the timed paths (`.item()/.cpu()/
int(tensor)`), no forward-recompute in backward, no extra full-tensor passes that slow the
baseline (inflating our speedups), saved-tensor set matches what Liger's own
`autograd.Function` saves.

## Verdicts

| op | verdict | notes |
| --- | --- | --- |
| relu_squared | ✅ clean | thin passthrough of the raw ops |
| dyt | ✅ clean | saves (x, α, γ, β) exactly like Liger's Function |
| sparsemax | ✅ clean | saves `out_flat` like Liger's Function; dim=-1 |
| poly_norm | ✅ clean on the benchmark path | contains a dead branch for weight shape (3, H) that computes the forward in PyTorch — never taken for this benchmark's (3,) weights; benchmark path is pure Liger |
| fused_add_rms_norm | ✅ minor | single-output form must feed the kernel `ds=0`: one `zeros_like(s)` memset per backward call. Inherent to bridging Liger's two-output interface; small vs the kernel itself |
| tvd | 🔧 fixed in review | wrapper added a full `where(p==q, 0, grads)` pass in the TIMED forward to patch tie-handling subgradients. Disagreement is only 0.5/BT per tied element (deep inside atol) — patch removed, re-gated |
| jsd | 🔧 fixed in review | backward `.clone()`d BOTH the precomputed grad and dout every call. Liger's `jsd_backward` is not in-place (returns a fresh `grad_output * dX`), so the clones only slowed the timed baseline backward — removed, re-gated |

## Known semantic caveats (documented, deliberate)

- **tvd** — task_spec treats BOTH p and q as differentiable (autograd through the naive
  forward gives dq = -dp). Liger returns only dp, so the wrapper adds one negation kernel
  for dq. Deviates from the kl_div precedent (target has no grad) but is internally
  consistent and symmetric for candidates; the negation is trivial.
- **jsd** — for half-precision inputs the wrapper upcasts to fp32 before calling Liger
  (Liger's bf16 path stores the precomputed grad in bf16, which cannot match the fp32-grade
  oracle). The bf16 baseline therefore effectively measures "Liger at fp32", i.e. moves
  more bytes than Liger-native bf16 would. Conservative FOR Liger comparisons: treat jsd's
  half-precision speedups as upper bounds.
- **sparsemax** — benchmark is fp32-only: sparsemax gradients are discontinuous at the
  support boundary and half-precision quantization flips boundary membership between any
  two correct implementations (pointwise comparison is ill-posed there).
- **dyt** — Liger's fwd kernel launches rows on CUDA grid axis Y (hard cap 65535), so the
  shape grid stays ≤ 65024 rows; Liger itself crashes above that.
- **Liger's own quirk (kept, it ships that way)** — tvd/jsd backward call
  `torch.equal(grad_output, tensor(1.0))` (a device sync) to skip the multiply for
  last-layer losses. We time Liger as shipped.

## Process notes

- All three first-round gate failures were BENCHMARK-SPEC bugs, not wrapper bugs
  (dyt shapes beyond the baseline's launch limit; sparsemax half-precision comparison;
  constant atol for √rows-growing reduction noise). Gate-fails-first-suspect-the-harness
  strikes again.
- A failed synthesis used to leave its last rejected wrapper in `strong_baselines/`
  (the gate imports from the product location), making the next prep run skip with a fake
  "wrapper exists". Fixed: failure now unlinks the artifact.
