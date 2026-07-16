"""Task spec for the Cross-Entropy Triton backward benchmark.

Operator: Cross-Entropy loss (hard label, mean reduction), matching Liger's cross_entropy
kernel config (no label smoothing / z-loss / class weight; ignore_index=-100).

Forward:  loss = mean_i( -log_softmax(logits_i)[target_i] )   -> scalar
Backward: dlogits = dloss * (softmax(logits) - onehot(target)) / N   (N = #tokens = BT)

Standalone backward API:
    cross_entropy_backward_triton(dloss, logits, target) -> dlogits

Autograd-pair API:
    cross_entropy_forward_with_saved(logits, target) -> (loss, saved_tensors)
    cross_entropy_backward_from_saved(dloss, saved_tensors) -> dlogits

Notes / why this case is different from the element-wise ones:
  - The cotangent `dloss` is a SCALAR (loss is scalar under mean reduction).
  - `target` is an int64 label tensor (non-differentiable); backward is only wrt logits.
    It is threaded through the saved tensors (an int tensor is still a torch.Tensor).
  - Liger computes the gradient INSIDE the forward and stores it in-place in `logits`, and its
    backward scales that stored gradient in-place -> the Liger wrapper must clone (idempotency).
  - Regime axis = V (vocab / cols): small V fits one block (single-pass softmax-grad), large V
    (e.g. 128k) needs streaming/multi-block reduction. Row count BT is just parallelism.
"""

from dataclasses import dataclass


CANDIDATE_FN_NAME = "cross_entropy_backward_triton"
OUTPUT_NAMES = ("dlogits",)

AUTOGRAD_PAIR_FORWARD_FN_NAME = "cross_entropy_forward_with_saved"
AUTOGRAD_PAIR_BACKWARD_FN_NAME = "cross_entropy_backward_from_saved"
# make_inputs returns (dloss, logits, target)
AUTOGRAD_PAIR_COTANGENT_INDEX = 0            # dloss (scalar)
AUTOGRAD_PAIR_FORWARD_INPUT_INDICES = (1, 2)  # logits, target
AUTOGRAD_PAIR_BACKWARD_EXTRA_INPUT_INDICES = ()  # target is carried in saved_tensors
AUTOGRAD_PAIR_MEMORY_INPUT_INDICES = (1,)     # logits defines the memory budget

AUTOGRAD_PAIR_API = """def cross_entropy_forward_with_saved(logits, target):
    return loss, saved_tensors

def cross_entropy_backward_from_saved(dloss, saved_tensors):
    return dlogits
"""

AUTOGRAD_PAIR_TASK_CONTEXT = """Forward is mean-reduced cross-entropy with hard labels:
    logits : [BT, V] float, target : [BT] int64 in [0, V)
    lse_i  = logsumexp(logits_i)                      # over vocab dim V
    loss_i = lse_i - logits_i[target_i]
    loss   = mean_i(loss_i) = sum_i(loss_i) / N,  N = BT (number of tokens)

Backward returns dlogits (same shape as logits):
    p_i        = softmax(logits_i)                    # over vocab dim V
    dlogits_i  = dloss * (p_i - onehot(target_i)) / N

dloss is a SCALAR cotangent. target is an int64 label tensor (no gradient); the forward may save
it (and softmax / logits) into saved_tensors for the backward. Use fp32 for the softmax/logsumexp
reduction and preserve the logits dtype in the output. The saved-tensor contract is flexible.
"""


@dataclass(frozen=True)
class TestCase:
    rows: int      # BT = batch * seq
    cols: int      # V = vocab size
    dtype_name: str
    atol_value: float
    rtol_value: float


CORRECTNESS_CASES = [
    TestCase(8, 512, "float32", 2e-5, 1e-3),
    TestCase(16, 1024, "float32", 2e-5, 1e-3),
    TestCase(32, 4096, "float16", 5e-4, 1e-2),
    TestCase(64, 2048, "float16", 5e-4, 1e-2),
    TestCase(32, 4096, "bfloat16", 2e-3, 2e-2),
    TestCase(64, 2048, "bfloat16", 2e-3, 2e-2),
]

# Regime axis = V (cols): small V is single-pass (fits one block), large V needs streaming
# multi-block softmax-grad. Sweep V from tiny to 128k at a typical BT, plus BT variation at a
# realistic vocab, plus non-power-of-two vocabs (gpt2=50257, +1 offsets).
_CE_BENCHMARK_SHAPES = [
    # small V (< REGIME_SPLIT): single-pass regime
    (4096, 512),
    (8192, 512),
    (4096, 4096),
    (16384, 4096),
    (4096, 8192),
    (2048, 8192),
    # large V (>= REGIME_SPLIT): streaming regime
    (4096, 16384),
    (512, 32000),
    (2048, 32000),
    (4096, 32000),
    (8192, 32000),
    (4096, 65536),
    (4096, 128256),   # llama3 vocab
    (777, 50257),     # gpt2 vocab, non-aligned rows/cols
]

_BENCHMARK_DTYPE_TOLERANCES = {
    "float32": (2e-5, 1e-3),
    "float16": (5e-4, 1e-2),
    "bfloat16": (2e-3, 2e-2),
}


def _make_benchmark_cases() -> list[TestCase]:
    cases: list[TestCase] = []
    for rows, cols in _CE_BENCHMARK_SHAPES:
        for dtype_name, (atol_v, rtol_v) in _BENCHMARK_DTYPE_TOLERANCES.items():
            cases.append(TestCase(rows, cols, dtype_name, atol_v, rtol_v))
    return cases


BENCHMARK_CASES = _make_benchmark_cases()


# --- Shape-regime metadata (for shape-aware evolution / dispatch) ---------------
# Cross-entropy backward is embarrassingly parallel across rows (each token independent), with a
# per-row reduction over the vocab dim V. So the optimal kernel structure changes with V (fits a
# single block vs needs streaming), NOT with row count. REGIME_SPLIT is the rule-of-thumb cut used
# only to build the small/large TRAINING suites; the deployment threshold is derived afterward.
REGIME_SPLIT = 16384  # vocab size V; small: < split (single-pass), large: >= split (streaming)


def regime_feature(case: "TestCase") -> float:
    """Scalar that determines which shape-regime a case falls in (vocab size V)."""
    return float(case.cols)


def case_weight(case: "TestCase") -> float:
    """Per-case weight for the weighted-geomean score when training a specialist.

    Emphasizes the tail of each regime (V far from the split) and de-emphasizes near-boundary
    shapes. Suite-agnostic log2-distance from REGIME_SPLIT.
    """
    import math

    dist = abs(math.log2(regime_feature(case)) - math.log2(REGIME_SPLIT))
    return float(max(1.0, round(dist)))


SMALL_CASES = [c for c in BENCHMARK_CASES if regime_feature(c) < REGIME_SPLIT]
LARGE_CASES = [c for c in BENCHMARK_CASES if regime_feature(c) >= REGIME_SPLIT]


def make_liger_autograd_pair_fns():
    """(forward_with_saved, backward_from_saved) built from Liger's raw cross_entropy_forward /
    cross_entropy_backward. Used as the perf baseline when AUTOGRAD_PAIR_PERF_BASELINE=liger."""
    try:
        from benchmark.triton_cross_entropy_backward_bench.strong_baselines.liger_cross_entropy import (
            make_liger_cross_entropy_autograd_pair_fns,
        )
    except ImportError:
        from strong_baselines.liger_cross_entropy import make_liger_cross_entropy_autograd_pair_fns
    return make_liger_cross_entropy_autograd_pair_fns()


def _dtype(torch_module, dtype_name: str):
    if dtype_name == "float32":
        return torch_module.float32
    if dtype_name == "float16":
        return torch_module.float16
    if dtype_name in ("bfloat16", "bf16"):
        return torch_module.bfloat16
    raise ValueError(f"Unsupported dtype: {dtype_name}")


def seed_for_case(case: TestCase) -> int:
    return case.rows * 100003 + case.cols


def case_metadata(case: TestCase):
    return {
        "shape": [case.rows, case.cols],
        "dtype": case.dtype_name,
        "reduction": "mean",
    }


def make_inputs(torch_module, case: TestCase):
    """Return (dloss, logits, target): scalar cotangent, [BT,V] logits, [BT] int64 labels."""
    dtype = _dtype(torch_module, case.dtype_name)
    logits = torch_module.randn((case.rows, case.cols), device="cuda", dtype=dtype)
    target = torch_module.randint(0, case.cols, (case.rows,), device="cuda", dtype=torch_module.int64)
    # non-unity positive scalar cotangent (exercises the grad_output-mul path)
    dloss = (torch_module.rand((), device="cuda", dtype=dtype) + 0.5)
    return dloss, logits, target


def torch_oracle(torch_module, dloss, logits, target):
    """PyTorch autograd backward: returns dlogits (in logits dtype)."""
    lr = logits.detach().clone().float().requires_grad_(True)
    loss = torch_module.nn.functional.cross_entropy(lr, target, reduction="mean", ignore_index=-100)
    loss.backward(dloss.float())
    return lr.grad.to(logits.dtype)


def autograd_pair_forward_oracle(torch_module, logits, target):
    """Reference forward output (scalar loss) for correctness checking of the autograd-pair."""
    loss = torch_module.nn.functional.cross_entropy(
        logits.float(), target, reduction="mean", ignore_index=-100
    )
    return loss.to(logits.dtype)


def atol(case: TestCase, output_name: str) -> float:
    return case.atol_value


def rtol(case: TestCase, output_name: str) -> float:
    return case.rtol_value


def correctness_hint() -> str:
    return (
        "p=softmax(logits, dim=-1); dlogits = dloss*(p - onehot(target))/N, N=BT; "
        "loss=mean_i(logsumexp(logits_i) - logits_i[target_i])"
    )
