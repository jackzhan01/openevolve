"""Task spec for the KL-divergence Triton backward benchmark.

Operator: KL-divergence loss (batchmean reduction, non-log target), matching Liger's kl_div kernel.
Forward:  loss = sum_ij y_true_ij * (log y_true_ij - y_pred_ij) / BT   -> scalar
Backward: d_input = dloss * (-y_true) / BT     (only wrt the input log-probs y_pred)

Standalone backward API:
    kl_div_backward_triton(dloss, y_pred, y_true) -> d_input
Autograd-pair API:
    kl_div_forward_with_saved(y_pred, y_true) -> (loss, saved_tensors)
    kl_div_backward_from_saved(dloss, saved_tensors) -> d_input

Notes:
  - `y_pred` is log-probabilities (log_softmax); `y_true` is probabilities (softmax, rows sum to 1).
  - The cotangent `dloss` is a SCALAR (batchmean loss is scalar).
  - Backward is ONLY wrt y_pred; y_true is the (float) target and is threaded through saved_tensors.
    `d_input = -y_true / BT` is a pure element-wise map (no reduction) — the forward, however, has a
    per-row reduction over V, so we still use V=cols as the regime axis and let dispatch decide.
"""

from dataclasses import dataclass


CANDIDATE_FN_NAME = "kl_div_backward_triton"
OUTPUT_NAMES = ("d_input",)
BT_EPS = 1e-10

AUTOGRAD_PAIR_FORWARD_FN_NAME = "kl_div_forward_with_saved"
AUTOGRAD_PAIR_BACKWARD_FN_NAME = "kl_div_backward_from_saved"
# make_inputs returns (dloss, y_pred, y_true)
AUTOGRAD_PAIR_COTANGENT_INDEX = 0             # dloss (scalar)
AUTOGRAD_PAIR_FORWARD_INPUT_INDICES = (1, 2)  # y_pred, y_true
AUTOGRAD_PAIR_BACKWARD_EXTRA_INPUT_INDICES = ()  # y_true carried in saved_tensors
AUTOGRAD_PAIR_MEMORY_INPUT_INDICES = (1,)     # y_pred defines the memory budget

AUTOGRAD_PAIR_API = """def kl_div_forward_with_saved(y_pred, y_true):
    return loss, saved_tensors

def kl_div_backward_from_saved(dloss, saved_tensors):
    return d_input
"""

AUTOGRAD_PAIR_TASK_CONTEXT = """Forward is batchmean KL divergence (non-log target):
    y_pred : [BT, V] log-probabilities, y_true : [BT, V] probabilities (rows sum to 1)
    loss   = sum_ij y_true_ij * (log(y_true_ij) - y_pred_ij) / BT      (scalar)

Backward returns d_input (same shape as y_pred), ONLY wrt the input log-probs y_pred:
    d_input_ij = dloss * (-y_true_ij) / BT

dloss is a SCALAR cotangent. y_true is the target (do NOT differentiate it); the forward may save it
into saved_tensors for the backward. The backward is a pure element-wise scale of -y_true (no
reduction). Use fp32 for the log/reduction in the forward and preserve the input dtype in outputs.
"""


@dataclass(frozen=True)
class TestCase:
    rows: int      # BT
    cols: int      # V
    dtype_name: str
    atol_value: float
    rtol_value: float


CORRECTNESS_CASES = [
    TestCase(8, 512, "float32", 1e-6, 1e-4),
    TestCase(16, 1024, "float32", 1e-6, 1e-4),
    TestCase(32, 4096, "float16", 1e-4, 1e-2),
    TestCase(64, 2048, "float16", 1e-4, 1e-2),
    TestCase(32, 4096, "bfloat16", 5e-4, 2e-2),
    TestCase(64, 2048, "bfloat16", 5e-4, 2e-2),
]

# Regime axis = cols (V). Sweep V from small to large (llama3 vocab 128k) at a few BT, plus non-pow2.
_KLDIV_BENCHMARK_SHAPES = [
    # small V (< REGIME_SPLIT)
    (4096, 512),
    (8192, 512),
    (4096, 4096),
    (16384, 4096),
    (4096, 8192),
    (2048, 8192),
    # large V (>= REGIME_SPLIT)
    (4096, 16384),
    (512, 32000),
    (2048, 32000),
    (4096, 32000),
    (4096, 65536),
    (4096, 128256),
    (777, 50257),
]

_BENCHMARK_DTYPE_TOLERANCES = {
    "float32": (1e-6, 1e-4),
    "float16": (1e-4, 1e-2),
    "bfloat16": (5e-4, 2e-2),
}


def _make_benchmark_cases() -> list[TestCase]:
    cases: list[TestCase] = []
    for rows, cols in _KLDIV_BENCHMARK_SHAPES:
        for dtype_name, (atol_v, rtol_v) in _BENCHMARK_DTYPE_TOLERANCES.items():
            cases.append(TestCase(rows, cols, dtype_name, atol_v, rtol_v))
    return cases


BENCHMARK_CASES = _make_benchmark_cases()


# --- Shape-regime metadata ---------------------------------------------------------------------
# KL-div forward has a per-row reduction over V (for the scalar loss); the backward is pure
# element-wise (-y_true/BT). We use V=cols as the regime axis (the forward reduction is where any
# regime would come from) and let the dispatch tool decide whether a regime actually exists — a
# backward-dominated element-wise op may well collapse to a single program (negative control).
REGIME_SPLIT = 16384  # cols; small: < split, large: >= split


def regime_feature(case: "TestCase") -> float:
    return float(case.cols)


def case_weight(case: "TestCase") -> float:
    import math

    dist = abs(math.log2(regime_feature(case)) - math.log2(REGIME_SPLIT))
    return float(max(1.0, round(dist)))


SMALL_CASES = [c for c in BENCHMARK_CASES if regime_feature(c) < REGIME_SPLIT]
LARGE_CASES = [c for c in BENCHMARK_CASES if regime_feature(c) >= REGIME_SPLIT]


def make_liger_autograd_pair_fns():
    """(forward_with_saved, backward_from_saved) from Liger's raw kldiv_forward/backward."""
    try:
        from benchmark.triton_kl_div_backward_bench.strong_baselines.liger_kl_div import (
            make_liger_kl_div_autograd_pair_fns,
        )
    except ImportError:
        from strong_baselines.liger_kl_div import make_liger_kl_div_autograd_pair_fns
    return make_liger_kl_div_autograd_pair_fns()


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
    return {"shape": [case.rows, case.cols], "dtype": case.dtype_name, "reduction": "batchmean"}


def make_inputs(torch_module, case: TestCase):
    """Return (dloss, y_pred, y_true): scalar cotangent, [BT,V] log-probs, [BT,V] probs."""
    dtype = _dtype(torch_module, case.dtype_name)
    logits_p = torch_module.randn((case.rows, case.cols), device="cuda", dtype=torch_module.float32)
    logits_t = torch_module.randn((case.rows, case.cols), device="cuda", dtype=torch_module.float32)
    y_pred = torch_module.log_softmax(logits_p, dim=-1).to(dtype)   # log-probabilities
    y_true = torch_module.softmax(logits_t, dim=-1).to(dtype)       # probabilities
    dloss = (torch_module.rand((), device="cuda", dtype=dtype) + 0.5)
    return dloss, y_pred, y_true


def torch_oracle(torch_module, dloss, y_pred, y_true):
    """PyTorch autograd backward wrt y_pred only: returns d_input (in y_pred dtype)."""
    yp = y_pred.detach().clone().float().requires_grad_(True)
    loss = torch_module.nn.functional.kl_div(yp, y_true.float(), reduction="batchmean", log_target=False)
    loss.backward(dloss.float())
    return yp.grad.to(y_pred.dtype)


def autograd_pair_forward_oracle(torch_module, y_pred, y_true):
    """Reference forward output (scalar loss)."""
    loss = torch_module.nn.functional.kl_div(
        y_pred.float(), y_true.float(), reduction="batchmean", log_target=False
    )
    return loss.to(y_pred.dtype)


def atol(case: TestCase, output_name: str) -> float:
    return case.atol_value


def rtol(case: TestCase, output_name: str) -> float:
    return case.rtol_value


def correctness_hint() -> str:
    return "d_input = dloss * (-y_true) / BT   (batchmean KL, backward only wrt log-prob input)"
