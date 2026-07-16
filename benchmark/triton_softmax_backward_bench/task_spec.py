"""Task spec for the Softmax Triton backward benchmark.

Operator: row-wise Softmax over the last dim, matching Liger's softmax kernel.
Forward:  y = softmax(x, dim=-1)
Backward: dx = y * (dy - sum(dy * y, dim=-1, keepdim=True))     (row reduction over cols)

Standalone backward API:
    softmax_backward_triton(dy, y) -> dx
Autograd-pair API:
    softmax_forward_with_saved(x) -> (y, saved_tensors)
    softmax_backward_from_saved(dy, saved_tensors) -> dx

Like Cross-Entropy, this is a per-row reduction over the last dim, so the optimal kernel
structure changes with the reduction width V=cols (single-pass vs streaming). Unlike CE, the
inputs are all floating point and the cotangent dy has the same shape as y. Liger itself already
dispatches single-block vs multi-block by cols, so it is a fairly strong baseline. Liger's
backward is NOT in-place (it reads y+dy and writes a fresh dx).
"""

from dataclasses import dataclass


CANDIDATE_FN_NAME = "softmax_backward_triton"
OUTPUT_NAMES = ("dx",)

AUTOGRAD_PAIR_FORWARD_FN_NAME = "softmax_forward_with_saved"
AUTOGRAD_PAIR_BACKWARD_FN_NAME = "softmax_backward_from_saved"
# make_inputs returns (dy, x)
AUTOGRAD_PAIR_COTANGENT_INDEX = 0            # dy (same shape as y)
AUTOGRAD_PAIR_FORWARD_INPUT_INDICES = (1,)   # x
AUTOGRAD_PAIR_BACKWARD_EXTRA_INPUT_INDICES = ()  # y is carried in saved_tensors
AUTOGRAD_PAIR_MEMORY_INPUT_INDICES = (1,)    # x defines the memory budget

AUTOGRAD_PAIR_API = """def softmax_forward_with_saved(x):
    return y, saved_tensors

def softmax_backward_from_saved(dy, saved_tensors):
    return dx
"""

AUTOGRAD_PAIR_TASK_CONTEXT = """Forward is numerically-stable row-wise softmax over the last dim:
    m_i   = max(x_i)                 # over cols
    e_i   = exp(x_i - m_i)
    y_i   = e_i / sum(e_i)           # softmax row

Backward returns dx (same shape as x):
    s_i    = sum(dy_i * y_i)         # per-row reduction over cols
    dx_i   = y_i * (dy_i - s_i)

The forward output y is the natural saved tensor for the backward (dx depends only on y and dy).
Use fp32 for the max/sum reductions and the exp, and preserve the input dtype in the output. The
saved-tensor contract is flexible (you may save y, or x, etc.).
"""


@dataclass(frozen=True)
class TestCase:
    rows: int
    cols: int      # V = softmax reduction width (last dim)
    dtype_name: str
    atol_value: float
    rtol_value: float


CORRECTNESS_CASES = [
    TestCase(8, 512, "float32", 2e-5, 2e-5),
    TestCase(16, 1024, "float32", 2e-5, 2e-5),
    TestCase(32, 4096, "float16", 5e-2, 5e-2),
    TestCase(64, 2048, "float16", 5e-2, 5e-2),
    TestCase(32, 4096, "bfloat16", 8e-2, 8e-2),
    TestCase(64, 2048, "bfloat16", 8e-2, 8e-2),
]

# Regime axis = cols (V, the softmax reduction width). Sweep cols from small (single-pass) to
# large (>65536, where Liger itself switches to a multi-block kernel), at a few row counts.
_SOFTMAX_BENCHMARK_SHAPES = [
    # small cols (< REGIME_SPLIT): single-pass regime
    (8192, 512),
    (4096, 512),
    (4096, 4096),
    (16384, 4096),
    (4096, 8192),
    (2048, 8192),
    # large cols (>= REGIME_SPLIT): wide reduction / streaming regime
    (4096, 16384),
    (2048, 16384),
    (4096, 32768),
    (2048, 65536),
    (4096, 65536),
    (1024, 131072),
]

_BENCHMARK_DTYPE_TOLERANCES = {
    "float32": (2e-5, 2e-5),
    "float16": (5e-2, 5e-2),
    "bfloat16": (8e-2, 8e-2),
}


def _make_benchmark_cases() -> list[TestCase]:
    cases: list[TestCase] = []
    for rows, cols in _SOFTMAX_BENCHMARK_SHAPES:
        for dtype_name, (atol_v, rtol_v) in _BENCHMARK_DTYPE_TOLERANCES.items():
            cases.append(TestCase(rows, cols, dtype_name, atol_v, rtol_v))
    return cases


BENCHMARK_CASES = _make_benchmark_cases()


# --- Shape-regime metadata (for shape-aware evolution / dispatch) ---------------
# Softmax backward is embarrassingly parallel across rows with a per-row reduction over cols, so
# the optimal kernel structure changes with the reduction width cols (fits one block vs needs
# streaming), NOT with row count. REGIME_SPLIT is the rule-of-thumb cut for the small/large
# TRAINING suites; the deployment threshold is derived afterward. (Liger switches its own
# single/multi-block kernel around cols=65536.)
REGIME_SPLIT = 16384  # cols; small: < split (single-pass), large: >= split (wide reduction)


def regime_feature(case: "TestCase") -> float:
    """Scalar that determines which shape-regime a case falls in (softmax width, cols)."""
    return float(case.cols)


def case_weight(case: "TestCase") -> float:
    """Per-case weight for the weighted-geomean score. log2-distance from REGIME_SPLIT."""
    import math

    dist = abs(math.log2(regime_feature(case)) - math.log2(REGIME_SPLIT))
    return float(max(1.0, round(dist)))


SMALL_CASES = [c for c in BENCHMARK_CASES if regime_feature(c) < REGIME_SPLIT]
LARGE_CASES = [c for c in BENCHMARK_CASES if regime_feature(c) >= REGIME_SPLIT]


def make_liger_autograd_pair_fns():
    """(forward_with_saved, backward_from_saved) built from Liger's raw softmax forward/backward.
    Used as the perf baseline when AUTOGRAD_PAIR_PERF_BASELINE=liger."""
    try:
        from benchmark.triton_softmax_backward_bench.strong_baselines.liger_softmax import (
            make_liger_softmax_autograd_pair_fns,
        )
    except ImportError:
        from strong_baselines.liger_softmax import make_liger_softmax_autograd_pair_fns
    return make_liger_softmax_autograd_pair_fns()


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
    return {"shape": [case.rows, case.cols], "dtype": case.dtype_name}


def make_inputs(torch_module, case: TestCase):
    """Return (dy, x): cotangent dy and input x, both [rows, cols] on CUDA."""
    dtype = _dtype(torch_module, case.dtype_name)
    x = torch_module.randn((case.rows, case.cols), device="cuda", dtype=dtype)
    dy = torch_module.randn((case.rows, case.cols), device="cuda", dtype=dtype)
    return dy, x


def torch_oracle(torch_module, dy, x):
    """PyTorch autograd backward: returns dx (in x dtype)."""
    xr = x.detach().clone().float().requires_grad_(True)
    y = torch_module.softmax(xr, dim=-1)
    y.backward(dy.float())
    return xr.grad.to(x.dtype)


def autograd_pair_forward_oracle(torch_module, x):
    """Reference forward output y for correctness checking of the autograd-pair."""
    return torch_module.softmax(x.float(), dim=-1).to(x.dtype)


def atol(case: TestCase, output_name: str) -> float:
    return case.atol_value


def rtol(case: TestCase, output_name: str) -> float:
    return case.rtol_value


def correctness_hint() -> str:
    return "y=softmax(x,-1); s=sum(dy*y,-1,keepdim); dx=y*(dy-s)"
