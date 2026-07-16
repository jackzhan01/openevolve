"""Task spec for the GeGLU Triton backward benchmark.

Operator: GeGLU  (GELU-gated MLP activation), using the tanh approximation of GELU so it
matches the Liger strong baseline (liger_kernel.ops.geglu).
Forward:  c = gelu_tanh(a) * b
          gelu_tanh(x) = 0.5*x*(1 + tanh(sqrt(2/pi)*(x + 0.044715*x^3)))
Backward: db = dc * gelu_tanh(a)
          da = dc * b * d/da[gelu_tanh(a)]

Standalone backward API:
    geglu_backward_triton(dc, a, b) -> (da, db)

Autograd-pair API:
    geglu_forward_with_saved(a, b) -> (c, saved_tensors)
    geglu_backward_from_saved(dc, saved_tensors) -> (da, db)

GeGLU is element-wise and memory-bandwidth-bound (like SwiGLU), so it is expected to be a
single-regime op (negative control for shape dispatch): regime_feature = numel (rows*cols).
"""

from dataclasses import dataclass


CANDIDATE_FN_NAME = "geglu_backward_triton"
OUTPUT_NAMES = ("da", "db")

AUTOGRAD_PAIR_FORWARD_FN_NAME = "geglu_forward_with_saved"
AUTOGRAD_PAIR_BACKWARD_FN_NAME = "geglu_backward_from_saved"
# make_inputs returns (dc, a, b)
AUTOGRAD_PAIR_COTANGENT_INDEX = 0          # dc
AUTOGRAD_PAIR_FORWARD_INPUT_INDICES = (1, 2)   # a, b
AUTOGRAD_PAIR_BACKWARD_EXTRA_INPUT_INDICES = ()  # no eps-like extra args
AUTOGRAD_PAIR_MEMORY_INPUT_INDICES = (1, 2)    # a, b define the memory budget

AUTOGRAD_PAIR_API = """def geglu_forward_with_saved(a, b):
    return c, saved_tensors

def geglu_backward_from_saved(dc, saved_tensors):
    return da, db
"""

AUTOGRAD_PAIR_TASK_CONTEXT = """Forward is element-wise GeGLU (GELU-gated activation), tanh approximation:
    k      = sqrt(2/pi) = 0.7978845608028654
    z      = k * (a + 0.044715 * a^3)
    gelu_a = 0.5 * a * (1 + tanh(z))
    c      = gelu_a * b

Backward returns da and db:
    db = dc * gelu_a
    da = dc * b * d(gelu_a)/da
       = dc * b * ( 0.5*(1 + tanh(z))
                    + 0.5*a*(1 - tanh(z)^2) * k*(1 + 3*0.044715*a^2) )

Both inputs a and b are [rows, cols].  Use the tanh approximation of GELU (NOT the exact
erf GELU).  Use fp32 for the gelu/tanh computation and preserve input dtypes in outputs.
The saved-tensor contract is flexible: forward can save any subset of intermediates
(a, b, gelu_a, tanh(z), etc.), and backward should use exactly what was saved.
"""


@dataclass(frozen=True)
class TestCase:
    rows: int
    cols: int
    dtype_name: str
    atol_value: float
    rtol_value: float


CORRECTNESS_CASES = [
    TestCase(8, 64, "float32", 2e-5, 2e-5),
    TestCase(17, 128, "float32", 2e-5, 2e-5),
    TestCase(32, 256, "float16", 5e-2, 5e-2),
    TestCase(64, 512, "float16", 5e-2, 5e-2),
    TestCase(32, 256, "bfloat16", 8e-2, 8e-2),
    TestCase(64, 512, "bfloat16", 8e-2, 8e-2),
]

# Shapes representative of LLM FFN intermediate activations:
#   rows = flattened batch * seq_len
#   cols = intermediate_size (gate/up projection width)
_GEGLU_BENCHMARK_SHAPES = [
    # small dynamic shapes
    (1, 512),
    (8, 1024),
    (32, 2048),
    (8, 4096),
    (1, 8192),
    # mid-range shapes filling the small->large numel gap, so the regime crossover
    # (deployment threshold) can be localized with resolution (numel 262K..1.05M)
    (64, 4096),
    (128, 4096),
    (256, 4096),
    # LLaMA-scale large shapes
    (512, 4096),
    (2048, 4096),
    (512, 14336),
    (2048, 14336),
    # non-tile-aligned
    (17, 255),
    (17, 1001),
]

_BENCHMARK_DTYPE_TOLERANCES = {
    "float32": (2e-5, 2e-5),
    "float16": (5e-2, 5e-2),
    "bfloat16": (8e-2, 8e-2),
}


def _make_benchmark_cases() -> list[TestCase]:
    cases: list[TestCase] = []
    for rows, cols in _GEGLU_BENCHMARK_SHAPES:
        for dtype_name, (atol_v, rtol_v) in _BENCHMARK_DTYPE_TOLERANCES.items():
            cases.append(TestCase(rows, cols, dtype_name, atol_v, rtol_v))
    return cases


BENCHMARK_CASES = _make_benchmark_cases()


# --- Shape-regime metadata (for shape-aware evolution / dispatch) ---------------
# GeGLU is element-wise and memory-bandwidth-bound, so the regime is governed by the
# total number of elements processed (rows*cols): tiny shapes are launch-overhead bound,
# large shapes are bandwidth bound. REGIME_SPLIT is the "rule-of-thumb" cut used only to
# build the small/large TRAINING suites; the real deployment threshold is derived afterward
# from where the two trained programs' latency curves cross. (Expected outcome: like SwiGLU,
# a single regime -> dispatch collapses to one program.)
REGIME_SPLIT = 1_000_000  # numel; small: < split, large: >= split


def regime_feature(case: "TestCase") -> float:
    """Scalar that determines which shape-regime a case falls in."""
    return float(case.rows * case.cols)


def case_weight(case: "TestCase") -> float:
    """Per-case weight for the weighted-geomean score, used when training a specialist.

    Emphasizes the tail of each regime (shapes far from the split boundary) and
    de-emphasizes near-boundary shapes. Suite-agnostic log2-distance from REGIME_SPLIT.
    """
    import math

    dist = abs(math.log2(regime_feature(case)) - math.log2(REGIME_SPLIT))
    return float(max(1.0, round(dist)))


SMALL_CASES = [c for c in BENCHMARK_CASES if regime_feature(c) < REGIME_SPLIT]
LARGE_CASES = [c for c in BENCHMARK_CASES if regime_feature(c) >= REGIME_SPLIT]


def make_liger_autograd_pair_fns():
    """(forward_with_saved, backward_from_saved) built from Liger's raw geglu_forward/
    geglu_backward. Used by the evaluator core as the performance baseline when
    AUTOGRAD_PAIR_PERF_BASELINE=liger, so evolution is scored against Liger instead of
    PyTorch autograd."""
    try:
        from benchmark.triton_geglu_backward_bench.strong_baselines.liger_geglu import (
            make_liger_geglu_autograd_pair_fns,
        )
    except ImportError:
        from strong_baselines.liger_geglu import make_liger_geglu_autograd_pair_fns
    return make_liger_geglu_autograd_pair_fns()


def _dtype(torch_module, dtype_name: str):
    if dtype_name == "float32":
        return torch_module.float32
    if dtype_name == "float16":
        return torch_module.float16
    if dtype_name in ("bfloat16", "bf16"):
        return torch_module.bfloat16
    raise ValueError(f"Unsupported dtype: {dtype_name}")


def seed_for_case(case: TestCase) -> int:
    return case.rows * 100000 + case.cols


def case_metadata(case: TestCase):
    return {
        "shape": [case.rows, case.cols],
        "dtype": case.dtype_name,
    }


def make_inputs(torch_module, case: TestCase):
    """Return (dc, a, b) all shaped [rows, cols] on CUDA."""
    dtype = _dtype(torch_module, case.dtype_name)
    a = torch_module.randn((case.rows, case.cols), device="cuda", dtype=dtype)
    b = torch_module.randn((case.rows, case.cols), device="cuda", dtype=dtype)
    dc = torch_module.randn((case.rows, case.cols), device="cuda", dtype=dtype)
    return dc, a, b


def torch_oracle(torch_module, dc, a, b):
    """PyTorch autograd backward: returns (da, db). Uses tanh-approx GELU."""
    a_ref = a.detach().clone().requires_grad_(True)
    b_ref = b.detach().clone().requires_grad_(True)
    c = torch_module.nn.functional.gelu(a_ref.float(), approximate="tanh").to(a_ref.dtype) * b_ref
    c.backward(dc.detach().clone())
    return a_ref.grad, b_ref.grad


def autograd_pair_forward_oracle(torch_module, a, b):
    """Reference forward output for correctness checking of the autograd-pair."""
    return torch_module.nn.functional.gelu(a.float(), approximate="tanh").to(a.dtype) * b


def atol(case: TestCase, output_name: str) -> float:
    return case.atol_value


def rtol(case: TestCase, output_name: str) -> float:
    return case.rtol_value


def correctness_hint() -> str:
    return (
        "gelu_a=0.5*a*(1+tanh(k*(a+0.044715*a^3))), k=sqrt(2/pi); "
        "db=dc*gelu_a; da=dc*b*d(gelu_a)/da"
    )
