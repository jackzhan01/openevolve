"""Task spec for the LayerNorm Triton backward benchmark."""

import os
from dataclasses import dataclass


CANDIDATE_FN_NAME = "layernorm_backward_triton"
OUTPUT_NAMES = ("dx", "dweight", "dbias")
EPS = 1e-5
# OpenEvolve speedup uses torch_oracle (PyTorch autograd). Override with BASELINE_PROGRAM_PATH
# only for legacy ablations against a fixed Triton seed program.
PERFORMANCE_BASELINE = "pytorch_autograd"


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

# Aligned with tests/atenir_correctness/run_correctness.py SHAPE_PRESETS["layernorm"].
_LAYERNORM_BENCHMARK_SHAPES_MIXED = [
    # dynamic
    (1, 768),
    (8, 1024),
    (32, 1536),
    (8, 4096),
    (1, 8192),
    # nontile
    (17, 127),
    (17, 513),
    (17, 1000),
]

# Small-batch / small-hidden-dim shapes: latency-dominated, fixed-cost sensitive.
_SMALL_BENCHMARK_SHAPES = [
    (1, 256),
    (4, 512),
    (8, 768),
    (17, 127),
]

# Large-batch / large-hidden-dim shapes: throughput-dominated, bandwidth sensitive.
_LARGE_BENCHMARK_SHAPES = [
    (32, 1536),
    (8, 4096),
    (1, 8192),
    (64, 8192),
]

_LAYERNORM_BENCHMARK_SHAPES_SMALL = [
    (1, 768),
    (8, 1024),
    (17, 127),
    (17, 513),
    (17, 1000),
]

_LAYERNORM_BENCHMARK_SHAPES_LARGE = [
    (1024, 1024),
    (4096, 1024),
    (16384, 1024),
    (65536, 1024),
    (131072, 1024),
]


def _tritonbench_layernorm_shape(i: int) -> tuple[int, int]:
    size = 2**i
    return size // 1024, 1024


_LAYERNORM_BENCHMARK_SHAPES_TB_SWEEP = [
    _tritonbench_layernorm_shape(i)
    for i in range(12, 28)
]

_LAYERNORM_BENCHMARK_SHAPES_TB_SMALL = [
    _tritonbench_layernorm_shape(i)
    for i in range(12, 17)
]

_LAYERNORM_BENCHMARK_SHAPES_TB_MEDIUM = [
    _tritonbench_layernorm_shape(i)
    for i in range(17, 23)
]

_LAYERNORM_BENCHMARK_SHAPES_TB_LARGE = [
    _tritonbench_layernorm_shape(i)
    for i in range(23, 28)
]

_LAYERNORM_BENCHMARK_SHAPES_TB_MIXED = [
    _tritonbench_layernorm_shape(i)
    for i in (12, 14, 16, 18, 20, 22, 24, 26, 27)
]

_LAYERNORM_BENCHMARK_SHAPE_SUITES = {
    "mixed": _LAYERNORM_BENCHMARK_SHAPES_MIXED,
    "all": _LAYERNORM_BENCHMARK_SHAPES_MIXED,
    "small": _LAYERNORM_BENCHMARK_SHAPES_SMALL,
    "large": _LAYERNORM_BENCHMARK_SHAPES_LARGE,
    "legacy_small": _SMALL_BENCHMARK_SHAPES,
    "legacy_large": _LARGE_BENCHMARK_SHAPES,
    "tb_small": _LAYERNORM_BENCHMARK_SHAPES_TB_SMALL,
    "tb_medium": _LAYERNORM_BENCHMARK_SHAPES_TB_MEDIUM,
    "tb_large": _LAYERNORM_BENCHMARK_SHAPES_TB_LARGE,
    "tb_mixed": _LAYERNORM_BENCHMARK_SHAPES_TB_MIXED,
    "tb_sweep": _LAYERNORM_BENCHMARK_SHAPES_TB_SWEEP,
    **{
        f"tb_i{i}": [_tritonbench_layernorm_shape(i)]
        for i in range(12, 28)
    },
}

_BENCHMARK_DTYPE_TOLERANCES = {
    "float32": (2e-5, 2e-5),
    "float16": (5e-2, 5e-2),
    "bfloat16": (8e-2, 8e-2),
}


def benchmark_suite_name() -> str:
    suite = os.environ.get(
        "LAYERNORM_BENCHMARK_SUITE",
        os.environ.get("AUTOGRAD_PAIR_SHAPE_PROFILE", "mixed"),
    ).strip().lower()
    if suite not in _LAYERNORM_BENCHMARK_SHAPE_SUITES:
        valid = ", ".join(sorted(_LAYERNORM_BENCHMARK_SHAPE_SUITES))
        raise ValueError(f"Unsupported LAYERNORM_BENCHMARK_SUITE={suite!r}; expected one of: {valid}")
    return suite


def benchmark_shapes() -> list[tuple[int, int]]:
    return list(_LAYERNORM_BENCHMARK_SHAPE_SUITES[benchmark_suite_name()])


def benchmark_dtype_names() -> list[str]:
    raw = os.environ.get("LAYERNORM_BENCHMARK_DTYPES", "float32,float16,bfloat16")
    aliases = {"bf16": "bfloat16", "fp32": "float32", "fp16": "float16"}
    dtypes = [aliases.get(part.strip().lower(), part.strip().lower()) for part in raw.split(",") if part.strip()]
    if not dtypes:
        raise ValueError("LAYERNORM_BENCHMARK_DTYPES must include at least one dtype")
    invalid = [dtype for dtype in dtypes if dtype not in _BENCHMARK_DTYPE_TOLERANCES]
    if invalid:
        valid = ", ".join(sorted(_BENCHMARK_DTYPE_TOLERANCES))
        raise ValueError(f"Unsupported LAYERNORM_BENCHMARK_DTYPES values {invalid!r}; expected any of: {valid}")
    return dtypes


def _make_benchmark_cases() -> list[TestCase]:
    cases: list[TestCase] = []
    for rows, cols in benchmark_shapes():
        for dtype_name in benchmark_dtype_names():
            atol_value, rtol_value = _BENCHMARK_DTYPE_TOLERANCES[dtype_name]
            cases.append(TestCase(rows, cols, dtype_name, atol_value, rtol_value))
    return cases


BENCHMARK_CASES = _make_benchmark_cases()


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
        "eps": EPS,
    }


def make_inputs(torch_module, case: TestCase):
    dtype = _dtype(torch_module, case.dtype_name)
    x = torch_module.randn((case.rows, case.cols), device="cuda", dtype=dtype)
    weight = torch_module.randn((case.cols,), device="cuda", dtype=dtype)
    bias = torch_module.randn((case.cols,), device="cuda", dtype=dtype)
    dy = torch_module.randn((case.rows, case.cols), device="cuda", dtype=dtype)
    return dy, x, weight, bias, EPS


def torch_oracle(torch_module, dy, x, weight, bias, eps):
    x_ref = x.detach().clone().requires_grad_(True)
    weight_ref = weight.detach().clone().requires_grad_(True)
    bias_ref = bias.detach().clone().requires_grad_(True)
    mean = torch_module.mean(x_ref.float(), dim=-1, keepdim=True)
    var = torch_module.mean((x_ref.float() - mean) ** 2, dim=-1, keepdim=True)
    xhat = (x_ref.float() - mean) * torch_module.rsqrt(var + eps)
    y_ref = (xhat * weight_ref.float() + bias_ref.float()).to(x.dtype)
    y_ref.backward(dy.detach().clone())
    return x_ref.grad, weight_ref.grad, bias_ref.grad


def atol(case: TestCase, output_name: str) -> float:
    return case.atol_value


def rtol(case: TestCase, output_name: str) -> float:
    return case.rtol_value


def correctness_hint() -> str:
    return (
        "xhat=(x-mean)/sqrt(var+eps), one=dy*weight, "
        "dx=(one-mean(one)-xhat*mean(xhat*one))*rstd, "
        "dweight=sum(dy*xhat, dim=0), dbias=sum(dy, dim=0)"
    )
