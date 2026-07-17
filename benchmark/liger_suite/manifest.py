"""Manifest of the Liger benchmark suite: which ops, which naive forward, which Liger sources.

Each entry builds ONE benchmark case (taskspec + prep only — spec, oracles, evaluators and the
gated Liger baseline wrapper). Running seed/evolve/dispatch on a case is the *user* of the
benchmark's job, not the suite builder's.

Scope: ops that fit the current 2D (rows x cols) TestCase contract. The rope / attention /
fused_linear families need a case-schema generalization first and are deliberately absent.
"""

from dataclasses import dataclass, field
from pathlib import Path

SUITE_DIR = Path(__file__).resolve().parent
FORWARDS = SUITE_DIR / "forwards"


@dataclass(frozen=True)
class SuiteOp:
    op: str                         # names the bench dir: benchmark/triton_<op>_backward_bench
    forward_file: str               # file under forwards/ (single top-level function)
    liger_modules: tuple[str, ...]  # liger_kernel.ops module names to pass as --liger-source
    notes: str = ""


SUITE_OPS: list[SuiteOp] = [
    SuiteOp("dyt", "dyt.py", ("dyt", "utils"),
            "y = gamma*tanh(alpha*x)+beta; HAVE_BETA=True; all 4 inputs differentiable"),
    SuiteOp("relu_squared", "relu_squared.py", ("relu_squared", "utils"),
            "elementwise relu(x)^2, no params"),
    SuiteOp("sparsemax", "sparsemax.py", ("sparsemax", "utils"),
            "simplex projection, dim=-1 pinned"),
    SuiteOp("tvd", "tvd.py", ("tvd", "utils"),
            "0.5*|p-q| batchmean; q is target (no grad), matches kl_div precedent"),
    SuiteOp("jsd", "jsd.py", ("jsd", "utils"),
            "generalized JSD, beta=0.5, log-space inputs; grad wrt student log_q only"),
    SuiteOp("poly_norm", "poly_norm.py", ("poly_norm", "utils"),
            "w0*norm(x^3)+w1*norm(x^2)+w2*norm(x)+b, eps=1e-6"),
    SuiteOp("fused_add_rms_norm", "fused_add_rms_norm.py", ("fused_add_rms_norm", "utils"),
            "single-output form of Liger's two-output (y, s) kernel; offset=0, llama casting; "
            "wrapper may need hand-written fallback"),
]


def liger_source_paths(entry: SuiteOp) -> tuple[str, ...]:
    """Resolve the entry's Liger module names to installed file paths (no hardcoded env paths)."""
    import liger_kernel.ops
    ops_dir = Path(liger_kernel.ops.__file__).parent
    return tuple(str(ops_dir / f"{m}.py") for m in entry.liger_modules)
