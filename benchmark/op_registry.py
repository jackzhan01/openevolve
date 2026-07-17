"""Single queryable registry of every benchmark op — no central list to hand-maintain.

Two signals are discovered and merged:

  * BUILT cases   — any `benchmark/triton_<op>_backward_bench/` that has a `task_spec.py`.
                    Its baseline is `liger` if `strong_baselines/liger_<op>.py` exists, else
                    `pytorch_autograd`; its forward is `forward_ref.py` when present.
  * BUILDABLE ops — declared in `benchmark/liger_suite/manifest.py` (a hand-written naive
                    forward + Liger source mapping), whether or not they have been built yet.

This is PURELY ADDITIVE: it is a lookup shortcut, never a gate. `get_op` on an unknown name
raises with the available list plus the reminder that any ad-hoc op still runs by passing
`--forward`/`--bench-dir` explicitly (the original contract).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

BENCH_ROOT = Path(__file__).resolve().parent
_PREFIX, _SUFFIX = "triton_", "_backward_bench"


@dataclass(frozen=True)
class OpInfo:
    name: str
    bench_dir: Path
    baseline: str                       # "liger" | "pytorch_autograd"
    built: bool                         # task_spec.py exists
    forward: str | None = None          # "path.py:fn" if a forward is known
    liger_sources: tuple[str, ...] = () # resolved Liger source paths, if any


def _op_from_dir(name: str) -> str:
    return name[len(_PREFIX):-len(_SUFFIX)]


def discover_ops() -> dict[str, OpInfo]:
    ops: dict[str, OpInfo] = {}

    # 1) built cases on disk
    for d in sorted(BENCH_ROOT.glob(f"{_PREFIX}*{_SUFFIX}")):
        if not (d / "task_spec.py").exists():
            continue
        op = _op_from_dir(d.name)
        liger = d / "strong_baselines" / f"liger_{op}.py"
        fwd = d / "forward_ref.py"
        ops[op] = OpInfo(
            name=op, bench_dir=d,
            baseline="liger" if liger.exists() else "pytorch_autograd",
            built=True,
            forward=f"{fwd}:{op}_forward_ref" if fwd.exists() else None,
        )

    # 2) buildable ops from the suite manifest (fills in forward + liger sources; adds any
    #    declared-but-not-yet-built op). Import lazily so the registry works even if the suite
    #    package is absent.
    try:
        from benchmark.liger_suite.manifest import SUITE_OPS, FORWARDS, liger_source_paths
    except Exception:
        SUITE_OPS = []

    for entry in SUITE_OPS:
        d = BENCH_ROOT / f"{_PREFIX}{entry.op}{_SUFFIX}"
        prior = ops.get(entry.op)
        ops[entry.op] = OpInfo(
            name=entry.op,
            bench_dir=prior.bench_dir if prior else d,
            baseline=prior.baseline if prior else "liger",
            built=prior.built if prior else (d / "task_spec.py").exists(),
            forward=f"{FORWARDS / entry.forward_file}:{entry.op}_forward" if not (prior and prior.forward)
                    else prior.forward,
            liger_sources=liger_source_paths(entry),
        )

    return dict(sorted(ops.items()))


def get_op(name: str) -> OpInfo:
    ops = discover_ops()
    try:
        return ops[name]
    except KeyError:
        raise KeyError(
            f"unknown op {name!r}. Known ops: {', '.join(ops)}.\n"
            f"For an op not in the registry, run it ad-hoc by passing --forward "
            f"(and --bench-dir) explicitly."
        ) from None


def list_ops() -> list[OpInfo]:
    return list(discover_ops().values())


def _main() -> int:
    rows = list_ops()
    print(f"{'op':<22}{'baseline':<18}{'built':<7}{'forward known':<14}")
    print("-" * 61)
    for o in rows:
        print(f"{o.name:<22}{o.baseline:<18}{('yes' if o.built else 'no'):<7}"
              f"{('yes' if o.forward else 'no'):<14}")
    print(f"\n{len(rows)} ops "
          f"({sum(o.built for o in rows)} built, {sum(o.baseline == 'liger' for o in rows)} liger)")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
