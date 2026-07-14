"""
Shared NCU report utilities for Triton kernels.

Key differences from ncu-report-skill/helpers/ncu_utils.py:
  - load_all_actions() iterates ALL kernel launches in the report, not just the first.
    A Triton autograd pair typically produces 2-6 separate kernel launches per benchmark call.
  - No CLI interface — all functions are called directly from Python.
  - Source-line mapping (action.source_info) is intentionally omitted: for Triton kernels,
    NCU maps PCs to compiled IR/PTX temp files, not to the original Python source.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Optional


# ---------------------------------------------------------------------------
# Locate the ncu_report Python module shipped with Nsight Compute
# ---------------------------------------------------------------------------

def _locate_ncu_report() -> Optional[str]:
    candidates = [
        "/usr/local/cuda-13.2/nsight-compute-2026.1.0/extras/python",
        "/usr/local/cuda/nsight-compute/extras/python",
    ]
    for root in ["/usr/local", "/opt/nvidia", "/opt/cuda"]:
        p = Path(root)
        if not p.is_dir():
            continue
        for sub in p.glob("cuda-*/nsight-compute-*/extras/python"):
            candidates.append(str(sub))
        for sub in p.glob("nsight-compute-*/extras/python"):
            candidates.append(str(sub))
    for c in candidates:
        if Path(c).is_dir() and (Path(c) / "ncu_report.py").exists():
            return c
    return None


try:
    import ncu_report  # type: ignore
except ImportError:
    found = _locate_ncu_report()
    if found:
        sys.path.insert(0, found)
        import ncu_report  # type: ignore  # noqa: F811
    else:
        raise ImportError(
            "ncu_report module not found. Add the Nsight Compute extras/python directory "
            "to PYTHONPATH, e.g.:\n"
            "  export PYTHONPATH=$PYTHONPATH:/usr/local/cuda/nsight-compute/extras/python"
        )


# ---------------------------------------------------------------------------
# Report loading — all kernels, not just the first
# ---------------------------------------------------------------------------

def load_all_actions(report_path: str | Path) -> tuple[Any, list[tuple[str, Any]]]:
    """Load a .ncu-rep file and return ALL kernel launch actions.

    Returns (report, [(kernel_name, action), ...]).
    The report object must stay alive as long as the actions are used.

    For a Triton autograd pair, the report typically contains one action per
    Triton JIT function called during the benchmark (e.g. fwd kernel, bwd dx
    kernel, bwd dwdb kernel, final reduction kernel).
    """
    report = ncu_report.load_report(str(report_path))
    actions: list[tuple[str, Any]] = []

    for rng in _iter_indexed(report, "range_by_idx", "num_ranges"):
        for action in _iter_indexed(rng, "action_by_idx", "num_actions"):
            actions.append((action.name(), action))

    return report, actions


def _iter_indexed(obj: Any, by_idx_name: str, count_name: str):
    """Yield obj.<by_idx_name>(i) for i = 0, 1, ...

    Uses obj.<count_name>() when available. The probing fallback MUST treat a
    None return as end-of-sequence: ncu_report returns None (rather than
    raising) for out-of-range indices, so a bare while/try loop spins forever.
    """
    by_idx = getattr(obj, by_idx_name)
    if hasattr(obj, count_name):
        try:
            n = getattr(obj, count_name)()
            for i in range(n):
                item = by_idx(i)
                if item is not None:
                    yield item
            return
        except Exception:
            pass
    i = 0
    while True:
        try:
            item = by_idx(i)
        except Exception:
            return
        if item is None:
            return
        yield item
        i += 1


# ---------------------------------------------------------------------------
# Safe metric access
# ---------------------------------------------------------------------------

def safe(action: Any, name: str, default: Any = None) -> Any:
    """Return a metric value, or `default` if the metric is missing or errors."""
    try:
        return action[name].value()
    except Exception:
        return default


# ---------------------------------------------------------------------------
# NCU rule engine
# ---------------------------------------------------------------------------

def rule_speedups(action: Any) -> list[tuple[float, str, str]]:
    """Return [(est_speedup_pct, rule_name, message), ...] sorted desc by speedup."""
    try:
        results = list(action.rule_results_as_dicts())
    except Exception:
        return []
    out = []
    for rr in results:
        try:
            est = float(rr.get("estimated_speedup_pct") or 0)
        except Exception:
            est = 0.0
        rule = rr.get("rule_name", "?")
        msg = rr.get("message_for_display", "?")
        out.append((est, rule, msg))
    out.sort(key=lambda x: -x[0])
    return out


# ---------------------------------------------------------------------------
# Curated metric set (B200 / sm_100 verified names)
# ---------------------------------------------------------------------------

B200_KEY_METRICS: list[str] = [
    # Launch geometry
    "launch__grid_size",
    "launch__block_size",
    "launch__waves_per_multiprocessor",
    "launch__registers_per_thread",
    "launch__shared_mem_per_block",
    "launch__occupancy_limit_blocks",
    "launch__occupancy_limit_registers",
    "launch__occupancy_limit_shared_mem",
    "launch__occupancy_limit_warps",
    "device__attribute_multiprocessor_count",
    # Timing
    "gpu__time_duration.sum",
    # Speed-of-light
    "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed",
    # Occupancy
    "sm__maximum_warps_per_active_cycle_pct",
    "sm__warps_active.avg.pct_of_peak_sustained_active",
    # Compute pipes
    "sm__inst_executed_pipe_fma.avg.pct_of_peak_sustained_elapsed",
    # Tensor core
    "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed",
    "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active",
    # DRAM
    "dram__bytes_read.sum",
    "dram__bytes_read.sum.pct_of_peak_sustained_elapsed",
    "dram__bytes_write.sum",
    "dram__bytes_write.sum.pct_of_peak_sustained_elapsed",
    # L1 / L2 caches
    "l1tex__t_sector_hit_rate.pct",
    "lts__t_sector_hit_rate.pct",
    # Memory instruction counts (spill detection)
    "smsp__sass_inst_executed_op_local_ld.sum",
    "smsp__sass_inst_executed_op_local_st.sum",
    "smsp__sass_inst_executed_op_shared.sum",
    # Coalescing
    "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum",
    "l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum",
    "smsp__sass_average_data_bytes_per_sector_mem_global_op_st.ratio",
    # Stall ratios (aggregate — valid for Triton)
    "smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_wait_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_math_pipe_throttle_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_mio_throttle_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_not_selected_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_membar_per_issue_active.ratio",
]
