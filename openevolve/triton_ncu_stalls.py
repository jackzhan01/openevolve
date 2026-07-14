"""
Stall-type breakdown for Triton kernel launches.

Replaces ncu-report-skill/helpers/extract_stall_hotspots.py.
Called directly as a Python function — no CLI, no subprocess.

Why no source-line mapping:
  extract_stall_hotspots.py uses action.source_info(pc) to map stall PCs to
  (file, line). For CUDA C + -lineinfo this resolves to .cu source lines.
  For Triton kernels there is no -lineinfo equivalent at the Python level —
  NCU maps PCs to compiled .ttgir / .ptx temp files that don't correspond to
  the original Python code. The file:line column would be meaningless.

  What IS valid for Triton:
    - Aggregate stall-type ratios (long_scoreboard, barrier, math_throttle …)
      These are hardware counters that work identically regardless of kernel
      language. They tell us WHAT the warps are waiting for, which directly
      maps to the 06-triton-diagnosis-playbook patterns.

Usage:
    from openevolve.triton_ncu_stalls import extract_stall_summary
    text = extract_stall_summary("/path/to/report.ncu-rep")
"""
from __future__ import annotations

from pathlib import Path

from openevolve.triton_ncu_utils import load_all_actions, safe


# Aggregate stall ratio metrics — one value per kernel, fully valid for Triton
_STALL_RATIO_METRICS: list[tuple[str, str]] = [
    ("long_scoreboard",   "smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio"),
    ("short_scoreboard",  "smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio"),
    ("wait",              "smsp__average_warps_issue_stalled_wait_per_issue_active.ratio"),
    ("barrier",           "smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio"),
    ("math_throttle",     "smsp__average_warps_issue_stalled_math_pipe_throttle_per_issue_active.ratio"),
    ("mio_throttle",      "smsp__average_warps_issue_stalled_mio_throttle_per_issue_active.ratio"),
    ("lg_throttle",       "smsp__average_warps_issue_stalled_lg_throttle_per_issue_active.ratio"),
    ("tex_throttle",      "smsp__average_warps_issue_stalled_tex_throttle_per_issue_active.ratio"),
    ("membar",            "smsp__average_warps_issue_stalled_membar_per_issue_active.ratio"),
    ("not_selected",      "smsp__average_warps_issue_stalled_not_selected_per_issue_active.ratio"),
    ("branch_resolving",  "smsp__average_warps_issue_stalled_branch_resolving_per_issue_active.ratio"),
    ("dispatch_stall",    "smsp__average_warps_issue_stalled_dispatch_stall_per_issue_active.ratio"),
    ("drain",             "smsp__average_warps_issue_stalled_drain_per_issue_active.ratio"),
    ("sleeping",          "smsp__average_warps_issue_stalled_sleeping_per_issue_active.ratio"),
    ("misc",              "smsp__average_warps_issue_stalled_misc_per_issue_active.ratio"),
]

# Triton-specific interpretation notes for the dominant stall types
_TRITON_NOTES: dict[str, str] = {
    "long_scoreboard":  "global tl.load not returned → set num_stages > 1 or increase num_warps",
    "barrier":          "Triton pipeline-stage sync or tl.dot epilogue → reduce num_stages or merge passes",
    "math_throttle":    "FMA/tensor pipe saturated → compute-bound, check tensor core utilization",
    "mio_throttle":     "tl.atomic_add contention → reduce atomic footprint",
    "wait":             "tl.dot output latency → increase ILP or tile sizes",
    "short_scoreboard": "short dependency chain → larger tiles or more ILP",
}


def extract_stall_summary(report_path: str | Path) -> str:
    """
    Return per-kernel stall-type breakdown for all kernels in the report.

    Ratios are sorted descending; types below 1% are omitted.
    A Triton-specific note is appended for dominant stall types.
    """
    report, actions = load_all_actions(report_path)

    if not actions:
        return "(no kernel actions found in report)"

    sections: list[str] = []

    for idx, (kernel_name, action) in enumerate(actions):
        lines: list[str] = [f"=== Stall breakdown: Kernel {idx + 1} ({kernel_name}) ==="]

        stalls: list[tuple[float, str]] = []
        for short_name, metric in _STALL_RATIO_METRICS:
            v = safe(action, metric)
            if v is not None and float(v) >= 0.01:
                stalls.append((float(v), short_name))

        if not stalls:
            lines.append("  (no stall data — report may need --set full or PM sampling)")
        else:
            stalls.sort(reverse=True)
            total = sum(v for v, _ in stalls)
            lines.append(f"  {'stall type':<20} {'ratio':>8}   {'% of stalls':>12}   note")
            lines.append("  " + "-" * 80)
            for v, name in stalls:
                pct = 100 * v / total if total > 0 else 0
                note = _TRITON_NOTES.get(name, "")
                lines.append(f"  {name:<20} {v:>8.3f}   {pct:>11.1f}%   {note}")

        sections.append("\n".join(lines))

    return "\n\n".join(sections)
