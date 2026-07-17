"""
Deterministic roofline triage for the NCU optimizer pass (Stage 0).

Classifies a kernel's bottleneck from NCU Speed-of-Light metrics before any
LLM call:

  - the HIGHER SOL value names the bottleneck (the resource nearest its peak)
  - both SOLs below the underutilized threshold -> "underutilized"
    (occupancy / stalls / launch config, not a saturated resource)
  - efficiency = max(compute SOL, memory SOL); at/above the roofline
    threshold there is nothing left for an optimizer to win

Inputs are the evaluator's pre-extracted ncu_* scalars, so this runs on data
the pass already has — no extra profiling, no LLM.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class RooflineResult:
    compute_sol_pct: float
    memory_sol_pct: float
    efficiency_pct: float
    headroom_pct: float
    bottleneck: str  # "memory" | "compute" | "underutilized" | "unknown"
    at_roofline: bool
    occupancy_pct: Optional[float] = None
    long_scoreboard_stall_ratio: Optional[float] = None
    warnings: list[str] = field(default_factory=list)


def analyze(
    ncu_metrics: Dict[str, Any],
    *,
    roofline_threshold_pct: float = 95.0,
    underutilized_threshold_pct: float = 60.0,
) -> RooflineResult:
    """Classify the bottleneck from the evaluator's ncu_* scalar metrics."""
    warnings: list[str] = []

    compute_sol = ncu_metrics.get("ncu_sm_throughput_pct")
    memory_sol = ncu_metrics.get("ncu_dram_throughput_pct")

    if compute_sol is None or compute_sol < 0:
        warnings.append("compute SOL (ncu_sm_throughput_pct) missing")
        compute_sol = 0.0
    if memory_sol is None or memory_sol < 0:
        warnings.append("memory SOL (ncu_dram_throughput_pct) missing")
        memory_sol = 0.0

    if not warnings or len(warnings) < 2:
        if memory_sol < underutilized_threshold_pct and compute_sol < underutilized_threshold_pct:
            bottleneck = "underutilized"
        elif memory_sol >= compute_sol:
            bottleneck = "memory"
        else:
            bottleneck = "compute"
    else:
        bottleneck = "unknown"

    efficiency = max(compute_sol, memory_sol)

    occ = ncu_metrics.get("ncu_occupancy_pct")
    lsb = ncu_metrics.get("ncu_long_scoreboard_stall_ratio")

    return RooflineResult(
        compute_sol_pct=float(compute_sol),
        memory_sol_pct=float(memory_sol),
        efficiency_pct=float(efficiency),
        headroom_pct=max(0.0, 100.0 - float(efficiency)),
        bottleneck=bottleneck,
        at_roofline=efficiency >= roofline_threshold_pct,
        occupancy_pct=float(occ) if isinstance(occ, (int, float)) and occ >= 0 else None,
        long_scoreboard_stall_ratio=(
            float(lsb) if isinstance(lsb, (int, float)) and lsb >= 0 else None
        ),
        warnings=warnings,
    )


def format_roofline(result: RooflineResult) -> str:
    """Human-readable triage summary for prompts and pass records."""
    lines = [
        f"- Primary bottleneck class: {result.bottleneck.upper()}",
        f"- Compute SOL (SM throughput): {result.compute_sol_pct:.1f}% of peak",
        f"- Memory SOL (DRAM throughput): {result.memory_sol_pct:.1f}% of peak",
        f"- Efficiency: {result.efficiency_pct:.1f}%  (headroom: {result.headroom_pct:.1f}%)",
        f"- At roofline: {'yes' if result.at_roofline else 'no'}",
    ]
    if result.occupancy_pct is not None:
        lines.append(f"- Achieved occupancy: {result.occupancy_pct:.1f}%")
    if result.long_scoreboard_stall_ratio is not None:
        lines.append(
            f"- Long-scoreboard stall ratio: {result.long_scoreboard_stall_ratio:.2f}"
        )
    if result.warnings:
        lines.append(f"- Warnings: {'; '.join(result.warnings)}")
    return "\n".join(lines)
