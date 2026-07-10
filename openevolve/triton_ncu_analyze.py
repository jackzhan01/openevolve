"""
Per-kernel metric extraction for Triton autograd pairs.

Replaces ncu-report-skill/helpers/analyze_reports.py.
Called directly as a Python function — no CLI, no subprocess, no run-dir convention.

Usage:
    from openevolve.triton_ncu_analyze import format_metrics_report
    text = format_metrics_report("/path/to/report.ncu-rep")
    # text is a multi-section string, one section per Triton kernel launch,
    # ready to paste into an LLM prompt.
"""
from __future__ import annotations

from pathlib import Path

from openevolve.triton_ncu_utils import B200_KEY_METRICS, load_all_actions, rule_speedups, safe


def format_metrics_report(report_path: str | Path) -> str:
    """
    Return a human-readable string covering ALL kernel launches in the report.

    For a Triton autograd pair with forward + several backward kernels, each kernel
    gets its own section. The LLM sees every kernel, not just the first one.
    """
    report, actions = load_all_actions(report_path)

    if not actions:
        return "(no kernel actions found in report)"

    sections: list[str] = []

    for idx, (kernel_name, action) in enumerate(actions):
        lines: list[str] = [f"=== Kernel {idx + 1}: {kernel_name} ==="]

        # Core metrics
        for m in B200_KEY_METRICS:
            v = safe(action, m)
            if v is not None:
                lines.append(f"  {m}: {v}")

        # Derived: coalescing ratio (sectors per request)
        sectors = safe(action, "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum")
        requests = safe(action, "l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum")
        if sectors is not None and requests and requests > 0:
            ratio = sectors / requests
            flag = "  ← UNCOALESCED" if ratio > 4.5 else ""
            lines.append(f"  [derived] sectors_per_request_ld: {ratio:.2f}  (ideal ≤4){flag}")

        # Derived: register spill present?
        local_ld = safe(action, "smsp__sass_inst_executed_op_local_ld.sum", 0)
        local_st = safe(action, "smsp__sass_inst_executed_op_local_st.sum", 0)
        if local_ld or local_st:
            lines.append(
                f"  [derived] REGISTER SPILL detected  "
                f"(local_ld={local_ld}, local_st={local_st})"
            )

        # Derived: shared memory staging present?
        shared = safe(action, "smsp__sass_inst_executed_op_shared.sum", 0)
        if shared == 0:
            lines.append("  [derived] no shared-memory staging (num_stages=1 likely)")

        # NCU rule engine — top 5 by estimated speedup
        rules = rule_speedups(action)
        if rules:
            lines.append("  NCU rule engine (top by Est. Speedup):")
            for speedup, rule, msg in rules[:5]:
                truncated = msg[:120] + "..." if len(msg) > 120 else msg
                lines.append(f"    [{speedup:.0f}%] {rule}: {truncated}")

        sections.append("\n".join(lines))

    return "\n\n".join(sections)
