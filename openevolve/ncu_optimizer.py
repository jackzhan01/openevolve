"""
NCU-guided silent optimizer.

After a child program is normally evaluated by the evolver, this module can run
a second "NCU optimization pass" on the same program as a staged pipeline:

  Stage 0 — deterministic roofline triage (no LLM): classify the bottleneck
            from SOL metrics; skip the pass entirely when the kernel is
            already at the hardware roofline.
  Stage 1 — diagnosis LLM call: GPU specs + roofline verdict + parsed NCU
            report + the bundled 05/06 Triton reference docs + source code
            -> structured JSON with root causes and recommended fixes (as
            many as the data supports — no artificial one-fix limit).
  Stage 2 — generation LLM call: the diagnosis + benchmark context + hard
            contract constraints -> complete rewritten kernel.

The caller re-evaluates the generated code and keeps it only if it scores
strictly higher than the original.

The evolver (database, prompt sampler, island logic) is completely unaware of
NCU: ncu_* metrics are stripped before any program is stored, both LLM calls
here are invisible to the evolution loop, and the stored program is just a
normal program with a (potentially improved) score. Nothing in this module
touches OpenEvolve's core architecture.
"""

import json
import logging
import re
import time
from pathlib import Path
from typing import Any, Dict, Optional

from openevolve.ncu_gpu_specs import format_gpu_specs, get_gpu_specs
from openevolve.ncu_roofline import RooflineResult, analyze as roofline_analyze, format_roofline

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Stage 1 — diagnosis prompt                                                   #
# --------------------------------------------------------------------------- #

_DIAGNOSIS_SYSTEM = """\
You are a GPU performance expert analyzing Nsight Compute (NCU) profiling data
for a Triton kernel program (a forward + backward autograd pair).

You will receive GPU hardware specifications, a deterministic roofline triage,
the parsed NCU profile (per-kernel metrics and stall breakdowns), the program
source, and two reference documents:
  - 05-triton-analysis-dimensions.md — how to read NCU metrics across 7 dimensions
  - 06-triton-diagnosis-playbook.md  — patterns that map NCU signals to Triton-level fixes

Your job is diagnosis only (no code): identify the performance bottlenecks and
recommend concrete fixes, grounded in the metric values. Cite actual numbers as
evidence for every claim.
"""

_DIAGNOSIS_USER_TEMPLATE = """\
## GPU specifications

{gpu_specs}

## Roofline triage (deterministic, from SOL metrics)

{roofline}

## NCU profiling results

{ncu_section}

## Program source (forward + backward)

```python
{code}
```

## Reference: 05 — Triton analysis dimensions

{dim_doc}

## Reference: 06 — Triton diagnosis playbook

{playbook_doc}

## Output

Respond with a JSON array (no markdown fence). Identify the bottleneck(s) you
find — order by importance, most critical first. For each bottleneck list every
root cause the data supports, and for each root cause the fix(es) you recommend.
There is no fixed number: report what the evidence justifies.

[
    {{
        "category": "memory" | "compute" | "underutilized",
        "summary": "One-line summary",
        "reasoning": "Explanation citing metric values",
        "root_causes": [
            {{
                "cause": "Description",
                "evidence": [{{"metric": "name", "value": 0.0, "interpretation": "meaning"}}],
                "fixes": [
                    {{"fix": "Actionable Triton-level instruction", "rationale": "Why this helps"}}
                ]
            }}
        ]
    }}
]
"""


# --------------------------------------------------------------------------- #
# Stage 2 — generation prompt                                                  #
# --------------------------------------------------------------------------- #

_GENERATION_SYSTEM = """\
You are a GPU performance engineer specializing in Triton kernel optimization.
You rewrite a forward + backward autograd pair to apply the fixes recommended
by a profiling-based diagnosis.

Hard constraints:
- Do NOT alter the forward or backward function signatures.
- Do NOT break the saved-tensor contract between forward and backward
  (backward must consume exactly what forward saves).
- Preserve numerical correctness for all supported dtypes and shapes.
- The optimization target is backward-pass latency; forward matters only where
  it affects saved tensors or the full-step time.
- You may apply several coordinated fixes from the diagnosis; prioritize the
  most impactful ones. Do not attempt unrelated rewrites the diagnosis does
  not support.
"""

_GENERATION_USER_TEMPLATE = """\
## GPU specifications

{gpu_specs}

## Current measured performance

{benchmark_context}

## Roofline triage

{roofline}

## Diagnosis (from NCU profiling analysis)

{diagnosis}

## Current source code (forward + backward)

```python
{code}
```

## Your task

Apply the recommended fixes to the code. Briefly state which fixes you are
applying and why (a few lines), then return the complete optimized source in a
single Python code block:

```python
# complete optimized code here
```

Nothing after the code block.
"""


def _build_ncu_section(
    metrics: Dict[str, float],
    report_path: Optional[str],
) -> str:
    """Assemble the NCU data block shown to the LLM."""
    parts: list[str] = []

    if report_path and Path(report_path).exists():
        t0 = time.time()
        logger.info(f"NCU prompt: running triton_ncu_analyze on {report_path}")
        try:
            from openevolve.triton_ncu_analyze import format_metrics_report
            analysis = format_metrics_report(report_path)
            logger.info(f"NCU prompt: triton_ncu_analyze done in {time.time()-t0:.1f}s ({len(analysis) if analysis else 0} chars)")
            if analysis:
                parts.append("### Key metrics — all kernels\n\n" + analysis)
        except Exception as exc:
            logger.warning(f"NCU prompt: triton_ncu_analyze failed after {time.time()-t0:.1f}s: {exc}")

        t1 = time.time()
        logger.info(f"NCU prompt: running triton_ncu_stalls on {report_path}")
        try:
            from openevolve.triton_ncu_stalls import extract_stall_summary
            stalls = extract_stall_summary(report_path)
            logger.info(f"NCU prompt: triton_ncu_stalls done in {time.time()-t1:.1f}s ({len(stalls) if stalls else 0} chars)")
            if stalls:
                parts.append("### Stall breakdown — all kernels\n\n" + stalls)
        except Exception as exc:
            logger.warning(f"NCU prompt: triton_ncu_stalls failed after {time.time()-t1:.1f}s: {exc}")

    # Always include the pre-extracted summary metrics as fallback / supplement
    if metrics:
        lines = ["### Pre-extracted summary (from evaluator)\n"]
        for k, v in sorted(metrics.items()):
            lines.append(f"  {k}: {v}")
        parts.append("\n".join(lines))

    return "\n\n".join(parts) if parts else "(no NCU data available)"


def _format_benchmark_context(benchmark_metrics: Optional[Dict[str, Any]]) -> str:
    """Render the evolver's clean metrics into performance context for Stage 2."""
    if not benchmark_metrics:
        return "(no benchmark data available)"
    interesting = [
        ("combined_score", "combined score (higher is better)"),
        ("speedup", "backward speedup vs baseline"),
        ("full_step_speedup", "forward+backward speedup vs baseline"),
        ("backward_from_saved_ms", "backward latency (ms)"),
        ("forward_ms", "forward latency (ms)"),
        ("forward_backward_full_step_ms", "full-step latency (ms)"),
        ("baseline_latency_ms", "baseline backward latency (ms)"),
        ("saved_bytes", "saved-tensor bytes"),
    ]
    lines = []
    for key, label in interesting:
        v = benchmark_metrics.get(key)
        if isinstance(v, (int, float)):
            lines.append(f"- {label}: {v:.4f}" if isinstance(v, float) else f"- {label}: {v}")
    score = benchmark_metrics.get("combined_score")
    if isinstance(score, (int, float)):
        lines.append(
            f"- target: the rewritten kernel must score HIGHER than {score:.4f} to be kept"
        )
    return "\n".join(lines) if lines else "(no benchmark data available)"


def _parse_diagnosis(response: str) -> tuple[Optional[list], str]:
    """Extract the JSON diagnosis from the LLM response.

    Returns (parsed_list_or_None, pretty_text). On parse failure the raw
    response text is used downstream — the generation prompt still works with
    free-form diagnosis prose.
    """
    for pattern in (r"\[[\s\S]*\]", r"\{[\s\S]*\}"):
        match = re.search(pattern, response)
        if not match:
            continue
        try:
            data = json.loads(match.group())
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict):
            data = [data]
        if isinstance(data, list):
            return data, json.dumps(data, indent=2)
    return None, response.strip()


_NCU_DOCS_DIR = Path(__file__).parent / "ncu_docs"


class NCUOptimizer:
    """
    Staged NCU optimization pass: roofline triage -> diagnosis LLM call ->
    generation LLM call. Returns optimized code for the caller to re-evaluate.

    Usage (inside a worker process):
        optimizer = NCUOptimizer(llm_ensemble=...)
        optimized_code = await optimizer.optimize(
            code, ncu_metrics, report_path, benchmark_metrics=child_metrics
        )
    """

    def __init__(self, llm_ensemble, skip_at_roofline_pct: float = 95.0):
        self.llm_ensemble = llm_ensemble
        self.skip_at_roofline_pct = float(skip_at_roofline_pct)
        self._dim_doc: Optional[str] = None
        self._playbook_doc: Optional[str] = None
        # Per-run artifacts of the most recent optimize() call, for callers
        # that persist the full pass record.
        self.last_roofline: Optional[str] = None
        self.last_ncu_section: Optional[str] = None
        self.last_diagnosis: Optional[str] = None
        self.last_response: Optional[str] = None
        self.last_skip_reason: Optional[str] = None

    def _load_docs(self) -> None:
        if self._dim_doc is None:
            dim_path = _NCU_DOCS_DIR / "05-triton-analysis-dimensions.md"
            self._dim_doc = dim_path.read_text(encoding="utf-8") if dim_path.exists() else "(doc not found)"
        if self._playbook_doc is None:
            pb_path = _NCU_DOCS_DIR / "06-triton-diagnosis-playbook.md"
            self._playbook_doc = pb_path.read_text(encoding="utf-8") if pb_path.exists() else "(doc not found)"

    async def _call_llm(self, system: str, user: str, phase: str) -> Optional[str]:
        t0 = time.time()
        logger.info("NCU optimizer: calling LLM (%s, user prompt %d chars)...", phase, len(user))
        logger.debug("NCU optimizer %s prompt\n=== SYSTEM ===\n%s\n=== USER ===\n%s", phase, system, user)
        try:
            response = await self.llm_ensemble.generate_with_context(
                system_message=system,
                messages=[{"role": "user", "content": user}],
            )
            logger.info("NCU optimizer: %s LLM returned in %.1fs (%d chars)",
                        phase, time.time() - t0, len(response) if response else 0)
        except Exception as exc:
            logger.warning("NCU optimizer: %s LLM call failed after %.1fs: %s", phase, time.time() - t0, exc)
            return None
        if not response:
            logger.warning("NCU optimizer: %s LLM returned empty response", phase)
            return None
        logger.debug("NCU optimizer %s response\n=== RESPONSE ===\n%s", phase, response)
        return response

    async def optimize(
        self,
        code: str,
        ncu_metrics: Dict[str, float],
        report_path: Optional[str] = None,
        benchmark_metrics: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        """
        Run the staged pass. Returns optimized code, or None if the pass was
        skipped (at roofline), a call failed, or the result is unchanged.
        """
        self.last_roofline = None
        self.last_ncu_section = None
        self.last_diagnosis = None
        self.last_response = None
        self.last_skip_reason = None

        # ---- Stage 0: deterministic roofline triage --------------------- #
        roofline = roofline_analyze(
            ncu_metrics, roofline_threshold_pct=self.skip_at_roofline_pct
        )
        roofline_text = format_roofline(roofline)
        self.last_roofline = roofline_text
        logger.info("NCU optimizer: roofline triage — bottleneck=%s, efficiency=%.1f%%",
                    roofline.bottleneck, roofline.efficiency_pct)

        if roofline.at_roofline:
            self.last_skip_reason = (
                f"at roofline ({roofline.efficiency_pct:.1f}% SOL >= "
                f"{self.skip_at_roofline_pct:.0f}%) — nothing left to win"
            )
            logger.info("NCU optimizer: skipping pass — %s", self.last_skip_reason)
            return None

        # ---- Stage 1: diagnosis LLM call --------------------------------- #
        self._load_docs()
        ncu_section = _build_ncu_section(ncu_metrics, report_path)
        self.last_ncu_section = ncu_section
        gpu_specs_text = format_gpu_specs(get_gpu_specs())

        diagnosis_user = _DIAGNOSIS_USER_TEMPLATE.format(
            gpu_specs=gpu_specs_text,
            roofline=roofline_text,
            ncu_section=ncu_section,
            code=code,
            dim_doc=self._dim_doc,
            playbook_doc=self._playbook_doc,
        )
        diagnosis_response = await self._call_llm(_DIAGNOSIS_SYSTEM, diagnosis_user, "diagnosis")
        if diagnosis_response is None:
            return None

        parsed, diagnosis_text = _parse_diagnosis(diagnosis_response)
        self.last_diagnosis = diagnosis_text
        if parsed is None:
            logger.warning("NCU optimizer: diagnosis JSON parse failed — using raw text")
        else:
            n_causes = sum(len(b.get("root_causes", [])) for b in parsed)
            n_fixes = sum(
                len(rc.get("fixes", []))
                for b in parsed
                for rc in b.get("root_causes", [])
            )
            logger.info(
                "NCU optimizer: diagnosis parsed — %d bottleneck(s), %d root cause(s), %d fix(es)",
                len(parsed), n_causes, n_fixes,
            )

        # ---- Stage 2: generation LLM call -------------------------------- #
        generation_user = _GENERATION_USER_TEMPLATE.format(
            gpu_specs=gpu_specs_text,
            benchmark_context=_format_benchmark_context(benchmark_metrics),
            roofline=roofline_text,
            diagnosis=diagnosis_text,
            code=code,
        )
        generation_response = await self._call_llm(_GENERATION_SYSTEM, generation_user, "generation")
        if generation_response is None:
            return None
        self.last_response = generation_response

        from openevolve.utils.code_utils import parse_full_rewrite

        optimized = parse_full_rewrite(generation_response, language="python")
        if not optimized or optimized.strip() == code.strip():
            logger.info("NCU optimizer: generation produced no change")
            return None
        return optimized
