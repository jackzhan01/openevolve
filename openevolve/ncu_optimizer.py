"""
NCU-guided silent optimizer.

After a child program is normally evaluated by the evolver, this module can run a second
"NCU optimization pass" on the same program:
  1. The evaluator is expected to have already produced ncu_* metrics for the child
     (caller's responsibility — typically by running the benchmark with NCU_MODE=always).
  2. This module takes those pre-extracted metrics, optionally parses the raw .ncu-rep
     file with openevolve's own helpers (triton_ncu_analyze / triton_ncu_stalls), and
     calls an LLM with the bundled ncu_docs/ Triton reference docs to identify and
     apply the single most impactful fix.
  3. The optimized code is returned to the caller, which evaluates it and keeps it only
     if it scores higher than the original.

The evolver (database, prompt sampler, island logic) is completely unaware of NCU:
  - ncu_* metrics are stripped before any program is stored in the database.
  - The optimizer's LLM call is invisible to the evolution loop.
  - The stored program is just a normal program with a (potentially improved) score.
"""

import logging
import time
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger(__name__)


_SYSTEM_PROMPT = """\
You are a GPU performance engineer specializing in Triton kernel optimization.

You will receive:
1. A Triton kernel's source code (forward + backward autograd pair).
2. Hardware profiling data from Nsight Compute (NCU) covering both passes.
3. Two reference documents:
   - 05-triton-analysis-dimensions.md  — how to read NCU metrics across 7 dimensions
   - 06-triton-diagnosis-playbook.md   — patterns that map NCU signals to Triton-level fixes

Your task is narrow and precise:
- Identify the SINGLE most significant performance bottleneck indicated by the NCU data.
- Apply exactly one fix to the code that addresses that bottleneck.
- Return the complete, corrected source code.

Hard constraints:
- Do NOT alter the forward or backward function signatures.
- Do NOT break the saved-tensor contract between forward and backward.
- Do NOT apply more than one fix.
- If the data does not clearly identify a bottleneck, return the original code unchanged.
- Optimisation target is backward-pass latency; forward is relevant only if it affects saved tensors.
"""

_USER_TEMPLATE = """\
## Source code (forward + backward)

```python
{code}
```

---

## NCU profiling results

{ncu_section}

---

## Reference: 05 — Triton analysis dimensions

{dim_doc}

---

## Reference: 06 — Triton diagnosis playbook

{playbook_doc}

---

## Your task

Step 1 — Walk through the relevant dimensions in doc 05 and state which metric is worst.
Step 2 — Match that metric to a pattern in doc 06.
Step 3 — State the single fix you will apply (one sentence).
Step 4 — Return the complete optimized code in a Python code block:

```python
# complete optimized code here
```

No other text outside the final code block.
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


_NCU_DOCS_DIR = Path(__file__).parent / "ncu_docs"


class NCUOptimizer:
    """
    Wraps the LLM call that turns NCU profiling data into an optimized kernel.

    Usage (inside a worker process):
        optimizer = NCUOptimizer(llm_ensemble=...)
        prompt = optimizer.build_prompt(code, ncu_metrics, report_path)
        optimized_code = await optimizer.optimize(code, ncu_metrics, report_path)
    """

    def __init__(self, llm_ensemble):
        self.llm_ensemble = llm_ensemble
        self._dim_doc: Optional[str] = None
        self._playbook_doc: Optional[str] = None

    def _load_docs(self) -> None:
        if self._dim_doc is None:
            dim_path = _NCU_DOCS_DIR / "05-triton-analysis-dimensions.md"
            self._dim_doc = dim_path.read_text(encoding="utf-8") if dim_path.exists() else "(doc not found)"
        if self._playbook_doc is None:
            pb_path = _NCU_DOCS_DIR / "06-triton-diagnosis-playbook.md"
            self._playbook_doc = pb_path.read_text(encoding="utf-8") if pb_path.exists() else "(doc not found)"

    def build_prompt(
        self,
        code: str,
        ncu_metrics: Dict[str, float],
        report_path: Optional[str] = None,
    ) -> Dict[str, str]:
        """
        Return {"system": ..., "user": ...} — the full LLM prompt for this optimization pass.
        This is intentionally a public method so callers can log or inspect the prompt.
        """
        self._load_docs()
        ncu_section = _build_ncu_section(ncu_metrics, report_path)
        user = _USER_TEMPLATE.format(
            code=code,
            ncu_section=ncu_section,
            dim_doc=self._dim_doc,
            playbook_doc=self._playbook_doc,
        )
        return {"system": _SYSTEM_PROMPT, "user": user}

    async def optimize(
        self,
        code: str,
        ncu_metrics: Dict[str, float],
        report_path: Optional[str] = None,
    ) -> Optional[str]:
        """
        Call the LLM and return optimized code, or None if the call fails or produces
        nothing different from the input.
        """
        t_start = time.time()
        logger.info("NCU optimizer: building prompt (report_path=%s)", report_path or "none")
        prompt = self.build_prompt(code, ncu_metrics, report_path)
        logger.info("NCU optimizer: prompt built in %.1fs (system=%d chars, user=%d chars)",
                    time.time() - t_start, len(prompt["system"]), len(prompt["user"]))

        logger.debug(
            "NCU optimizer prompt\n"
            "=== SYSTEM ===\n%s\n"
            "=== USER ===\n%s",
            prompt["system"],
            prompt["user"],
        )

        t_llm = time.time()
        logger.info("NCU optimizer: calling LLM...")
        try:
            response = await self.llm_ensemble.generate_with_context(
                system_message=prompt["system"],
                messages=[{"role": "user", "content": prompt["user"]}],
            )
            logger.info("NCU optimizer: LLM returned in %.1fs (%d chars)",
                        time.time() - t_llm, len(response) if response else 0)
        except Exception as exc:
            logger.warning("NCU optimizer: LLM call failed after %.1fs: %s", time.time() - t_llm, exc)
            return None

        if not response:
            logger.warning("NCU optimizer: LLM returned empty response")
            return None

        logger.debug("NCU optimizer LLM response\n=== RESPONSE ===\n%s", response)

        t_parse = time.time()
        logger.info("NCU optimizer: parsing code from LLM response...")
        from openevolve.utils.code_utils import parse_full_rewrite
        optimized = parse_full_rewrite(response, language="python")
        logger.info("NCU optimizer: parse done in %.1fs — result: %s",
                    time.time() - t_parse,
                    "no change" if (not optimized or optimized.strip() == code.strip()) else f"{len(optimized)} chars")

        if not optimized or optimized.strip() == code.strip():
            return None
        return optimized
