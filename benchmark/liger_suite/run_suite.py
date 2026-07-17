"""Build the Liger benchmark suite: taskspec + prep for every op in the manifest.

    python -m benchmark.liger_suite.run_suite [--only op1,op2] [--force]

Needs a GPU and OPENAI_API_KEY (taskspec's self-check and the wrapper gate run real kernels).
One op failing (e.g. its wrapper gate exhausts attempts) does NOT block the rest — the run
ends with a status table plus suite_status.json, and failed ops are listed for hand fallback.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from benchmark.liger_suite.manifest import SUITE_OPS, FORWARDS, liger_source_paths  # noqa: E402
from pipeline.case_harness_agent.orchestrate import OrchestratorConfig, orchestrate  # noqa: E402

BUILD_STAGES = ["taskspec", "prep"]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="build the Liger benchmark suite (taskspec+prep)")
    ap.add_argument("--only", default="", help="comma-separated subset of ops to build")
    ap.add_argument("--force", action="store_true", help="rebuild even if artifacts exist")
    ap.add_argument("--model", default="gpt-5.5")
    ap.add_argument("--api-base", default="https://api.openai.com/v1")
    args = ap.parse_args(argv)

    picked = [e for e in SUITE_OPS if not args.only or e.op in args.only.split(",")]
    status: dict[str, dict] = {}
    for i, entry in enumerate(picked, 1):
        bench = REPO_ROOT / "benchmark" / f"triton_{entry.op}_backward_bench"
        print(f"\n{'#' * 72}\n# [{i}/{len(picked)}] {entry.op}  ({entry.notes})\n{'#' * 72}", flush=True)
        t0 = time.time()
        cfg = OrchestratorConfig(
            op=entry.op,
            bench_dir=bench,
            forward=f"{FORWARDS / entry.forward_file}",
            model=args.model, api_base=args.api_base,
            api_key=os.environ.get("OPENAI_API_KEY"),
            perf_baseline="liger",
            liger_sources=liger_source_paths(entry),
            force=args.force,
        )
        try:
            rc = orchestrate(cfg, BUILD_STAGES, [])
        except SystemExit as e:  # a stage hard-fail must not kill the batch
            rc = int(e.code or 1)
        except Exception as e:
            print(f"[suite] {entry.op} crashed: {e}", flush=True)
            rc = 1
        wrapper = bench / "strong_baselines" / f"liger_{entry.op}.py"
        status[entry.op] = {
            "ok": rc == 0,
            "seconds": round(time.time() - t0, 1),
            "task_spec": (bench / "task_spec.py").exists(),
            "wrapper": wrapper.exists(),
            "notes": entry.notes,
        }
        print(f"[suite] {entry.op}: {'OK' if rc == 0 else 'FAILED'} in {status[entry.op]['seconds']}s", flush=True)

    out = Path(__file__).parent / "suite_status.json"
    out.write_text(json.dumps(status, indent=2), encoding="utf-8")
    print(f"\n{'=' * 72}\nLiger suite build summary  ({out})\n{'=' * 72}")
    print(f"{'op':<22}{'build':<8}{'task_spec':<11}{'wrapper':<9}{'seconds':<8}")
    for op, s in status.items():
        print(f"{op:<22}{'OK' if s['ok'] else 'FAIL':<8}{str(s['task_spec']):<11}{str(s['wrapper']):<9}{s['seconds']:<8}")
    failed = [op for op, s in status.items() if not s["ok"]]
    if failed:
        print(f"\nneeds hand fallback: {', '.join(failed)}")
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
