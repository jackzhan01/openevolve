"""Weighted-geomean min-speedup wrapper for the LayerNorm autograd-pair evaluator."""

from __future__ import annotations

import os
import sys

os.environ.setdefault("AUTOGRAD_PAIR_SCORE_MODE", "speed_memory_min_weighted_geomean")
os.environ.setdefault("AUTOGRAD_PAIR_FULL_STEP_WEIGHT", "0.5")
os.environ.setdefault("AUTOGRAD_PAIR_MEMORY_PENALTY_WEIGHT", "0.05")
os.environ.setdefault("AUTOGRAD_PAIR_WORST_CASE_GUARD", "1")

from evaluator_autograd_pair import evaluate, main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
