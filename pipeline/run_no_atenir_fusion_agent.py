"""Top-level entry point for the no-AtenIR fusion ablation."""

from pipeline.no_atenir_fusion_agent.cli import *  # noqa: F401,F403
from pipeline.no_atenir_fusion_agent.cli import main


if __name__ == "__main__":
    raise SystemExit(main())
