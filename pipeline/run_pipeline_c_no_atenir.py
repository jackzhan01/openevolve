"""Pipeline C: no-AtenIR forward-source-only autograd-pair baseline."""

from pipeline.no_atenir_fusion_agent.cli import *  # noqa: F401,F403
from pipeline.no_atenir_fusion_agent.cli import main


if __name__ == "__main__":
    raise SystemExit(main())
