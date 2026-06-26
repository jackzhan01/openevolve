"""Pipeline B: LLM-free hand-written dispatch autograd-pair seed."""

from pipeline.handwritten_dispatch.cli import *  # noqa: F401,F403
from pipeline.handwritten_dispatch.cli import main


if __name__ == "__main__":
    raise SystemExit(main())
