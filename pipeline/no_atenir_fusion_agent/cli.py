"""CLI for no-AtenIR forward-reference-only autograd-pair synthesis."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from pipeline.no_atenir_fusion_agent.synthesize import (
    NoAtenIRFusionConfig,
    synthesize_no_atenir_fusion,
)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="No-AtenIR forward-reference-only autograd-pair fusion agent"
    )
    parser.add_argument(
        "--forward",
        default="benchmark.triton_layernorm_backward_bench.forward_ref:layernorm_forward_ref",
    )
    parser.add_argument("--output-dir", default="no_atenir_fusion_layernorm")
    parser.add_argument("--api-base", default="https://api.openai.com/v1")
    parser.add_argument("--model", default="gpt-5.5")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=16000)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    return synthesize_no_atenir_fusion(
        NoAtenIRFusionConfig(
            forward=args.forward,
            output_dir=Path(args.output_dir).resolve(),
            api_base=args.api_base,
            model=args.model,
            api_key=args.api_key,
            max_attempts=args.max_attempts,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            timeout=args.timeout,
            python=args.python,
            dry_run=args.dry_run,
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())
