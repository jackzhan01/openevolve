"""CLI for the autograd-pair saved-tensor fusion agent."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from pipeline.autograd_pair_fusion_agent.synthesize import (
    AutogradPairConfig,
    synthesize_autograd_pair,
)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Autograd-pair saved-tensor fusion agent")
    parser.add_argument("--forward", required=True)
    parser.add_argument(
        "--example-input",
        default="[(8,64) f32, (64) f32, (64) f32]",
        help="AtenIR extraction input spec for the forward reference",
    )
    parser.add_argument("--output-dir", default="autograd_pair_layernorm")
    parser.add_argument(
        "--evaluator",
        default="benchmark/triton_layernorm_backward_bench/evaluator_autograd_pair.py",
        help="Evaluator used to verify generated autograd-pair programs",
    )
    parser.add_argument("--api-base", default="https://api.openai.com/v1")
    parser.add_argument("--model", default="gpt-5.5")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=16000)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--eval-timeout", type=int, default=120,
                        help="kill a candidate kernel eval after this many seconds (guards against "
                             "a hanging/deadlocked generated kernel)")
    parser.add_argument("--dtype", action="append", default=None)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--lowering-context-file", default=None)
    parser.add_argument("--pair-api", default=None)
    parser.add_argument("--pair-api-file", default=None)
    parser.add_argument("--task-context", default=None)
    parser.add_argument("--task-context-file", default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    lowering_context = None
    if args.lowering_context_file:
        lowering_context = Path(args.lowering_context_file).read_text(encoding="utf-8")
    pair_api = args.pair_api
    if args.pair_api_file:
        pair_api = Path(args.pair_api_file).read_text(encoding="utf-8")
    task_context = args.task_context
    if args.task_context_file:
        task_context = Path(args.task_context_file).read_text(encoding="utf-8")
    return synthesize_autograd_pair(
        AutogradPairConfig(
            forward=args.forward,
            example_input=args.example_input,
            output_dir=Path(args.output_dir).resolve(),
            evaluator=args.evaluator,
            api_base=args.api_base,
            model=args.model,
            api_key=args.api_key,
            max_attempts=args.max_attempts,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            timeout=args.timeout,
            eval_timeout=args.eval_timeout,
            dtypes=tuple(args.dtype or ["float32", "float16", "bfloat16"]),
            python=args.python,
            lowering_context=lowering_context,
            pair_api=pair_api,
            task_context=task_context,
            dry_run=args.dry_run,
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())
