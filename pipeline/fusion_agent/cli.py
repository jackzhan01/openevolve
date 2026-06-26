"""CLI for AtenIR backward fusion synthesis."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from pipeline.fusion_agent.synthesize import FusionConfig, synthesize_fusion


def _default_api_signature(public_api: str) -> str:
    if public_api == "layernorm_backward_triton":
        return "def layernorm_backward_triton(dy, x, weight, bias, eps=1e-5):"
    return f"def {public_api}(*args):"


def _default_return_contract(public_api: str) -> str:
    if public_api == "layernorm_backward_triton":
        return "return dx, dweight, dbias"
    return "return the backward gradients in the public API's expected order"


def _default_scalar_args(api_signature: str, public_api: str) -> tuple[str, ...]:
    if "eps" in api_signature or public_api in {
        "layernorm_backward_triton",
        "rmsnorm_backward_triton",
    }:
        return ("1e-5",)
    return ()


def _parse_indices(value: str | None) -> tuple[int, ...] | None:
    if value is None:
        return None
    if not value.strip():
        return ()
    return tuple(int(part) for part in value.split(",") if part.strip())


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AtenIR Backward Fusion Agent")
    parser.add_argument("--forward", required=True)
    parser.add_argument(
        "--example-input",
        default="[(8,64) f32, (64) f32, (64) f32]",
        help="AtenIR extraction example input, e.g. '[(8,64) f32, (64) f32]'",
    )
    parser.add_argument("--public-api", default="layernorm_backward_triton")
    parser.add_argument(
        "--api-signature",
        default=None,
        help="Python signature shown to the LLM, e.g. 'def rmsnorm_backward_triton(dy, x, weight, eps=1e-5):'",
    )
    parser.add_argument(
        "--return-contract",
        default=None,
        help="Return statement/contract shown to the LLM, e.g. 'return dx, dweight'",
    )
    parser.add_argument("--op", default="layernorm")
    parser.add_argument("--mode", default="dynamic", choices=["static", "dynamic", "nontile"])
    parser.add_argument("--output-dir", default="atenir_fusion_layernorm")
    parser.add_argument("--api-base", default="https://api.openai.com/v1")
    parser.add_argument("--model", default="gpt-5.5")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=16000)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--dtype", action="append", default=None)
    parser.add_argument("--atol", type=float, default=2e-5)
    parser.add_argument("--rtol", type=float, default=2e-5)
    parser.add_argument("--fp16-atol", type=float, default=5e-2)
    parser.add_argument("--fp16-rtol", type=float, default=5e-2)
    parser.add_argument(
        "--scalar",
        action="append",
        default=None,
        help="Scalar argument passed to verifier; omit for scalar-free APIs unless the signature contains eps",
    )
    parser.add_argument(
        "--backward-input-indices",
        default=None,
        help="Comma-separated forward input indices to pass to backward after dy; default passes all inputs",
    )
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument(
        "--lowering-context-file",
        default=None,
        help="Optional Markdown/text file with verified per-op lowering context for the fusion prompt",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    dtypes = tuple(args.dtype or ["float32", "float16"])
    api_signature = args.api_signature or _default_api_signature(args.public_api)
    return_contract = args.return_contract or _default_return_contract(args.public_api)
    scalar_args = tuple(args.scalar) if args.scalar is not None else _default_scalar_args(
        api_signature,
        args.public_api,
    )
    lowering_context = None
    if args.lowering_context_file:
        lowering_context = Path(args.lowering_context_file).read_text(encoding="utf-8")
    return synthesize_fusion(
        FusionConfig(
            forward=args.forward,
            example_input=args.example_input,
            public_api=args.public_api,
            api_signature=api_signature,
            return_contract=return_contract,
            op=args.op,
            mode=args.mode,
            output_dir=Path(args.output_dir).resolve(),
            api_base=args.api_base,
            model=args.model,
            api_key=args.api_key,
            max_attempts=args.max_attempts,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            timeout=args.timeout,
            dtypes=dtypes,
            atol=args.atol,
            rtol=args.rtol,
            fp16_atol=args.fp16_atol,
            fp16_rtol=args.fp16_rtol,
            scalar_args=scalar_args,
            backward_input_indices=_parse_indices(args.backward_input_indices),
            python=args.python,
            lowering_context=lowering_context,
            dry_run=args.dry_run,
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())
