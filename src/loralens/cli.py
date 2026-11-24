"""Command line: ``loralens info`` and ``loralens resize``."""

from __future__ import annotations

import argparse
import json
import os
import sys

from . import __version__
from .adapter import AdapterError, load_adapter
from .info import human_bytes, render, summarize
from .resize import report, resize, write


def _positive_int(text: str) -> int:
    try:
        value = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"not a whole number: {text!r}") from None
    if value < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return value


def _energy(text: str) -> float:
    try:
        value = float(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"not a number: {text!r}") from None
    if value > 1 and value <= 100:
        value /= 100
    if not 0 < value <= 1:
        raise argparse.ArgumentTypeError("use a fraction like 0.95 or a percent like 95")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="loralens",
        description="Inspect LoRA adapters (kohya, PEFT, diffusers) and lower their rank with SVD.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    commands = parser.add_subparsers(dest="command", required=True)

    info = commands.add_parser("info", help="show format, rank, alpha, targets and metadata")
    info.add_argument("path", help=".safetensors file, or a PEFT adapter folder")
    info.add_argument("--config", help="adapter_config.json (default: next to the file)")
    info.add_argument("--modules", action="store_true", help="also list every module")
    info.add_argument("--json", action="store_true", help="print JSON instead of text")

    rz = commands.add_parser("resize", help="lower the rank of every LoRA module")
    rz.add_argument("path", help=".safetensors file, or a PEFT adapter folder")
    rz.add_argument("-o", "--output", help="output file (a folder for PEFT adapters)")
    target = rz.add_mutually_exclusive_group(required=True)
    target.add_argument("--rank", type=_positive_int, help="new rank for every module")
    target.add_argument("--energy", type=_energy,
                        help="per module, the smallest rank that keeps this share of the change (0.95 or 95)")
    rz.add_argument("--max-rank", type=_positive_int, help="cap for --energy")
    rz.add_argument("--min-rank", type=_positive_int, default=1, help="floor for --energy (default 1)")
    rz.add_argument("--config", help="adapter_config.json (default: next to the file)")
    rz.add_argument("--dry-run", action="store_true", help="report what would change, write nothing")
    rz.add_argument("--force", action="store_true", help="overwrite the output if it exists")
    rz.add_argument("--json", action="store_true", help="print JSON instead of text")
    return parser


def _ranks(counts: dict) -> str:
    keys = sorted(counts)
    return str(keys[0]) if len(keys) == 1 else f"{keys[0]} to {keys[-1]}"


def render_report(rep: dict, written: list[str]) -> str:
    lines = [f"resized  {rep['resized']} of {rep['modules']} LoRA modules"]
    for status, n in rep["kept"].items():
        lines.append(f"{'':9}{n} {status}")
    lines.append(f"rank     {_ranks(rep['old_ranks'])} -> {_ranks(rep['new_ranks'])}")
    lines.append(f"size     {human_bytes(rep['bytes_before'])} -> {human_bytes(rep['bytes_after'])}")
    lines.append(f"energy   {rep['energy_mean']:.1%} kept on average, lowest {rep['energy_min']:.1%}")
    for w in rep["worst"]:
        lines.append(f"{'':9}{w['energy']:.1%} at rank {w['rank']}: {w['name']}")
    if written:
        for path in written:
            lines.append(f"wrote    {path}")
    else:
        lines.append("dry run  nothing written")
    return "\n".join(lines)


def _check_output(args, adapter) -> str:
    peft = adapter.config is not None
    output = args.output
    if output is None:
        raise AdapterError("give an output with -o (or use --dry-run)")
    source = os.path.abspath(adapter.path)
    if peft:
        target = os.path.abspath(output)
        if target == os.path.dirname(source) or target == source:
            raise AdapterError("for a PEFT adapter, -o must be a new folder, not the input's folder")
        existing = [os.path.join(output, n) for n in ("adapter_model.safetensors", "adapter_config.json")]
        if os.path.isfile(output):
            raise AdapterError(f"{output} is a file; a PEFT adapter is written to a folder")
    else:
        if os.path.isdir(output):
            output = os.path.join(output, os.path.basename(adapter.path))
        if os.path.abspath(output) == source:
            raise AdapterError("the output is the input file; choose another path")
        existing = [output]
    clash = [p for p in existing if os.path.exists(p)]
    if clash and not args.force:
        raise AdapterError(f"{clash[0]} already exists; use --force to overwrite it")
    return output


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        adapter = load_adapter(args.path, args.config)
        if args.command == "info":
            summary = summarize(adapter)
            if args.json:
                if not args.modules:
                    summary.pop("module_rows")
                print(json.dumps(summary, indent=2))
            else:
                print(render(summary, show_modules=args.modules))
            return 0

        output = None if args.dry_run else _check_output(args, adapter)
        results, tensors, metadata, config = resize(
            adapter, rank=args.rank, energy=args.energy, max_rank=args.max_rank, min_rank=args.min_rank
        )
        before = sum(t.nbytes for t in adapter.tensors.values())
        after = sum(t.nbytes for t in tensors.values())
        written = [] if output is None else write(output, tensors, metadata, config)
        rep = report(results, before, after)
        if args.json:
            print(json.dumps({**rep, "written": written}, indent=2))
        else:
            print(render_report(rep, written))
        return 0
    except AdapterError as e:
        print(f"loralens: error: {e}", file=sys.stderr)
        return 2
    except OSError as e:
        print(f"loralens: error: {e.filename or ''}: {e.strerror}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
