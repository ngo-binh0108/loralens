"""Summarise what is inside a LoRA file."""

from __future__ import annotations

import os
from collections import Counter

from .adapter import Adapter, target_groups
from .st import FLOAT_DTYPES


def module_rows(adapter: Adapter) -> list[dict]:
    rows = []
    for module in adapter.modules:
        row: dict = {"name": module.name, "kind": module.kind}
        if module.kind == "lora":
            down = adapter.tensors[module.key("down")]
            up = adapter.tensors[module.key("up")]
            rank = down.shape[0]
            alpha, source = adapter.alpha(module, rank)
            row.update(
                rank=rank,
                alpha=alpha,
                alpha_source=source,
                scale=adapter.scale(alpha, rank),
                shape=[up.shape[0], *down.shape[1:]],
                params=down.numel + up.numel,
                dtype=down.dtype,
                dora="dora" in module.keys,
            )
        rows.append(row)
    return rows


def summarize(adapter: Adapter) -> dict:
    rows = module_rows(adapter)
    lora = [r for r in rows if r["kind"] == "lora"]
    kinds = Counter(r["kind"] for r in rows if r["kind"] != "other")
    base, base_source = adapter.guess_base()

    notes = []
    defaults = sum(1 for r in lora if r["alpha_source"] == "default")
    if defaults:
        notes.append(f"{defaults} module(s) have no alpha; assuming alpha = rank (scale 1)")
    if any(r["dora"] for r in lora):
        notes.append("DoRA magnitudes found; resize keeps those modules unchanged")
    for kind in ("loha", "lokr", "oft", "full", "lora-tucker"):
        if kinds.get(kind):
            notes.append(f"{kinds[kind]} {kind} module(s) can't be resized and are copied as they are")
    if kinds.get("incomplete"):
        notes.append(f"{kinds['incomplete']} module(s) have only one of their down/up pair")
    if adapter.loose:
        notes.append(f"{len(adapter.loose)} tensor(s) are not part of a LoRA module")

    return {
        "file": adapter.path,
        "bytes": os.path.getsize(adapter.path),
        "format": adapter.format,
        "config": adapter.config_path,
        "base": base,
        "base_source": base_source,
        "modules": dict(kinds),
        "ranks": dict(Counter(r["rank"] for r in lora)),
        "alphas": dict(Counter(r["alpha"] for r in lora)),
        "scales": dict(Counter(round(r["scale"], 6) for r in lora)),
        "rslora": adapter.rslora,
        "dtypes": dict(Counter(t.dtype for t in adapter.tensors.values())),
        "params": sum(t.numel for t in adapter.tensors.values() if t.dtype in FLOAT_DTYPES),
        "targets": dict(target_groups(adapter).most_common()),
        "metadata": {k: v for k, v in adapter.metadata.items() if k.startswith(_SHOWN_METADATA)},
        "notes": notes,
        "module_rows": rows,
    }


_SHOWN_METADATA = (
    "ss_network_module", "ss_network_dim", "ss_network_alpha", "ss_network_args",
    "ss_base_model_version", "ss_sd_model_name", "ss_output_name", "ss_resolution",
    "ss_num_train_images", "ss_epoch", "ss_steps", "ss_learning_rate",
    "modelspec.title", "modelspec.architecture", "loralens_resize",
)


def human_bytes(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} GB"


def human_count(n: int) -> str:
    if n >= 1_000_000_000:
        return f"{n / 1e9:.2f}B"
    if n >= 1_000_000:
        return f"{n / 1e6:.1f}M"
    if n >= 1_000:
        return f"{n / 1e3:.1f}K"
    return str(n)


def _number(x: float) -> str:
    return f"{x:g}"


def _spread(counts: dict, fmt=str) -> str:
    if not counts:
        return "-"
    if len(counts) == 1:
        return fmt(next(iter(counts)))
    common = ", ".join(f"{fmt(k)} x{v}" for k, v in Counter(counts).most_common(4))
    more = "" if len(counts) <= 4 else f", {len(counts) - 4} more"
    return f"{fmt(min(counts))} to {fmt(max(counts))} ({common}{more})"


def render(summary: dict, show_modules: bool = False) -> str:
    lines = []

    def row(label: str, value: str) -> None:
        lines.append(f"{label:<9}{value}")

    row("file", f"{summary['file']} ({human_bytes(summary['bytes'])})")
    fmt = summary["format"]
    if summary["config"]:
        fmt += f" ({os.path.basename(summary['config'])})"
    row("format", fmt)
    if summary["base"]:
        row("base", f"{summary['base']} (from {summary['base_source']})")
    row("modules", ", ".join(f"{n} {kind}" for kind, n in summary["modules"].items()) or "-")
    if summary["ranks"]:
        row("rank", _spread(summary["ranks"]))
        alpha = _spread(summary["alphas"], _number)
        scale = _spread(summary["scales"], _number)
        row("alpha", f"{alpha} (scale {scale}{', rsLoRA' if summary['rslora'] else ''})")
    row("dtype", ", ".join(summary["dtypes"]))
    row("params", human_count(summary["params"]))
    if summary["targets"]:
        targets = list(summary["targets"].items())
        text = ", ".join(f"{name} {n}" for name, n in targets[:8])
        if len(targets) > 8:
            text += f", {len(targets) - 8} more"
        row("targets", text)
    for key, value in summary["metadata"].items():
        row("meta", f"{key} = {value if len(value) <= 80 else value[:77] + '...'}")
    for note in summary["notes"]:
        row("note", note)

    if show_modules:
        lines.append("")
        lines.append(f"{'rank':>5} {'alpha':>7} {'scale':>7}  {'shape':<16} name")
        for r in summary["module_rows"]:
            if r["kind"] != "lora":
                lines.append(f"{'':>5} {'':>7} {'':>7}  {r['kind']:<16} {r['name']}")
                continue
            shape = "x".join(str(n) for n in r["shape"])
            lines.append(f"{r['rank']:>5} {_number(r['alpha']):>7} {_number(round(r['scale'], 4)):>7}  {shape:<16} {r['name']}")
    return "\n".join(lines)
