"""Lower the rank of LoRA modules with a truncated SVD.

A LoRA module adds ``scale * up @ down`` to a base weight, where ``up`` is
(out, r) and ``down`` is (r, in). The best rank-k approximation of that product
(in the least-squares sense) is its truncated SVD. The product is never built:
QR factors of ``up`` and ``down.T`` reduce the SVD to an r x r matrix, so a
Flux-sized module costs about as much as reading it.

The new module keeps the old scale, so ``new_alpha = scale * k`` (or
``scale * sqrt(k)`` with rsLoRA), and the change it makes to the base weight is
the closest rank-k one to the original.
"""

from __future__ import annotations

import json
import os
from collections import Counter
from dataclasses import dataclass

import numpy as np

from . import __version__
from .adapter import Adapter, AdapterError, Module, peft_name
from .st import Tensor, save


@dataclass
class ModuleResult:
    name: str
    status: str
    old_rank: int | None = None
    new_rank: int | None = None
    energy: float | None = None
    alpha: float | None = None


class FactorSVD:
    """SVD of up @ down from its factors: up @ down = (qu @ p) diag(s) (vt @ qd.T).

    The big products are only formed for the k directions that are kept.
    """

    def __init__(self, up: np.ndarray, down: np.ndarray):
        self.qu, ru = np.linalg.qr(up.astype(np.float64))
        self.qd, rd = np.linalg.qr(down.astype(np.float64).T)
        self.p, self.s, self.vt = np.linalg.svd(ru @ rd.T, full_matrices=False)

    def factors(self, k: int) -> tuple[np.ndarray, np.ndarray]:
        """New up (out, k) and down (k, in), splitting each singular value evenly."""
        root = np.sqrt(self.s[:k])
        return self.qu @ (self.p[:, :k] * root), (root[:, None] * self.vt[:k]) @ self.qd.T


def factor_svd(up: np.ndarray, down: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Full U, s, Vt of up @ down, built from its factors."""
    f = FactorSVD(up, down)
    return f.qu @ f.p, f.s, f.vt @ f.qd.T


def choose_rank(
    s: np.ndarray,
    old_rank: int,
    rank: int | None = None,
    energy: float | None = None,
    max_rank: int | None = None,
    min_rank: int = 1,
) -> tuple[int, float]:
    """Pick the new rank and return it with the share of energy it keeps.

    Energy is the squared Frobenius norm of the module's weight change, so 0.95
    keeps the smallest rank whose singular values hold 95% of it.
    """
    power = s.astype(np.float64) ** 2
    total = float(power.sum())
    if rank is not None:
        k = rank
    elif total == 0.0:
        k = min_rank
    else:
        cumulative = np.cumsum(power) / total
        k = int(np.searchsorted(cumulative, energy - 1e-9)) + 1
    if max_rank is not None:
        k = min(k, max_rank)
    k = max(1, min(max(k, min_rank), old_rank))
    kept = float(power[:k].sum()) / total if total else 1.0
    return k, kept


def _resize_module(adapter: Adapter, module: Module, options: dict) -> tuple[ModuleResult, dict[str, Tensor]]:
    tensors = adapter.tensors
    down_key, up_key = module.key("down"), module.key("up")
    down, up = tensors[down_key], tensors[up_key]
    old_rank = down.shape[0]
    alpha, _ = adapter.alpha(module, old_rank)
    scale = adapter.scale(alpha, old_rank)

    if "dora" in module.keys:
        return ModuleResult(module.name, "kept (DoRA)", old_rank, old_rank, 1.0, alpha), {}
    if any(n != 1 for n in up.shape[2:]):
        return ModuleResult(module.name, "kept (up has a kernel)", old_rank, old_rank, 1.0, alpha), {}

    up2 = up.to_float32().reshape(up.shape[0], old_rank)
    down2 = down.to_float32().reshape(old_rank, -1)
    svd = FactorSVD(up2, down2)
    k, kept = choose_rank(svd.s, old_rank, **options)
    if k >= old_rank:
        return ModuleResult(module.name, "kept", old_rank, old_rank, 1.0, alpha), {}

    new_up, new_down = svd.factors(k)
    new_up = new_up.reshape((up.shape[0], k) + tuple(up.shape[2:]))
    new_down = new_down.reshape((k,) + tuple(down.shape[1:]))
    new_alpha = scale * (np.sqrt(k) if adapter.rslora else k)

    changed = {
        down_key: Tensor.from_array(new_down, down.dtype),
        up_key: Tensor.from_array(new_up, up.dtype),
    }
    alpha_key = module.key("alpha")
    if alpha_key is not None:
        old = tensors[alpha_key]
        changed[alpha_key] = Tensor.from_array(np.full(old.shape, new_alpha, dtype=np.float32), old.dtype)
    return ModuleResult(module.name, "resized", old_rank, k, kept, float(new_alpha)), changed


def _clean(x: float) -> float | int:
    rounded = round(x)
    return int(rounded) if abs(x - rounded) < 1e-6 else round(x, 6)


def peft_config(config: dict, results: list[ModuleResult]) -> dict:
    """Rewrite r / lora_alpha, with rank_pattern / alpha_pattern for modules that differ."""
    rows = [(peft_name(r.name), r.new_rank, _clean(r.alpha)) for r in results if r.new_rank is not None]
    new = dict(config)
    if not rows:
        return new
    common_rank = Counter(k for _, k, _ in rows).most_common(1)[0][0]
    common_alpha = Counter(a for _, _, a in rows).most_common(1)[0][0]
    new["r"] = common_rank
    new["lora_alpha"] = common_alpha
    new["rank_pattern"] = {name: k for name, k, _ in rows if k != common_rank}
    new["alpha_pattern"] = {name: a for name, _, a in rows if a != common_alpha}
    return new


def _metadata(adapter: Adapter, results: list[ModuleResult], options: dict) -> dict[str, str]:
    metadata = dict(adapter.metadata)
    for stale in ("sshs_model_hash", "sshs_legacy_hash"):
        metadata.pop(stale, None)
    ranks = {r.new_rank for r in results if r.new_rank is not None}
    alphas = {_clean(r.alpha) for r in results if r.alpha is not None}
    if "ss_network_dim" in metadata:
        metadata["ss_network_dim"] = str(ranks.pop()) if len(ranks) == 1 else "Dynamic"
    if "ss_network_alpha" in metadata:
        metadata["ss_network_alpha"] = str(alphas.pop()) if len(alphas) == 1 else "Dynamic"
    used = {k: v for k, v in options.items() if v is not None}
    metadata["loralens_resize"] = json.dumps(
        {"version": __version__, "source": os.path.basename(adapter.path), **used}, sort_keys=True
    )
    return metadata


def resize(
    adapter: Adapter,
    rank: int | None = None,
    energy: float | None = None,
    max_rank: int | None = None,
    min_rank: int = 1,
) -> tuple[list[ModuleResult], dict[str, Tensor], dict[str, str], dict | None]:
    """Resize every plain LoRA module. Returns results, new tensors, metadata and PEFT config."""
    if (rank is None) == (energy is None):
        raise AdapterError("give exactly one of rank or energy")
    if rank is not None and rank < 1:
        raise AdapterError("rank must be at least 1")
    if energy is not None and not 0.0 < energy <= 1.0:
        raise AdapterError("energy must be above 0 and at most 1")
    if max_rank is not None and max_rank < 1:
        raise AdapterError("max rank must be at least 1")
    if min_rank < 1:
        raise AdapterError("min rank must be at least 1")
    if not adapter.lora_modules():
        raise AdapterError("there are no plain LoRA modules to resize")

    options = {"rank": rank, "energy": energy, "max_rank": max_rank, "min_rank": min_rank}
    results: list[ModuleResult] = []
    changed: dict[str, Tensor] = {}
    for module in adapter.modules:
        if module.kind == "lora":
            result, update = _resize_module(adapter, module, options)
            changed.update(update)
        elif module.kind == "other":
            continue
        else:
            result = ModuleResult(module.name, f"copied ({module.kind})")
        results.append(result)

    tensors = {name: changed.get(name, tensor) for name, tensor in adapter.tensors.items()}
    config = peft_config(adapter.config, results) if adapter.config is not None else None
    return results, tensors, _metadata(adapter, results, options), config


def write(path: str, tensors: dict[str, Tensor], metadata: dict[str, str], config: dict | None) -> list[str]:
    """Write the result. PEFT output is a directory with weights and adapter_config.json."""
    if config is None:
        save(path, tensors, metadata)
        return [path]
    os.makedirs(path, exist_ok=True)
    weights = os.path.join(path, "adapter_model.safetensors")
    config_file = os.path.join(path, "adapter_config.json")
    save(weights, tensors, metadata)
    with open(config_file, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
        f.write("\n")
    return [weights, config_file]


def report(results: list[ModuleResult], before: int, after: int) -> dict:
    lora = [r for r in results if r.old_rank is not None]
    resized = [r for r in lora if r.status == "resized"]
    energies = [r.energy for r in lora]
    worst = sorted(resized, key=lambda r: r.energy)[:5]
    return {
        "modules": len(lora),
        "resized": len(resized),
        "kept": {status: n for status, n in Counter(r.status for r in results if r.status != "resized").items()},
        "old_ranks": dict(Counter(r.old_rank for r in lora)),
        "new_ranks": dict(Counter(r.new_rank for r in lora)),
        "energy_mean": float(np.mean(energies)) if energies else 1.0,
        "energy_min": min(energies) if energies else 1.0,
        "worst": [{"name": r.name, "rank": r.new_rank, "energy": r.energy} for r in worst],
        "bytes_before": before,
        "bytes_after": after,
        "results": [r.__dict__ for r in results],
    }
