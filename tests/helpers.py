"""Build small synthetic LoRA files for the tests."""

from __future__ import annotations

import json
import os

import numpy as np

from loralens.st import Tensor, save


def low_rank_pair(out_dim, in_dim, rank, true_rank, seed=0, singular=None):
    """up (out, rank) and down (rank, in) whose product has rank `true_rank`.

    With `singular`, the product's non-zero singular values are exactly those.
    """
    rng = np.random.default_rng(seed)
    if singular is None:
        singular = np.linspace(3.0, 1.0, true_rank)
    singular = np.asarray(singular, dtype=np.float64)
    u, _ = np.linalg.qr(rng.standard_normal((out_dim, true_rank)))
    v, _ = np.linalg.qr(rng.standard_normal((in_dim, true_rank)))
    up = np.zeros((out_dim, rank))
    down = np.zeros((rank, in_dim))
    up[:, :true_rank] = u * np.sqrt(singular)
    down[:true_rank] = (v * np.sqrt(singular)).T
    mix, _ = np.linalg.qr(rng.standard_normal((rank, rank)))
    return (up @ mix).astype(np.float32), (mix.T @ down).astype(np.float32)


def write_kohya(path, modules, dtype="F32", metadata=None, extra=None):
    """modules: {name: (up, down, alpha or None)}"""
    tensors = {}
    for name, (up, down, alpha) in modules.items():
        tensors[f"{name}.lora_down.weight"] = Tensor.from_array(down, dtype)
        tensors[f"{name}.lora_up.weight"] = Tensor.from_array(up, dtype)
        if alpha is not None:
            tensors[f"{name}.alpha"] = Tensor.from_array(np.float32(alpha), dtype)
    tensors.update(extra or {})
    save(str(path), tensors, metadata)
    return str(path)


def write_peft(folder, modules, config, dtype="F32"):
    """modules: {module name without base_model.model.: (up, down)}"""
    os.makedirs(folder, exist_ok=True)
    tensors = {}
    for name, (up, down) in modules.items():
        tensors[f"base_model.model.{name}.lora_A.weight"] = Tensor.from_array(down, dtype)
        tensors[f"base_model.model.{name}.lora_B.weight"] = Tensor.from_array(up, dtype)
    save(os.path.join(folder, "adapter_model.safetensors"), tensors)
    with open(os.path.join(folder, "adapter_config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f)
    return str(folder)
