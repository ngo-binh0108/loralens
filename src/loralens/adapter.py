"""Load a LoRA file and group its tensors into modules.

Three naming schemes are understood:

* kohya / sd-scripts: ``lora_unet_..._to_q.lora_down.weight``, ``.lora_up.weight``
  and an optional scalar ``.alpha``.
* PEFT: ``base_model.model....q_proj.lora_A.weight`` / ``.lora_B.weight``, with
  alpha in ``adapter_config.json`` (``lora_alpha``, ``alpha_pattern``, ``use_rslora``).
* diffusers: ``....to_q.lora.down.weight`` / ``.lora.up.weight``, or PEFT-style
  keys without a config file.

LyCORIS modules (LoHa, LoKr, OFT) and full ``diff`` weights are recognised so they
can be reported, but only plain LoRA modules can be resized.
"""

from __future__ import annotations

import json
import math
import os
import re
from collections import Counter
from dataclasses import dataclass, field

from .st import SafetensorsError, Tensor, load

_SUFFIXES = [
    (".lora_down.weight", "down"),
    (".lora_up.weight", "up"),
    (".lora_mid.weight", "mid"),
    (".lora_A.weight", "down"),
    (".lora_B.weight", "up"),
    (".lora_B.bias", "bias"),
    (".lora_embedding_A", "down"),
    (".lora_embedding_B", "up"),
    (".lora.down.weight", "down"),
    (".lora.up.weight", "up"),
    (".lora_linear_layer.down.weight", "down"),
    (".lora_linear_layer.up.weight", "up"),
    (".alpha", "alpha"),
    (".dora_scale", "dora"),
    (".lora_magnitude_vector", "dora"),
    (".hada_w1_a", "loha"), (".hada_w1_b", "loha"), (".hada_w2_a", "loha"),
    (".hada_w2_b", "loha"), (".hada_t1", "loha"), (".hada_t2", "loha"),
    (".lokr_w1", "lokr"), (".lokr_w2", "lokr"), (".lokr_w1_a", "lokr"), (".lokr_w1_b", "lokr"),
    (".lokr_w2_a", "lokr"), (".lokr_w2_b", "lokr"), (".lokr_t2", "lokr"),
    (".oft_blocks", "oft"), (".oft_diag", "oft"),
    (".diff", "full"), (".diff_b", "full"),
]
_KOHYA_PREFIXES = ("lora_unet_", "lora_te_", "lora_te1_", "lora_te2_", "lora_te3_", "lora_transformer_")
_PEFT_PREFIX = "base_model.model."


class AdapterError(ValueError):
    """The input can't be read as a LoRA adapter."""


@dataclass
class Module:
    name: str
    keys: dict[str, list[str]] = field(default_factory=dict)

    @property
    def kind(self) -> str:
        roles = self.keys
        for special in ("loha", "lokr", "oft", "full"):
            if special in roles:
                return special
        if "down" in roles and "up" in roles:
            if len(roles["down"]) == 1 and len(roles["up"]) == 1:
                return "lora-tucker" if "mid" in roles else "lora"
        if "down" in roles or "up" in roles:
            return "incomplete"
        return "other"

    def key(self, role: str) -> str | None:
        found = self.keys.get(role)
        return found[0] if found else None


@dataclass
class Adapter:
    path: str
    tensors: dict[str, Tensor]
    metadata: dict[str, str]
    config: dict | None
    config_path: str | None
    modules: list[Module]
    loose: list[str]

    @property
    def format(self) -> str:
        keys = self.tensors.keys()
        kinds = {m.kind for m in self.modules}
        if any(k.startswith(_KOHYA_PREFIXES) for k in keys):
            return "lycoris" if kinds & {"loha", "lokr", "oft"} else "kohya"
        if self.config is not None:
            return "peft"
        if any(".lora.down." in k or ".lora_linear_layer." in k for k in keys):
            return "diffusers"
        if any(".lora_A." in k or k.endswith(".lora_embedding_A") for k in keys):
            if any(k.startswith(_PEFT_PREFIX) for k in keys):
                return "peft"
            return "diffusers"
        return "unknown"

    @property
    def rslora(self) -> bool:
        return bool(self.config and self.config.get("use_rslora"))

    def lora_modules(self) -> list[Module]:
        return [m for m in self.modules if m.kind == "lora"]

    def rank(self, module: Module) -> int:
        return self.tensors[module.key("down")].shape[0]

    def alpha(self, module: Module, rank: int) -> tuple[float, str]:
        """Return the module's alpha and where it came from: file, config or default."""
        key = module.key("alpha")
        if key is not None:
            return float(self.tensors[key].to_float32().reshape(-1)[0]), "file"
        if self.config is not None:
            name = peft_name(module.name)
            pattern = match_pattern(self.config.get("alpha_pattern"), name)
            if pattern is not None:
                return float(pattern), "config"
            if self.config.get("lora_alpha") is not None:
                return float(self.config["lora_alpha"]), "config"
        return float(rank), "default"

    def scale(self, alpha: float, rank: int) -> float:
        return alpha / math.sqrt(rank) if self.rslora else alpha / rank

    def guess_base(self) -> tuple[str | None, str | None]:
        """A best guess at the base model, and what the guess is based on."""
        if self.config and self.config.get("base_model_name_or_path"):
            return str(self.config["base_model_name_or_path"]), "adapter_config.json"
        for key in ("modelspec.architecture", "ss_base_model_version"):
            if self.metadata.get(key):
                return self.metadata[key], f"metadata {key}"
        keys = list(self.tensors)
        if any("double_blocks" in k or "single_blocks" in k or "single_transformer_blocks" in k for k in keys):
            return "Flux", "key names"
        if any("joint_blocks" in k for k in keys):
            return "SD3", "key names"
        if any(k.startswith(("lora_te1_", "lora_te2_")) for k in keys):
            return "SDXL", "key names"
        if any(k.startswith("lora_unet_") for k in keys):
            return "SD 1.x / 2.x", "key names"
        return None, None


def peft_name(module_name: str) -> str:
    """The module name PEFT matches rank_pattern and alpha_pattern against."""
    if module_name.startswith(_PEFT_PREFIX):
        return module_name[len(_PEFT_PREFIX):]
    return module_name


def match_pattern(patterns: dict | None, name: str):
    """PEFT's lookup: a pattern matches the end of the name, on a dot boundary."""
    for pattern, value in (patterns or {}).items():
        if re.match(rf"(.*\.)?(?:{pattern})$", name):
            return value
    return None


def split_key(key: str) -> tuple[str | None, str | None]:
    for suffix, role in _SUFFIXES:
        if key.endswith(suffix):
            return key[: -len(suffix)], role
    return None, None


def group_modules(keys) -> tuple[list[Module], list[str]]:
    modules: dict[str, Module] = {}
    loose: list[str] = []
    for key in keys:
        name, role = split_key(key)
        if name is None:
            loose.append(key)
            continue
        module = modules.setdefault(name, Module(name))
        module.keys.setdefault(role, []).append(key)
    return list(modules.values()), loose


def resolve_paths(path: str, config_path: str | None) -> tuple[str, str | None]:
    if os.path.isdir(path):
        weights = os.path.join(path, "adapter_model.safetensors")
        if not os.path.exists(weights):
            if os.path.exists(os.path.join(path, "adapter_model.bin")):
                raise AdapterError(f"{path} has adapter_model.bin (a pickle); only .safetensors is supported")
            raise AdapterError(f"{path} has no adapter_model.safetensors")
        path = weights
    elif not os.path.exists(path):
        raise AdapterError(f"no such file: {path}")
    if path.lower().endswith((".bin", ".pt", ".pth", ".ckpt")):
        raise AdapterError(f"{path} is a pickle file; only .safetensors is supported")
    if config_path is None:
        sibling = os.path.join(os.path.dirname(os.path.abspath(path)), "adapter_config.json")
        if os.path.exists(sibling):
            config_path = sibling
    return path, config_path


def load_adapter(path: str, config_path: str | None = None) -> Adapter:
    explicit_config = config_path is not None
    path, config_path = resolve_paths(path, config_path)
    try:
        tensors, metadata = load(path)
    except SafetensorsError as e:
        raise AdapterError(str(e)) from None

    modules, loose = group_modules(tensors)
    if not any(m.kind != "other" for m in modules):
        raise AdapterError(f"{path} has no LoRA tensors (no lora_down/lora_up or lora_A/lora_B keys)")

    config = None
    if config_path is not None:
        peft_keys = any(".lora_A." in k or ".lora_embedding_A" in k for k in tensors)
        if explicit_config or peft_keys:
            try:
                with open(config_path, encoding="utf-8") as f:
                    config = json.load(f)
            except OSError as e:
                raise AdapterError(f"can't read {config_path}: {e.strerror}") from None
            except json.JSONDecodeError as e:
                raise AdapterError(f"{config_path} is not valid JSON ({e})") from None
            if not isinstance(config, dict):
                raise AdapterError(f"{config_path} is not a JSON object")
        else:
            config_path = None

    adapter = Adapter(path, tensors, metadata, config, config_path, modules, loose)
    for module in adapter.lora_modules():
        down = tensors[module.key("down")]
        up = tensors[module.key("up")]
        if len(down.shape) < 2 or len(up.shape) < 2 or down.shape[0] != up.shape[1]:
            raise AdapterError(
                f"{module.name}: down {list(down.shape)} and up {list(up.shape)} don't share a rank"
            )
    return adapter


def target_groups(adapter: Adapter) -> Counter:
    """Count LoRA modules by what they attach to: lora_unet/lora_te1 for kohya, q_proj/to_k otherwise."""
    groups: Counter = Counter()
    for module in adapter.modules:
        if module.kind == "other":
            continue
        if module.name.startswith(_KOHYA_PREFIXES):
            groups["_".join(module.name.split("_")[:2])] += 1
        else:
            groups[module.name.rsplit(".", 1)[-1]] += 1
    return groups
