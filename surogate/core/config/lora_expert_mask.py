"""`lora_expert_mask`: expert LoRA on a chosen subset of each MoE layer's routed experts.

Values:

* ``all`` (or unset): every routed expert of every MoE layer gets expert LoRA, as without the key.
* ``none``: no routed expert does. The shared-expert, attention, linear-attention and router adapters are
  unaffected, so this freezes the routed experts while keeping the rest of the target set.
* a path to a JSON mask file (or the same object inline)::

      {"format": "surogate-expert-mask-v1", "num_layers": 40, "num_experts": 256,
       "layers": {"0": [17, 102, 3], "1": [44, 9], "2": [], ...}}

  ``layers`` maps a layer index to the global ids of the experts that get LoRA; a layer that is not listed
  gets none. ``num_layers``/``num_experts`` are optional and, when present, must match the model. Other keys
  (score, p, source, ...) are carried along unread. An exported adapter's ``surogate_expert_mask`` entry has
  this format, so it can be fed back as a mask.

The trainer allocates the selected experts' adapters only ("compact": A [K_l, r, in], B [K_l, out, r]) and
exports them under their global ids (``mlp.experts.{id}.{proj}``).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

MASK_FORMAT = "surogate-expert-mask-v1"


@dataclass(frozen=True)
class LoRAExpertMask:
    """A parsed mask: the listed layers' expert ids (sorted, unique); unlisted layers select none."""

    source: str
    layers: dict[int, tuple[int, ...]] = field(default_factory=dict)
    declared_num_layers: int | None = None
    declared_num_experts: int | None = None

    def experts(self, layer: int) -> tuple[int, ...]:
        return self.layers.get(layer, ())

    def per_layer(self) -> list[list[int]]:
        """Layer lists up to the highest listed layer (what LoRAAdapterConfig.set_expert_mask takes)."""
        if not self.layers:
            return []
        return [list(self.layers.get(layer, ())) for layer in range(max(self.layers) + 1)]

    def total(self) -> int:
        return sum(len(ids) for ids in self.layers.values())

    def to_json(self, num_layers: int, num_experts: int) -> dict[str, Any]:
        return {"format": MASK_FORMAT, "num_layers": num_layers, "num_experts": num_experts,
                "source": self.source,
                "layers": {str(layer): list(self.experts(layer)) for layer in range(num_layers)}}


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _parse_layers(raw: Any, source: str) -> dict[int, tuple[int, ...]]:
    if isinstance(raw, list):
        raw = {str(i): ids for i, ids in enumerate(raw)}
    if not isinstance(raw, dict):
        raise ValueError(f"lora_expert_mask ({source}): 'layers' must map layer indices to expert id lists")
    layers: dict[int, tuple[int, ...]] = {}
    for key, ids in raw.items():
        try:
            layer = int(key)
        except (TypeError, ValueError):
            raise ValueError(f"lora_expert_mask ({source}): layer key {key!r} is not an integer") from None
        if layer < 0 or str(layer) != str(key).strip():
            raise ValueError(f"lora_expert_mask ({source}): layer key {key!r} is not a layer index")
        if layer in layers:
            raise ValueError(f"lora_expert_mask ({source}): layer {layer} is listed twice")
        if not isinstance(ids, (list, tuple)) or not all(_is_int(e) for e in ids):
            raise ValueError(f"lora_expert_mask ({source}): layer {layer} must list integer expert ids")
        if len(set(ids)) != len(ids):
            dup = sorted({e for e in ids if list(ids).count(e) > 1})
            raise ValueError(f"lora_expert_mask ({source}): layer {layer} lists expert(s) {dup} more than once")
        if any(e < 0 for e in ids):
            raise ValueError(f"lora_expert_mask ({source}): layer {layer} lists a negative expert id")
        layers[layer] = tuple(sorted(ids))
    return layers


def mask_from_object(obj: Any, source: str) -> LoRAExpertMask:
    """A mask from the JSON object of a mask file (or an inline mapping)."""
    if not isinstance(obj, dict):
        raise ValueError(f"lora_expert_mask ({source}): expected a JSON object with a 'layers' entry")
    fmt = obj.get("format", MASK_FORMAT)
    if fmt != MASK_FORMAT:
        raise ValueError(f"lora_expert_mask ({source}): unknown format {fmt!r} (expected {MASK_FORMAT!r})")
    if "layers" not in obj:
        raise ValueError(f"lora_expert_mask ({source}): the mask has no 'layers' entry")
    declared = {}
    for key in ("num_layers", "num_experts"):
        if obj.get(key) is not None:
            if not _is_int(obj[key]) or obj[key] <= 0:
                raise ValueError(f"lora_expert_mask ({source}): {key} must be a positive integer")
            declared[key] = obj[key]
    return LoRAExpertMask(source=source, layers=_parse_layers(obj["layers"], source),
                          declared_num_layers=declared.get("num_layers"),
                          declared_num_experts=declared.get("num_experts"))


def parse_lora_expert_mask(value: Any) -> LoRAExpertMask | None:
    """Parse the config value: None for every expert (``all``/unset), else the mask."""
    if value is None:
        return None
    if isinstance(value, LoRAExpertMask):
        return value
    if isinstance(value, dict):
        return mask_from_object(value, "inline")
    if not isinstance(value, (str, Path)):
        raise ValueError(f"lora_expert_mask must be 'all', 'none' or a mask file path, got {value!r}")
    text = str(value).strip()
    if text.lower() == "all":
        return None
    if text.lower() == "none":
        return LoRAExpertMask(source="none")
    path = Path(text).expanduser()
    if not path.is_file():
        raise ValueError(f"lora_expert_mask: '{text}' is neither 'all', 'none' nor an existing mask file")
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"lora_expert_mask ({path}): not valid JSON: {exc}") from None
    return mask_from_object(obj, str(path.resolve()))


def validate_lora_expert_mask(mask: LoRAExpertMask, *, num_layers: int, num_experts: int,
                              moe_layers: set[int] | None = None) -> None:
    """Check a mask against the model: layer indices in [0, num_layers), expert ids in [0, num_experts)."""
    src = mask.source
    if num_experts <= 0:
        raise ValueError(f"lora_expert_mask ({src}): the model has no routed experts")
    if mask.declared_num_layers is not None and mask.declared_num_layers != num_layers:
        raise ValueError(f"lora_expert_mask ({src}): the mask is for {mask.declared_num_layers} layers, "
                         f"the model has {num_layers}")
    if mask.declared_num_experts is not None and mask.declared_num_experts != num_experts:
        raise ValueError(f"lora_expert_mask ({src}): the mask is for {mask.declared_num_experts} experts per "
                         f"layer, the model has {num_experts}")
    for layer, ids in sorted(mask.layers.items()):
        if layer >= num_layers:
            raise ValueError(f"lora_expert_mask ({src}): lists layer {layer}, but the model has {num_layers} "
                             f"layers (0..{num_layers - 1})")
        bad = [e for e in ids if e >= num_experts]
        if bad:
            raise ValueError(f"lora_expert_mask ({src}): layer {layer} lists expert(s) {bad}, but the model has "
                             f"{num_experts} experts per layer (0..{num_experts - 1})")
        if ids and moe_layers is not None and layer not in moe_layers:
            raise ValueError(f"lora_expert_mask ({src}): layer {layer} lists experts, but it has no routed experts")


def check_lora_expert_mask_options(mask: LoRAExpertMask, *, lora: bool, target_modules, ep_size: int,
                                   selective_expert_dequant: bool, rollout_parity: bool = False) -> None:
    """Refuse the run options whose expert indexing the compact expert adapters cannot share."""
    src = mask.source
    if not lora:
        raise ValueError(f"lora_expert_mask ({src}) requires lora: true")
    if int(ep_size or 1) > 1:
        raise ValueError(f"lora_expert_mask ({src}) is not supported with ep_size > 1 (expert parallelism / LLEP)")
    if selective_expert_dequant:
        raise ValueError(f"lora_expert_mask ({src}) is not supported with qlora_selective_expert_dequant: true")
    if rollout_parity:
        raise ValueError(f"lora_expert_mask ({src}) is not supported with rollout parity")
    if "gate_up_proj" in set(target_modules or ()):
        raise ValueError(f"lora_expert_mask ({src}) is not supported with gate_up_proj LoRA targets")


# ------------------------------------------------------------------ parameter counts (Qwen3.5 MoE)


def _text_config(config: dict) -> dict:
    return config.get("text_config", config)


def qwen3_5_moe_lora_counts(config: dict, *, rank: int, target_modules, train_router: bool,
                            mask: LoRAExpertMask | None = None) -> tuple[int, int]:
    """(parameters, tensors) of the adapter the trainer allocates for a Qwen3.5/3.6 MoE checkpoint config.

    The optimizer's own count ("Flash AdamW 8-bit optimizer: T tensors, P params"): one tensor per lora_A /
    lora_B, a grouped expert projection counting once per layer. It mirrors ModularLoRAWeightsManager: q/k/v/o
    on the full-attention layers (q with its gate, 2 x heads x head_dim); in_proj_qkv/z/a/b and out_proj on
    the linear-attention layers; up/down on the shared expert (never gate_proj); gate/up/down on the routed
    experts selected by `mask` (all without one); the router LoRA (rank r, mlp.gate) with train_router. `out_proj`
    without `o_proj` names the linear-attention output only, as the trainer resolves it.
    """
    t = _text_config(config)
    names = set(target_modules or ())
    if "all" in names:
        names = {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
                 "in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b", "out_proj"}
    r = int(rank)
    c = int(t["hidden_size"])
    layers = int(t["num_hidden_layers"])
    types = t.get("layer_types") or [
        "full_attention" if (i + 1) % int(t.get("full_attention_interval", 4)) == 0 else "linear_attention"
        for i in range(layers)]
    hq, hkv, hd = int(t["num_attention_heads"]), int(t["num_key_value_heads"]), int(t["head_dim"])
    hk, dk = int(t["linear_num_key_heads"]), int(t["linear_key_head_dim"])
    hv, dv = int(t["linear_num_value_heads"]), int(t["linear_value_head_dim"])
    e = int(t["num_experts"])
    d_moe = int(t["moe_intermediate_size"])
    d_shared = int(t.get("shared_expert_intermediate_size") or 0)

    def pair(fan_in: int, fan_out: int) -> int:
        return r * fan_in + fan_out * r

    attention = {"q_proj": pair(c, 2 * hq * hd), "k_proj": pair(c, hkv * hd), "v_proj": pair(c, hkv * hd),
                 "o_proj": pair(hq * hd, c)}
    linear = {"in_proj_qkv": pair(c, 2 * hk * dk + hv * dv), "in_proj_z": pair(c, hv * dv),
              "in_proj_a": pair(c, hv), "in_proj_b": pair(c, hv), "out_proj": pair(hv * dv, c)}
    expert = {"gate_proj": pair(c, d_moe), "up_proj": pair(c, d_moe), "down_proj": pair(d_moe, c)}
    shared = {"up_proj": pair(c, d_shared), "down_proj": pair(d_shared, c)} if d_shared > 0 else {}

    params = tensors = 0
    for layer in range(layers):
        if types[layer] == "full_attention":
            for name, n in attention.items():
                # out_proj alone names the linear-attention output; it adds the attention output only
                # together with o_proj.
                if name in names:
                    params, tensors = params + n, tensors + 2
        else:
            for name, n in linear.items():
                if name in names:
                    params, tensors = params + n, tensors + 2
        k = e if mask is None else len(mask.experts(layer))
        if k:
            for name, n in expert.items():
                if name in names:
                    params, tensors = params + n * k, tensors + 2
        for name, n in shared.items():
            if name in names:
                params, tensors = params + n, tensors + 2
        if train_router:
            params, tensors = params + pair(c, e), tensors + 2
    return params, tensors
