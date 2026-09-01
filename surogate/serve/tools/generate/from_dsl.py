"""Reads a serve target's shape from the training DSL declaration.

This is the join that makes the DSL the single source of truth for model
architecture. `surogate/dsl/models/*.py` already declares every quantity a serve
target's `config.h` states — hidden size, layer schedule, head geometry, the
gated-delta-net dims — and the training runtime compiles that declaration to IR
on every run, so the values are not merely written down somewhere else: they are
the values training actually uses. Restating them in C++ is what allows the two
to drift.

**Two inputs, one source of truth.** `from_dsl` takes the architecture name and
the checkpoint's own `config.json`, exactly as `ir_builder.build_dsl_ir_for_model`
does. The declaration owns the architecture — what the parameters are, how the
layers are scheduled, what is derived from what. The config owns this instance's
sizing — which 0.8B, which 4B. A handful of scalars (`rope_theta`) are pure
pass-throughs that the declaration neither derives nor interprets; those are read
straight from the config and marked as such below, rather than being invented
here.

**Correctness is proven by regeneration, not review.** `check_roundtrip.py`
feeds these specs to the emitters and diffs against the committed targets. A
declaration that cannot reproduce the hand-written C++ byte for byte is wrong
about the model, and that is a far more useful thing to learn from a diff than
from a production incident.
"""

from __future__ import annotations

import json
import re
from typing import Any

from target_spec import AttentionSpec, LinearAttentionSpec, LoraSlice, ParamSpec, TargetSpec

#: Quantities the declaration does not derive — read verbatim from the
#: checkpoint config. Keep this list short and explicit: every entry is a place
#: where the DSL is not yet the source of truth.
CONFIG_PASSTHROUGH = ("rope_theta",)

_LAYER_INDEX = re.compile(r"blocks\[(\d+)\]\.(.+)")


def _compile(architecture: str, hf_config: dict[str, Any]) -> dict[str, Any]:
    from surogate.dsl.py_compiler import compile_model_for_hf

    raw = compile_model_for_hf(architecture, hf_config)
    ir = json.loads(raw) if isinstance(raw, str) else raw
    if not ir.get("success"):
        raise ValueError(f"DSL compilation failed for {architecture}: {ir.get('errors')}")
    return ir


def _module(ir: dict[str, Any]) -> dict[str, Any]:
    modules = ir.get("modules") or []
    if not modules:
        raise ValueError("DSL IR carries no modules")
    return modules[0]


def _config(ir: dict[str, Any]) -> dict[str, Any]:
    return ir.get("config") or (_module(ir).get("config") or {})


def _nested(config: dict[str, Any], dotted: str) -> Any:
    """`text_config.rope_parameters.rope_theta`-style lookup, the way the
    `@nn.hf_config` decorator resolves its own mappings."""

    if dotted in config:
        return config[dotted]
    cursor: Any = config
    for part in dotted.split("."):
        if not isinstance(cursor, dict) or part not in cursor:
            return None
        cursor = cursor[part]
    return cursor


def _passthrough(hf_config: dict[str, Any], key: str) -> Any:
    """A scalar the declaration does not carry, from the checkpoint config.

    Tries the nested spellings the Qwen family uses before the bare key, so a
    text-config-nested model resolves without a per-model table.
    """

    for candidate in (f"text_config.rope_parameters.{key}", f"rope_parameters.{key}",
                      f"text_config.{key}", key):
        value = _nested(hf_config, candidate)
        if value is not None:
            return value
    return None


def _hf_name(hf_mapping: dict[str, Any], dsl_name: str) -> str | None:
    """The checkpoint path for one canonical parameter name.

    Per-layer params arrive as `blocks[7].full_q_proj_weight`; the mapping may
    carry that exact key (the hybrid expansion emits physical indices) or the
    bare name with a `{layer}` placeholder. Only plain-string mappings resolve to
    a single path — `fuse`/`split`/`stack_experts` specs describe a
    transformation, and flattening them to one name here would be a lie.
    """

    direct = hf_mapping.get(dsl_name)
    if isinstance(direct, str):
        return direct
    match = _LAYER_INDEX.match(dsl_name)
    if match:
        layer, field = match.group(1), match.group(2)
        template = hf_mapping.get(field)
        if isinstance(template, str):
            return template.replace("{layer}", layer)
    return None


def _params(ir: dict[str, Any]) -> tuple[ParamSpec, ...]:
    module = _module(ir)
    hf_mapping = module.get("hf_mapping") or ir.get("hf_mapping") or {}
    forward = module.get("forward") or {}
    declared = {**(module.get("params") or {}), **(forward.get("params") or {})}

    specs: list[ParamSpec] = []
    for dsl_name, entry in declared.items():
        if not isinstance(entry, dict):
            continue
        slices = tuple(
            LoraSlice(
                name=str(target.get("name", "")),
                offset=int(target.get("offset", 0)),
                size=int(target.get("size", 0)),
            )
            for target in (entry.get("lora_targets") or [])
        )
        specs.append(
            ParamSpec(
                dsl_name=dsl_name,
                hf_name=_hf_name(hf_mapping, dsl_name),
                shape=tuple(str(dim) for dim in (entry.get("shape") or ())),
                lora=slices,
            )
        )
    specs.sort(key=lambda p: p.dsl_name)
    return tuple(specs)


def from_dsl(
    architecture: str,
    hf_config: dict[str, Any],
    *,
    name: str,
    **overrides: Any,
) -> TargetSpec:
    """Project a training DSL declaration onto a serve `TargetSpec`.

    `name` is the serve target's C++ namespace — a deployment choice with no
    counterpart in the declaration, so it is passed rather than derived.
    `overrides` cover serving policy that the architecture does not determine
    (`mtp_draft_tokens`, for instance).
    """

    ir = _compile(architecture, hf_config)
    config = _config(ir)

    def need(key: str) -> Any:
        if key not in config:
            raise ValueError(
                f"the {architecture} declaration does not surface {key!r} in its runtime "
                f"config; it must be an int/bool attribute on the model instance for the "
                f"serve contract to read it"
            )
        return config[key]

    head_dim = int(config.get("head_size") or config.get("D") or 0)
    attention = AttentionSpec(
        query_heads=int(need("num_query_heads")),
        kv_heads=int(need("num_kv_heads")),
        head_dim=head_dim,
        rotary_dim=int(need("rotary_dim")),
    )

    linear_attention = None
    if config.get("has_linear_blocks"):
        linear_attention = LinearAttentionSpec(
            key_heads=int(need("linear_num_key_heads")),
            key_head_dim=int(need("linear_key_head_dim")),
            value_heads=int(need("linear_num_value_heads")),
            value_head_dim=int(need("linear_value_head_dim")),
            conv_kernel=int(need("linear_conv_kernel_dim")),
        )

    rope_theta = overrides.pop("rope_theta", None)
    if rope_theta is None:
        rope_theta = _passthrough(hf_config, "rope_theta")
        if rope_theta is None:
            raise ValueError(
                "rope_theta is a config pass-through and is absent from this checkpoint config"
            )

    token_domain = overrides.pop("token_domain", 0)
    spec = TargetSpec(
        name=name,
        token_domain=int(token_domain),
        hidden=int(need("d_model")),
        layers=int(need("n_layers")),
        intermediate=int(need("d_ff")),
        vocab=int(need("vocab_size")),
        rms_epsilon=float(need("eps")),
        rope_theta=float(rope_theta),
        attention=attention,
        linear_attention=linear_attention,
        native_context=int(config.get("max_seq") or TargetSpec.native_context),
        attention_interval=int(config.get("full_attention_interval", 4)),
        params=_params(ir),
        **overrides,
    )
    spec.validate()
    return spec


def from_model_dir(model_dir: str, *, name: str, **overrides: Any) -> TargetSpec:
    """`from_dsl` for a checkpoint on disk, resolving its architecture the same
    way the training entry points do."""

    from surogate.dsl.ir_builder import load_hf_config, resolve_architecture

    hf_config = load_hf_config(model_dir)
    return from_dsl(resolve_architecture(hf_config), hf_config, name=name, **overrides)
