"""Shared helpers for the `tensor-*` debug subcommands.

These subcommands introspect the compiled phase tree / regions / arena /
layout produced by `_surogate.SurogateTrainer`'s constructor + first
compile pass. Unlike `activations` / `gradients`, they do not drive a
training step — they just need the graph compiled.

Build pattern:
    trainer = build_introspection_trainer(resolved)
    layout = trainer.get_debug_tensor_layout()
    ...

The trainer is built with `ngpu=1` regardless of the config's `gpus` field;
single-rank introspection is sufficient for layout auditing (layout is
rank-identical by construction, per design/buffer-runtime-v4.md §Determinism).
Weights are NOT imported — compile only needs shapes + options.
"""

from __future__ import annotations

from typing import Any

from surogate.utils.logger import get_logger
from surogate.utils.tensor import to_surogate_dtype

from ._shared import ResolvedModel

logger = get_logger()


def build_introspection_trainer(resolved: ResolvedModel) -> Any:
    """Construct an `_surogate.SurogateTrainer` tuned for debug introspection.

    Runs single-rank, honors the user's (B, T), LoRA, QLoRA, and recipe
    settings so the compiled layout matches production. Does NOT import
    weights — the first `get_debug_*` call triggers the compile, which
    only needs shapes + options.

    `load_config` does NOT invoke `SFTConfig.__post_init__` (only
    `TokenizeDatasets.__init__` does, per surogate/train/tokenize.py:540).
    Call it explicitly here so `runtime_config` / `lora_config` /
    `qlora_config` / `model_dir` / `torch_dtype` are populated.
    """
    from surogate import _surogate

    config = resolved.config
    if getattr(config, "runtime_config", None) is None:
        config.__post_init__()
    options = config.runtime_config
    if options is None:
        raise RuntimeError("config.__post_init__ did not populate `runtime_config`")

    # Build + wire the DSL IR JSON (trainer.py:50-57 pattern). Without it
    # the C++ `DslModel` constructor raises.
    from surogate.dsl.ir_builder import build_dsl_ir_for_model

    dsl_extra = {}
    if getattr(config, "ep_size", 1) > 1:
        dsl_extra["ep_size"] = config.ep_size
    ir_json = build_dsl_ir_for_model(config.model_dir, extra_config=dsl_extra or None)
    options.dsl_ir_json = ir_json

    # Single-rank introspection — layout is rank-identical by construction.
    ngpu = 1

    dtype = to_surogate_dtype(config.torch_dtype)
    pretrained = _surogate.PretrainedConfig.from_pretrained(config.model_dir, dtype)

    lora_config = getattr(config, "lora_config", None)
    qlora_config = getattr(config, "qlora_config", None)

    trainer = _surogate.SurogateTrainer(
        ngpu=ngpu,
        config=pretrained,
        options=options,
        batch_size=int(config.per_device_train_batch_size),
        seq_len=int(config.sequence_len),
        grad_accum=1,  # debug introspection — no accumulation needed
        memcpy_all_gather=bool(getattr(config, "memcpy_all_gather", True)),
        memcpy_send_recv=bool(getattr(config, "memcpy_send_recv", True)),
        lora_config=lora_config,
        qlora_config=qlora_config,
    )
    return trainer


def write_run_and_model_records(writer: Any, resolved: ResolvedModel, subcommand: str) -> None:
    """Emit the shared RUN + MODEL header records every subcommand writes.

    Captures enough config metadata (architecture, model dir, B, T, recipe,
    LoRA/QLoRA toggles) to reproduce the introspection context. Subcommands
    add their own tag-specific records afterwards.
    """
    from .schema import Tag

    config = resolved.config
    # `config.recipe` is populated from YAML before __post_init__ runs; prefer
    # that to `runtime_config.recipe_name` which isn't available until trainer
    # construction.
    recipe = getattr(config, "recipe", "?") or "?"
    writer.write(
        Tag.RUN,
        subcommand=subcommand,
        architecture=resolved.architecture,
        model_id=resolved.model_id,
        model_dir=resolved.model_dir,
        batch_size=int(config.per_device_train_batch_size),
        seq_len=int(config.sequence_len),
        recipe=recipe,
        lora=bool(getattr(config, "lora", False)),
        qlora_bnb=bool(getattr(config, "qlora_bnb", False)),
        qlora_fp8=bool(getattr(config, "qlora_fp8", False)),
        qlora_fp4=bool(getattr(config, "qlora_fp4", False)),
    )
    writer.write(
        Tag.MODEL,
        architecture=resolved.architecture,
        hidden_size=resolved.hf_config.get("hidden_size"),
        num_hidden_layers=resolved.hf_config.get("num_hidden_layers"),
        num_attention_heads=resolved.hf_config.get("num_attention_heads"),
        num_key_value_heads=resolved.hf_config.get("num_key_value_heads"),
        intermediate_size=resolved.hf_config.get("intermediate_size"),
        vocab_size=resolved.hf_config.get("vocab_size"),
    )
