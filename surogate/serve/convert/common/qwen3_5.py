"""Resolve the hybrid Qwen checkpoint before building any serving objects."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass, field
import math

from surogate.serve.artifact.geometry import validate_geometry, validate_resolved_geometry
from .checkpoint import positive_int
from .declaration import Declaration, declare


@dataclass(frozen=True)
class Geometry:
    declared: Declaration = field(repr=False, compare=False)
    layers: int
    hidden: int
    intermediate: int
    vocab: int
    query_heads: int
    kv_heads: int
    head_dim: int
    gdn_key_heads: int
    gdn_key_head_dim: int
    gdn_value_heads: int
    gdn_value_head_dim: int
    gdn_conv_kernel: int
    layer_types: tuple[str, ...]
    rotary_dim: int
    rms_epsilon: float
    rope_theta: float
    max_context: int
    mtp_layers: int
    token_domain: int
    experts: int = 0
    experts_per_token: int = 0
    shared_intermediate: int = 0
    observed_scope: object = field(default=None, repr=False, compare=False)

    @property
    def text_config(self):
        config = self.declared.hf_config
        return config.get("text_config", config)

    @property
    def model_type(self):
        return self.declared.hf_config.get("model_type")

    @property
    def tied_embeddings(self):
        return self.text_config["tie_word_embeddings"]

    @property
    def query_size(self):
        return self.query_heads * self.head_dim

    @property
    def kv_size(self):
        return self.kv_heads * self.head_dim

    @property
    def key_dim(self):
        return self.gdn_key_heads * self.gdn_key_head_dim

    @property
    def value_dim(self):
        return self.gdn_value_heads * self.gdn_value_head_dim

    @property
    def convolution_dim(self):
        return 2 * self.key_dim + self.value_dim

    @property
    def full_attention_layers(self):
        return tuple(i for i, kind in enumerate(self.layer_types) if kind == "full_attention")

    @property
    def gdn_layers(self):
        return tuple(i for i, kind in enumerate(self.layer_types) if kind == "linear_attention")

    @property
    def draft_vocab(self):
        # Conversion policy: at most this many shortlist entries, bounded by the checkpoint.
        return min(131072, self.token_domain)


def geometry_from_config(config: Mapping, *, mixture: bool = False, token_domain: int | None = None,
                         architecture_base: str | None = None) -> Geometry:
    source = deepcopy(dict(config))
    text = source.get("text_config", source)
    if not isinstance(text, dict):
        raise ValueError("config.text_config must be an object")
    dimensions = ("hidden_size", "num_hidden_layers", "vocab_size", "num_attention_heads",
                  "num_key_value_heads", "head_dim", "linear_num_key_heads",
                  "linear_key_head_dim", "linear_num_value_heads", "linear_value_head_dim",
                  "linear_conv_kernel_dim", "max_position_embeddings")
    for name in dimensions:
        positive_int(text, name)
    if text["linear_conv_kernel_dim"] != 4:
        raise ValueError("serving currently supports linear_conv_kernel_dim=4")
    width = "moe_intermediate_size" if mixture else "intermediate_size"
    positive_int(text, width)
    if mixture:
        for name in ("num_experts", "num_experts_per_tok", "shared_expert_intermediate_size"):
            positive_int(text, name)
        if text["num_experts_per_tok"] > text["num_experts"]:
            raise ValueError("num_experts_per_tok exceeds num_experts")
        if text.get("norm_topk_prob", True) is not True:
            raise ValueError("this target requires normalized top-k routing")
    elif text.get("num_experts", 0):
        raise ValueError("a mixture checkpoint requires the mixture target")
    if text["num_hidden_layers"] > 256:
        raise ValueError("this target supports at most 256 text layers")
    if text["num_attention_heads"] % text["num_key_value_heads"]:
        raise ValueError("query heads must be divisible by KV heads")
    if text["linear_num_value_heads"] % text["linear_num_key_heads"]:
        raise ValueError("linear value heads must be divisible by linear key heads")
    if text.get("attention_bias", False):
        raise ValueError("this target does not implement attention projection biases")
    if text.get("hidden_act", "silu") != "silu":
        raise ValueError("this target requires the SiLU activation")
    text["tie_word_embeddings"] = text.get("tie_word_embeddings", source.get("tie_word_embeddings", False))
    if not isinstance(text["tie_word_embeddings"], bool):
        raise ValueError("tie_word_embeddings must be boolean")
    layers = text["num_hidden_layers"]
    schedule = text.get("layer_types")
    if schedule is None:
        interval = positive_int(text, "full_attention_interval")
        schedule = ["full_attention" if (i + 1) % interval == 0 else "linear_attention"
                    for i in range(layers)]
    if not isinstance(schedule, list) or len(schedule) != layers or any(
        item not in ("full_attention", "linear_attention") for item in schedule
    ):
        raise ValueError("layer_types must declare one full or linear attention kind per layer")
    text["layer_types"] = schedule
    rope = text.get("rope_parameters")
    if not isinstance(rope, dict):
        raise ValueError("rope_parameters is required")
    if rope.get("rope_type", "default") != "default":
        raise ValueError("this target implements default MRoPE only")
    for label, value in (("rms_norm_eps", text.get("rms_norm_eps")),
                         ("rope_theta", rope.get("rope_theta")),
                         ("partial_rotary_factor", rope.get("partial_rotary_factor"))):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"{label} must be finite and positive")
    factor = rope["partial_rotary_factor"]
    rotary = int(text["head_dim"] * factor)
    if factor > 1 or rotary <= 0 or rotary % 2:
        raise ValueError("partial_rotary_factor must produce a positive even rotary width within head_dim")
    sections = rope.get("mrope_section")
    # The current position kernel implements this interleaved MRoPE partition.
    # This is a kernel capability, independent of the checkpoint's model name or width.
    if sections != [11, 11, 10] or rotary != 64 or rope.get("mrope_interleaved", True) is not True:
        raise ValueError("this target requires interleaved MRoPE with sections [11, 11, 10] and rotary_dim 64")
    mtp = text.get("mtp_num_hidden_layers", 0)
    if isinstance(mtp, bool) or not isinstance(mtp, int) or mtp not in (0, 1):
        raise ValueError("this target supports zero or one MTP layer")
    vision_geometry_block(source, text_hidden=text["hidden_size"])
    base = architecture_base or ("Qwen3_5Moe" if mixture else "Qwen3_5")
    architecture = base + ("ForConditionalGeneration" if "text_config" in source else "ForCausalLM")
    named = source.get("architectures", [architecture])
    if not isinstance(named, list) or not named or named[0] not in (
        base + "ForConditionalGeneration", base + "ForCausalLM"
    ):
        raise ValueError(f"checkpoint does not declare the {base} architecture")
    # The wrapper follows the actual config layout, which exporters sometimes flatten.
    source["architectures"] = [architecture]
    declared = declare(architecture, source)
    g = Geometry(
        declared=declared, layers=layers, hidden=text["hidden_size"], intermediate=text[width],
        vocab=text["vocab_size"], query_heads=text["num_attention_heads"],
        kv_heads=text["num_key_value_heads"], head_dim=text["head_dim"],
        gdn_key_heads=text["linear_num_key_heads"], gdn_key_head_dim=text["linear_key_head_dim"],
        gdn_value_heads=text["linear_num_value_heads"], gdn_value_head_dim=text["linear_value_head_dim"],
        gdn_conv_kernel=text["linear_conv_kernel_dim"], layer_types=tuple(schedule), rotary_dim=rotary,
        rms_epsilon=text["rms_norm_eps"], rope_theta=rope["rope_theta"],
        max_context=text["max_position_embeddings"], mtp_layers=mtp,
        token_domain=text["vocab_size"] if token_domain is None else token_domain,
        experts=text.get("num_experts", 0), experts_per_token=text.get("num_experts_per_tok", 0),
        shared_intermediate=text.get("shared_expert_intermediate_size", 0),
    )
    for size in (g.convolution_dim + g.value_dim, g.experts * 2 * g.intermediate,
                 g.experts * g.hidden, 2 * g.shared_intermediate):
        if size > 2147483647:
            raise ValueError("checkpoint projection rows exceed int32")
    geometry_block(g)
    return g


def geometry_from_checkpoint(root, config: Mapping | None = None, *, mixture: bool = False,
                             extra_names=()) -> Geometry:
    from dataclasses import replace
    from pathlib import Path
    import json
    from .checkpoint import tokenizer_domain
    from .conversion import checkpoint_tensor_names
    from .quant_scope import observed_scope

    source = deepcopy(dict(config)) if config is not None else json.loads((Path(root) / "config.json").read_text())
    names = set(checkpoint_tensor_names(root)) | set(extra_names)
    text = source.get("text_config", source)
    if not any(name.startswith("mtp.") for name in names):
        text["mtp_num_hidden_layers"] = 0
    if not any("visual." in name for name in names):
        source.pop("vision_config", None)
    g = geometry_from_config(source, mixture=mixture, token_domain=tokenizer_domain(root))
    return replace(g, observed_scope=observed_scope(names))


def geometry_block(g: Geometry, *, token_domain: int | None = None,
                   mtp: bool | None = None) -> dict:
    values = {
        "hidden": g.hidden, "residual": g.hidden, "layers": g.layers,
        "intermediate": g.intermediate, "output_rows": g.vocab,
        "token_domain": g.token_domain if token_domain is None else token_domain,
        "query_heads": g.query_heads, "kv_heads": g.kv_heads, "head_dim": g.head_dim,
        "rotary_dim": g.rotary_dim, "rms_epsilon": g.rms_epsilon, "rope_theta": g.rope_theta,
        "attention_scale": 1 / math.sqrt(g.head_dim), "max_context": g.max_context,
        "gdn_key_heads": g.gdn_key_heads, "gdn_key_head_dim": g.gdn_key_head_dim,
        "gdn_value_heads": g.gdn_value_heads, "gdn_value_head_dim": g.gdn_value_head_dim,
        "gdn_conv_kernel": g.gdn_conv_kernel, "gdn_scale": 1 / math.sqrt(g.gdn_key_head_dim),
        "mtp_layers": g.mtp_layers if mtp is None else g.mtp_layers * int(mtp),
        "draft_vocab": g.draft_vocab,
    }
    if g.experts:
        values.update(experts=g.experts, experts_per_token=g.experts_per_token,
                      shared_intermediate=g.shared_intermediate, routed_scale=1.0)
    return validate_resolved_geometry(values)


def vision_geometry_block(config: Mapping, *, text_hidden: int | None = None) -> dict | None:
    vision = config.get("vision_config")
    if vision is None:
        return None
    if not isinstance(vision, Mapping):
        raise ValueError("vision_config must be an object")
    for name in ("depth", "hidden_size", "intermediate_size", "num_heads", "in_channels",
                 "temporal_patch_size", "patch_size", "spatial_merge_size",
                 "num_position_embeddings", "out_hidden_size"):
        positive_int(vision, name)
    hidden, heads = vision["hidden_size"], vision["num_heads"]
    if hidden % heads or (hidden // heads) % 4:
        raise ValueError("vision attention requires a head width divisible by four")
    if text_hidden is not None and vision["out_hidden_size"] != text_hidden:
        raise ValueError("vision out_hidden_size must equal the text hidden_size")
    return validate_geometry({
        "layers": vision["depth"], "hidden": hidden, "intermediate": vision["intermediate_size"],
        "heads": heads, "patch_dim": vision["in_channels"] * vision["temporal_patch_size"] * vision["patch_size"] ** 2,
        "merge": vision["spatial_merge_size"], "position_embeddings": vision["num_position_embeddings"],
        "rotary_dim": hidden // heads, "output_hidden": vision["out_hidden_size"],
        # These are algorithm constants of this vision architecture, not size presets.
        "rope_theta": 10000.0, "norm_epsilon": 1e-6,
    }, vision=True)


def vision_tower(g: Geometry) -> dict:
    vg = vision_geometry_block(g.declared.hf_config, text_hidden=g.hidden)
    if vg is None:
        return {}
    return dict(layers=vg["layers"], hidden=vg["hidden"], intermediate=vg["intermediate"],
                qkv_rows=3 * vg["hidden"], patch_rows=vg["patch_dim"],
                position_embeddings=vg["position_embeddings"],
                merger_hidden=vg["hidden"] * vg["merge"] ** 2)


def config_from_gguf(reader, arch: str) -> dict:
    """Normalize a GGUF checkpoint using its metadata and stored tensor identities."""
    def kv(name):
        value = reader.kv(f"{arch}.{name}")
        if value is None:
            raise ValueError(f"GGUF is missing {arch}.{name}")
        return value

    embedding = reader.tensor("token_embd.weight")
    if embedding is None or len(embedding.shape) != 2:
        raise ValueError("GGUF is missing a matrix token_embd.weight")
    nextn = reader.kv(f"{arch}.nextn_predict_layers", 0)
    if isinstance(nextn, bool) or not isinstance(nextn, int) or not 0 <= nextn <= 1:
        raise ValueError("GGUF nextn_predict_layers must be 0 or 1")
    blocks = positive_int({"block_count": kv("block_count")}, "block_count")
    if blocks <= nextn:
        raise ValueError("GGUF must contain text blocks before nextn layers")
    head = kv("attention.key_length")
    positive_int({"attention.key_length": head}, "attention.key_length")
    rotary = kv("rope.dimension_count")
    positive_int({"rope.dimension_count": rotary}, "rope.dimension_count")
    state = kv("ssm.state_size")
    inner = kv("ssm.inner_size")
    value_heads = kv("ssm.time_step_rank")
    for name, value in (("ssm.state_size", state), ("ssm.inner_size", inner), ("ssm.time_step_rank", value_heads)):
        positive_int({name: value}, name)
    if inner % value_heads:
        raise ValueError("GGUF linear inner size must be divisible by its value-head count")
    sections = list(kv("rope.dimension_sections"))
    if len(sections) == 4 and sections[-1] == 0:
        sections.pop()
    mixture = arch in ("qwen35moe", "qwen3_5_moe", "qwen3_6_moe", "qwen4exp")
    text = {
        "hidden_size": kv("embedding_length"), "num_hidden_layers": blocks - nextn,
        "vocab_size": int(embedding.shape[1]), "num_attention_heads": kv("attention.head_count"),
        "num_key_value_heads": kv("attention.head_count_kv"), "head_dim": head,
        "linear_num_key_heads": kv("ssm.group_count"), "linear_key_head_dim": state,
        "linear_num_value_heads": value_heads, "linear_value_head_dim": inner // value_heads,
        "linear_conv_kernel_dim": kv("ssm.conv_kernel"),
        "full_attention_interval": kv("full_attention_interval"),
        "max_position_embeddings": kv("context_length"),
        "rms_norm_eps": kv("attention.layer_norm_rms_epsilon"),
        "tie_word_embeddings": reader.tensor("output.weight") is None,
        "mtp_num_hidden_layers": nextn, "hidden_act": "silu",
        "rope_parameters": {"rope_type": "default", "rope_theta": kv("rope.freq_base"),
                            "partial_rotary_factor": rotary / head,
                            "mrope_section": sections, "mrope_interleaved": True},
    }
    if mixture:
        text.update(moe_intermediate_size=kv("expert_feed_forward_length"),
                    shared_expert_intermediate_size=kv("expert_shared_feed_forward_length"),
                    num_experts=kv("expert_count"), num_experts_per_tok=kv("expert_used_count"),
                    norm_topk_prob=True)
    else:
        text["intermediate_size"] = kv("feed_forward_length")
    # GGUF has no vision tensors in this text checkpoint. A separate tower needs its own config.
    return {**text, "architectures": ["Qwen3_5MoeForCausalLM" if mixture else "Qwen3_5ForCausalLM"],
            "model_type": "qwen3_5_moe" if mixture else "qwen3_5"}
