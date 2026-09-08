"""Spark-X2.5, with all dimensions and attention schedules read from config.json."""

from __future__ import annotations

import math
from collections.abc import Mapping

from .. import nn
from ..block_schema import ServeObject
from ..blocks.common import STANDARD_MODEL_NAME_REMAP
from ..blocks.spark2_5 import Spark2_5FullBlock, Spark2_5SlidingBlock
from ..modules import Embedding, LMHead, RMSNorm
from ..specs import ActivationScope


def positive_int(config, name):
    value = config.get(name)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"config.{name} must be a positive integer")
    return value


def resolve_spark_config(config: Mapping[str, object]) -> dict:
    expected = {
        "architectures": ["Spark2_5ForCausalLM"],
        "model_type": "spark2_5",
        "hidden_act": "gelu",
        "headwise_attn_output_gate": True,
        "gate_attn_act_mode": "sigmoid",
        "attention_bias": False,
        "mlp_bias": False,
    }
    for name, value in expected.items():
        if config.get(name) != value:
            raise ValueError(f"config.{name}: expected {value!r}, got {config.get(name)!r}")
    if config.get("rope_scaling") is not None:
        raise ValueError("Spark rope_scaling is unsupported")
    if not isinstance(config.get("tie_word_embeddings"), bool):
        raise ValueError("config.tie_word_embeddings must be a boolean")
    dims = {
        dst: positive_int(config, src)
        for dst, src in (
            ("hidden", "hidden_size"),
            ("layers", "num_hidden_layers"),
            ("intermediate", "intermediate_size"),
            ("vocab", "vocab_size"),
            ("query_heads", "num_attention_heads"),
            ("kv_heads", "num_key_value_heads"),
            ("head_dim", "head_dim"),
            ("sliding_window", "sliding_window"),
            ("max_context", "max_position_embeddings"),
        )
    }
    if dims["layers"] > 256 or dims["query_heads"] % dims["kv_heads"]:
        raise ValueError("invalid Spark layer count or query/key head ratio")
    schedule = config.get("layer_types")
    if (
        not isinstance(schedule, list)
        or len(schedule) != dims["layers"]
        or any(kind not in ("full_attention", "sliding_attention") for kind in schedule)
    ):
        raise ValueError("config.layer_types must declare full_attention or sliding_attention for every layer")

    def positive_float(value, label):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"{label} must be a positive finite number")
        return float(value)

    ropes = config.get("rope_parameters")
    if not isinstance(ropes, dict):
        raise ValueError("config.rope_parameters must declare both attention types")
    rotations = {}
    for kind, prefix in (("full_attention", ""), ("sliding_attention", "sliding_")):
        rope = ropes.get(kind)
        if (
            not isinstance(rope, dict)
            or set(rope) - {"rope_theta", "partial_rotary_factor", "rope_type"}
            or rope.get("rope_type", "default") != "default"
        ):
            raise ValueError(f"config.rope_parameters.{kind} must declare default RoPE")
        factor = positive_float(rope.get("partial_rotary_factor"), f"{kind}.partial_rotary_factor")
        width = dims["head_dim"] * factor
        if width != int(width) or not 0 < width <= dims["head_dim"] or int(width) % 2:
            raise ValueError(f"{kind}.partial_rotary_factor must yield an even rotary dimension within head_dim")
        rotations[prefix + "rotary_dim"] = int(width)
        rotations[prefix + "rope_theta"] = positive_float(rope.get("rope_theta"), f"{kind}.rope_theta")
    return dict(
        **dims,
        **rotations,
        layer_types=tuple(schedule),
        rms_epsilon=positive_float(config.get("rms_norm_eps"), "config.rms_norm_eps"),
        tied_output_head=config["tie_word_embeddings"],
    )


@nn.hf_config(
    architecture="Spark2_5ForCausalLM",
    model_type="spark2_5",
    d_model="hidden_size",
    n_layers="num_hidden_layers",
    num_query_heads="num_attention_heads",
    num_kv_heads="num_key_value_heads",
    d_ff="intermediate_size",
    vocab_size="vocab_size",
    max_seq="max_position_embeddings",
    head_size="head_dim",
    eps="rms_norm_eps",
    layer_types="layer_types",
    sliding_window="sliding_window",
    rope_parameters="rope_parameters",
    tie_word_embeddings="tie_word_embeddings",
    hidden_act="hidden_act",
    headwise_attn_output_gate="headwise_attn_output_gate",
    gate_attn_act_mode="gate_attn_act_mode",
    attention_bias="attention_bias",
    mlp_bias="mlp_bias",
    rope_scaling="rope_scaling",
)
class Spark2_5Model(nn.Model):
    _name_remap_ = STANDARD_MODEL_NAME_REMAP
    _hf_block_mappings_ = {
        "embedding": "model.embedding.weight",
        "final_norm": "model.norm.weight",
        "lm_head": "lm_head.weight",
        "ln1_weight": "model.layers.{layer}.input_layernorm.weight",
        "ln2_weight": "model.layers.{layer}.post_attention_layernorm.weight",
        "qkv_weight": "model.layers.{layer}.self_attn.q_k_v_proj.weight",
        "attn_gate_weight": "model.layers.{layer}.self_attn.g_proj.weight",
        "out_weight": "model.layers.{layer}.self_attn.out_proj.weight",
        "mlp_gate_weight": "model.layers.{layer}.mlp.gate_proj.weight",
        "mlp_up_weight": "model.layers.{layer}.mlp.up_proj.weight",
        "mlp_down_weight": "model.layers.{layer}.mlp.down_proj.weight",
    }
    _serve_objects_ = (
        ServeObject("text/token_embedding", "w8", ("Vocab", "C"), ("embedding",), scope="model"),
        ServeObject("text/final_norm", "bf16", ("C",), ("final_norm",), scope="model"),
        ServeObject("text/output_head", "w8", ("Vocab", "C"), ("lm_head",), scope="model"),
    )
    _serve_blocks_ = {"full": Spark2_5FullBlock, "sliding": Spark2_5SlidingBlock}

    @staticmethod
    def _serve_block_schedule_(config):
        return [kind.removesuffix("_attention") for kind in config["layer_types"]]

    def __init__(
        self,
        vocab_size=None,
        d_model=None,
        n_layers=None,
        num_query_heads=None,
        num_kv_heads=None,
        d_ff=None,
        max_seq=None,
        head_size=None,
        eps=None,
        layer_types=None,
        sliding_window=None,
        rope_parameters=None,
        tie_word_embeddings=None,
        hidden_act="gelu",
        headwise_attn_output_gate=True,
        gate_attn_act_mode="sigmoid",
        attention_bias=False,
        mlp_bias=False,
        rope_scaling=None,
    ):
        super().__init__()
        arguments = {key: value for key, value in locals().items() if key not in ("self", "__class__")}
        mapping = {
            "vocab_size": "vocab_size",
            "d_model": "hidden_size",
            "n_layers": "num_hidden_layers",
            "num_query_heads": "num_attention_heads",
            "num_kv_heads": "num_key_value_heads",
            "d_ff": "intermediate_size",
            "max_seq": "max_position_embeddings",
            "head_size": "head_dim",
            "eps": "rms_norm_eps",
        }
        source = {mapping.get(key, key): value for key, value in arguments.items()}
        source.update(architectures=["Spark2_5ForCausalLM"], model_type="spark2_5")
        geometry = resolve_spark_config(source)
        for key, value in arguments.items():
            setattr(self, key, value)
        self.D = head_size
        self.rotary_dim = geometry["rotary_dim"]
        self.sliding_rotary_dim = geometry["sliding_rotary_dim"]
        self.residual_fp32 = True
        self.fused_qkv_lora = True
        self.full_rope_type = self.sliding_rope_type = "default"
        self.full_rope_theta = geometry["rope_theta"]
        self.sliding_rope_theta = geometry["sliding_rope_theta"]
        self.full_partial_rotary_factor = self.rotary_dim / head_size
        self.sliding_partial_rotary_factor = self.sliding_rotary_dim / head_size
        self.block_types = self._serve_block_schedule_({"layer_types": layer_types})
        self.n_full_blocks = self.block_types.count("full")
        self.n_sliding_blocks = self.block_types.count("sliding")
        common = dict(
            d_model=d_model,
            num_query_heads=num_query_heads,
            num_kv_heads=num_kv_heads,
            head_size=head_size,
            d_ff=d_ff,
            max_seq=max_seq,
            eps=eps,
        )
        block_configs = []
        for kind, cls in self._serve_blocks_.items():
            count = self.block_types.count(kind)
            if count:
                block_configs.append(
                    (
                        kind + "_blocks",
                        cls,
                        count,
                        dict(
                            **common,
                            sliding_window=sliding_window if kind == "sliding" else 0,
                            partial_rotary_factor=getattr(self, kind + "_partial_rotary_factor"),
                        ),
                    )
                )
        self.embedding = Embedding(vocab_size, d_model)
        self.hybrid_blocks = nn.HybridBlockStack(
            block_configs=block_configs, block_types=self.block_types, n_layers=n_layers
        )
        self.final_norm = RMSNorm(d_model, eps=eps, residual_dtype="fp32")
        self.lm_head = LMHead(vocab_size, d_model)

    def forward(self, token_ids, position_ids, targets):
        G = ActivationScope.GLOBAL
        self._register_activation("token_ids", ("B", "T"), dtype="int32", scope=G)
        self._register_activation("position_ids", ("T",), dtype="int32", scope=G)
        self._register_activation("targets", ("B", "T"), dtype="int32", scope=G, aliases=["labels"])
        self._register_activation("freq_cis", ("max_seq", "D", 2), dtype="fp32", scope=G, aliases=["rope_freqs"])
        shape = ("B", "T", "d_model")
        for name in ("residual0", "residualN", "residual_final"):
            self._register_activation(name, shape, dtype="fp32", scope=G)
        self._register_activation("x0", shape, scope=G, aliases=["encoded"])
        self._register_activation("xN", shape, scope=G)
        self._register_activation("xF", shape, scope=G, aliases=["ln_final"])
        self._register_activation("xF_flat", ("B * T", "d_model"), scope=G)
        self._register_activation("ln_final_rstd", ("B", "T"), dtype="fp32", save=True, scope=G)
        self._register_activation("loss", ("B * T",), dtype="fp32", scope=G, aliases=["losses"])
        x = self.embedding(token_ids)
        residual = self._zeros(["B", "T", "d_model"], dtype="fp32")
        x, residual = self.hybrid_blocks(x, residual, position_ids)
        residual, x = self.final_norm(residual, x)
        return self.lm_head(x, targets)
