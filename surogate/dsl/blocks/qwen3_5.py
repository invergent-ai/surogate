"""Qwen3.5 dense transformer blocks."""

from __future__ import annotations

from .. import nn
from ..block_schema import BlockSchema, SlotDecl
from ..modules import GatedDeltaNetMixer, GenericMLP, Qwen3_5Attention, RMSNormPlus1, _resolve_rotary_dim


QWEN3_5_ATTN_BLOCK_REMAP: dict[str, str] = {
    # --- attn_norm (RMSNormPlus1) -> ln1 / res_ffn ---
    "attn_norm_weight": "ln1_weight",
    "attn_norm_weight_eff": "ln1_weight_eff",
    "attn_norm_res": "res_ffn",
    "attn_norm_y": "ln1",
    "attn_norm_rstd": "ln1_rstd",
    # --- self_attn (Qwen3_5Attention) -> full_* canonical names ---
    "self_attn_q_proj_weight": "full_q_proj_weight",
    "self_attn_q_proj_bias": "full_q_proj_bias",
    "self_attn_k_proj_weight": "full_k_proj_weight",
    "self_attn_k_proj_bias": "full_k_proj_bias",
    "self_attn_v_proj_weight": "full_v_proj_weight",
    "self_attn_v_proj_bias": "full_v_proj_bias",
    "self_attn_out_weight": "full_out_weight",
    "self_attn_out_bias": "full_out_bias",
    "self_attn_q_norm_weight": "q_norm_weight",
    "self_attn_k_norm_weight": "k_norm_weight",
    "self_attn_rope_freqs": "rope_freqs",
    # Activations from Qwen3_5Attention
    "self_attn_x_flat": "x_flat",
    "self_attn_q_proj": "full_q_proj",
    "self_attn_k_proj": "full_k_proj",
    "self_attn_v_proj": "full_v_proj",
    "self_attn_q_proj_4d": "full_q_proj_4d",
    "self_attn_q": "full_q",
    "self_attn_gate": "full_gate",
    "self_attn_k": "full_k",
    "self_attn_v": "full_v",
    "self_attn_q_norm_weight_eff": "q_norm_weight_eff",
    "self_attn_k_norm_weight_eff": "k_norm_weight_eff",
    "self_attn_qkv": "qkv",
    "self_attn_qkv_rope": "qkv_rope",
    "self_attn_att": "att",
    "self_attn_att_4d": "att_4d",
    "self_attn_att_flat": "att_flat",
    "self_attn_lse": "lse",
    "self_attn_att_out": "att_out",
    "self_attn_att_out_flat": "att_out_flat",
    # --- mlp_norm (RMSNormPlus1) -> ln2 / res_att ---
    "mlp_norm_weight": "ln2_weight",
    "mlp_norm_weight_eff": "ln2_weight_eff",
    "mlp_norm_res": "res_att",
    "mlp_norm_y": "ln2",
    "mlp_norm_rstd": "ln2_rstd",
    # --- mlp (GenericMLP / SwiGLU) ---
    "mlp_act": "swiglu",
    "mlp_act_flat": "swiglu_flat",
    "mlp_x_flat": "mlp_x_flat",
}

QWEN3_5_LINEAR_BLOCK_REMAP: dict[str, str] = {
    # --- attn_norm (RMSNormPlus1) -> ln1 / res_ffn ---
    "attn_norm_weight": "ln1_weight",
    "attn_norm_weight_eff": "ln1_weight_eff",
    "attn_norm_res": "res_ffn",
    "attn_norm_y": "ln1",
    "attn_norm_rstd": "ln1_rstd",
    # --- mixer (GatedDeltaNetMixer) -> lin_* canonical names ---
    "mixer_in_proj_qkv_weight": "lin_in_proj_qkv_weight",
    "mixer_in_proj_z_weight": "lin_in_proj_z_weight",
    "mixer_in_proj_b_weight": "lin_in_proj_b_weight",
    "mixer_in_proj_a_weight": "lin_in_proj_a_weight",
    "mixer_conv_weight": "lin_conv_weight",
    "mixer_A_log": "lin_A_log",
    "mixer_dt_bias": "lin_dt_bias",
    "mixer_norm_weight": "lin_norm_weight",
    "mixer_out_weight": "lin_out_weight",
    # Activations from GatedDeltaNetMixer
    "mixer_x_flat": "lin_x_flat",
    "mixer_mixed_qkv_flat": "lin_mixed_qkv_flat",
    "mixer_mixed_qkv": "lin_mixed_qkv",
    "mixer_conv_w2d": "lin_conv_w2d",
    "mixer_conv_out_cf": "lin_conv_out_cf",
    "mixer_query": "lin_query",
    "mixer_key": "lin_key",
    "mixer_value": "lin_value",
    "mixer_z_flat": "lin_z_flat",
    "mixer_z": "lin_z",
    "mixer_b_flat": "lin_b_flat",
    "mixer_b": "lin_b",
    "mixer_a_flat": "lin_a_flat",
    "mixer_a": "lin_a",
    "mixer_decay": "lin_decay",
    "mixer_query_rep": "lin_query_rep",
    "mixer_key_rep": "lin_key_rep",
    "mixer_core_flat": "lin_core_flat",
    "mixer_z_norm_flat": "lin_z_norm_flat",
    "mixer_gated_flat": "lin_gated_flat",
    "mixer_gated": "lin_gated",
    "mixer_gated_bt_flat": "lin_gated_bt_flat",
    "mixer_out_flat": "lin_att_out_flat",
    "mixer_out": "lin_att_out",
    # --- mlp_norm (RMSNormPlus1) -> ln2 / res_att ---
    "mlp_norm_weight": "ln2_weight",
    "mlp_norm_weight_eff": "ln2_weight_eff",
    "mlp_norm_res": "res_att",
    "mlp_norm_y": "ln2",
    "mlp_norm_rstd": "ln2_rstd",
    # --- mlp (GenericMLP / SwiGLU) ---
    "mlp_act": "swiglu",
    "mlp_act_flat": "swiglu_flat",
    "mlp_x_flat": "mlp_x_flat",
}

# ---- Model-level name remaps (used by surogate.dsl.models.qwen3_5) ----

QWEN3_5_MODEL_NAME_REMAP: dict[str, str] = {
    # --- embedding ---
    "embedding_weight": "embedding",
    "embedding_out": "x0",
    # --- final_norm (RMSNormPlus1) ---
    "final_norm_weight": "final_norm",
    "final_norm_weight_eff": "final_norm_eff",
    "final_norm_res": "residual_final",
    "final_norm_y": "xF",
    "final_norm_rstd": "ln_final_rstd",
    # --- lm_head ---
    "lm_head_weight": "lm_head",
    "lm_head_loss": "loss",
    "lm_head_x_flat": "xF_flat",
}

QWEN3_5_VL_MODEL_NAME_REMAP: dict[str, str] = {
    "embedding_weight": "embedding",
    # No "embedding_out" -> "x0" -- mask_scatter output takes that name
    "final_norm_weight": "final_norm",
    "final_norm_weight_eff": "final_norm_eff",
    "final_norm_res": "residual_final",
    "final_norm_y": "xF",
    "final_norm_rstd": "ln_final_rstd",
    "lm_head_weight": "lm_head",
    "lm_head_loss": "loss",
    "lm_head_x_flat": "xF_flat",
}


class Qwen3_5AttentionBlock(nn.Block):
    """Qwen3.5 full-attention decoder block (token mixer + MLP).

    Uses separate Q/K/V projections (not fused QKV), QK-Norm with weight+1 bias,
    partial MRoPE, sigmoid-gated attention output, and SwiGLU MLP.
    Both norms use the RMSNormPlus1 pattern (weight + 1 before rmsnorm).
    """

    _name_remap_ = QWEN3_5_ATTN_BLOCK_REMAP
    schema = BlockSchema(
        slots=(
            SlotDecl("full_q_proj_weight", kind="param", shape=("QProjDim", "C")),
            SlotDecl("full_k_proj_weight", kind="param", shape=("KVDim", "C")),
            SlotDecl("full_v_proj_weight", kind="param", shape=("KVDim", "C")),
            SlotDecl("full_out_weight", kind="param", shape=("C", "QProjDim")),
            SlotDecl("mlp_up_weight", kind="param", shape=("2M", "C"), residency="auto"),
            SlotDecl("mlp_down_weight", kind="param", shape=("C", "M"), residency="auto"),
            SlotDecl("res_att", shape=("B", "T", "C"), save_for_backward=True),
            SlotDecl("qkv_rope", shape=("B", "T", "QKV"), save_for_backward=True),
        ),
        attrs={"block_family": "qwen3_5_attention"},
    )

    def __init__(
        self,
        d_model: int,
        num_query_heads: int,
        num_kv_heads: int,
        head_size: int,
        d_ff: int,
        max_seq: int,
        eps: float = 1e-6,
        use_qkv_bias: bool = False,
        partial_rotary_factor: float = 0.25,
        mrope_section: tuple[int, int, int] | list[int] = (11, 11, 10),
    ):
        super().__init__()
        self.d_model = d_model
        self.num_query_heads = num_query_heads
        self.num_kv_heads = num_kv_heads
        self.head_size = head_size
        self.d_ff = d_ff
        self.max_seq = max_seq
        self.eps = eps
        self.use_qkv_bias = use_qkv_bias
        self.partial_rotary_factor = partial_rotary_factor
        if mrope_section is None or len(mrope_section) < 3:
            mrope_section = (11, 11, 10)
        self.mrope_section = list(mrope_section)

        # Derived dimensions for shape resolution
        self.D = head_size
        self.Hq = num_query_heads
        self.Hkv = num_kv_heads
        self.C = d_model
        self.M = d_ff
        self.MUp = 2 * d_ff
        self.MaxSeq = max_seq
        self.AttnDim = num_query_heads * head_size
        self.QProjDim = 2 * self.AttnDim
        self.KVDim = num_kv_heads * head_size
        self.QKV = (num_query_heads + 2 * num_kv_heads) * head_size
        self.RotaryDim = _resolve_rotary_dim(head_size, partial_rotary_factor)

        self.attn_norm = RMSNormPlus1(d_model, eps=eps)
        self.self_attn = Qwen3_5Attention(
            d_model,
            num_query_heads,
            num_kv_heads,
            head_size,
            max_seq,
            use_qkv_bias=use_qkv_bias,
            eps=eps,
            partial_rotary_factor=partial_rotary_factor,
            mrope_section=mrope_section,
        )
        self.mlp_norm = RMSNormPlus1(d_model, eps=eps)
        self.mlp = GenericMLP(d_model, d_ff)

    def forward(self, x, residual, position_ids):
        # Pre-attention normalization (fused residual + rmsnorm with weight+1)
        residual, h = self.attn_norm(residual, x)
        # Attention
        h = self.self_attn(h, position_ids)
        # Pre-MLP normalization (fused residual + rmsnorm with weight+1)
        residual, h = self.mlp_norm(residual, h)
        # MLP
        h = self.mlp(h)
        return h, residual


class Qwen3_5LinearBlock(nn.Block):
    """Qwen3.5 linear-attention (Gated DeltaNet) decoder block (token mixer + MLP).

    Uses Gated DeltaNet linear attention (NOT Mamba2), with conv1d,
    chunk_gated_delta_rule, and gated rmsnorm. Both norms use the
    RMSNormPlus1 pattern (weight + 1 before rmsnorm).
    """

    _name_remap_ = QWEN3_5_LINEAR_BLOCK_REMAP
    schema = BlockSchema(
        slots=(
            SlotDecl("lin_in_proj_qkv_weight", kind="param", shape=("ConvDim", "C")),
            SlotDecl("lin_in_proj_z_weight", kind="param", shape=("ValueDim", "C")),
            SlotDecl("lin_in_proj_b_weight", kind="param", shape=("Hk", "C")),
            SlotDecl("lin_in_proj_a_weight", kind="param", shape=("Hk", "C")),
            SlotDecl("lin_conv_weight", kind="param", shape=("ConvDim", 1, "ConvK")),
            SlotDecl("lin_out_weight", kind="param", shape=("C", "ValueDim")),
            SlotDecl("mlp_up_weight", kind="param", shape=("2M", "C"), residency="auto"),
            SlotDecl("mlp_down_weight", kind="param", shape=("C", "M"), residency="auto"),
            SlotDecl("lin_conv_state", shape=("B", "ConvDim", "ConvK"), save_for_backward=True),
            SlotDecl("lin_delta_state", shape=("B", "Hv", "Vd"), save_for_backward=True),
            SlotDecl("res_att", shape=("B", "T", "C"), save_for_backward=True),
        ),
        attrs={"block_family": "qwen3_5_linear"},
    )

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        linear_conv_kernel_dim: int = 4,
        linear_key_head_dim: int = 128,
        linear_value_head_dim: int = 128,
        linear_num_key_heads: int = 16,
        linear_num_value_heads: int = 32,
        chunk_size: int = 64,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.linear_conv_kernel_dim = linear_conv_kernel_dim
        self.linear_key_head_dim = linear_key_head_dim
        self.linear_value_head_dim = linear_value_head_dim
        self.linear_num_key_heads = linear_num_key_heads
        self.linear_num_value_heads = linear_num_value_heads
        self.chunk_size = chunk_size
        self.eps = eps

        if linear_num_value_heads % linear_num_key_heads != 0:
            raise ValueError(
                "Qwen3_5LinearBlock requires linear_num_value_heads to be divisible by linear_num_key_heads"
            )

        # Derived dimensions for shape resolution
        self.C = d_model
        self.M = d_ff
        self.MUp = 2 * d_ff
        self.Hk = linear_num_key_heads
        self.Hv = linear_num_value_heads
        self.Kd = linear_key_head_dim
        self.Vd = linear_value_head_dim
        self.KeyDim = self.Hk * self.Kd
        self.ValueDim = self.Hv * self.Vd
        self.ConvK = linear_conv_kernel_dim
        self.ConvDim = self.KeyDim * 2 + self.ValueDim
        self.HeadRepeat = self.Hv // self.Hk

        self.attn_norm = RMSNormPlus1(d_model, eps=eps)
        self.mixer = GatedDeltaNetMixer(
            d_model,
            linear_conv_kernel_dim=linear_conv_kernel_dim,
            linear_key_head_dim=linear_key_head_dim,
            linear_value_head_dim=linear_value_head_dim,
            linear_num_key_heads=linear_num_key_heads,
            linear_num_value_heads=linear_num_value_heads,
            chunk_size=chunk_size,
            eps=eps,
        )
        self.mlp_norm = RMSNormPlus1(d_model, eps=eps)
        self.mlp = GenericMLP(d_model, d_ff)

    def forward(self, x, residual, position_ids):
        del position_ids  # Unused in linear-attention layers.

        # Pre-attention normalization (fused residual + rmsnorm with weight+1)
        residual, h = self.attn_norm(residual, x)
        # Linear attention (Gated DeltaNet)
        h = self.mixer(h)
        # Pre-MLP normalization (fused residual + rmsnorm with weight+1)
        residual, h = self.mlp_norm(residual, h)
        # MLP
        h = self.mlp(h)
        return h, residual
