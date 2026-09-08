"""Small config-driven models exercising every supported GRPO training family."""


def configurations():
    from tests.test_spark25_dsl import config as spark
    from tests.test_lfm2_moe_dsl import _mini_lfm2_moe_config
    from tests.test_lfm2_vl_dsl import _mini_lfm2_vl_config
    from tests.test_laguna_dsl import _mini_laguna_config
    from tests.serve.test_gemma4_checkpoint_config import checkpoint
    from tests.serve.test_qwen3_vl import config_for as qwen_vl
    from tests.grpo.test_qwen3_5_shared_weights import config_for as qwen35

    base = dict(hidden_size=256, intermediate_size=512, num_hidden_layers=2, num_attention_heads=4,
                num_key_value_heads=2, head_dim=64, vocab_size=512, max_position_embeddings=256,
                rms_norm_eps=1e-6, hidden_act="silu", tie_word_embeddings=True,
                rope_theta=10000., attention_bias=False, eos_token_id=1)
    result = {name: base | dict(architectures=[arch], model_type=name) for name, arch in (
        ("llama", "LlamaForCausalLM"), ("qwen3", "Qwen3ForCausalLM"), ("qwen3_moe", "Qwen3MoeForCausalLM"))}
    # MiniCPM5 publishes Llama with query width independent of hidden size.
    result["minicpm5"] = result["llama"] | dict(hidden_size=192, tie_word_embeddings=False)
    result["qwen3_moe"].update(num_experts=4, num_experts_per_tok=2, moe_intermediate_size=128)
    result["qwen3_vl"] = qwen_vl()
    result["qwen3_5"] = qwen35(nested=False)
    result["qwen3_5_moe"] = qwen35(nested=False) | dict(
        architectures=["Qwen3_5MoeForCausalLM"], model_type="qwen3_5_moe_text", num_experts=4,
        num_experts_per_tok=2, moe_intermediate_size=128, shared_expert_intermediate_size=128)
    result["spark"] = spark() | dict(hidden_size=256, head_dim=64, intermediate_size=512)
    result["gemma3"] = base | dict(architectures=["Gemma3ForCausalLM"], model_type="gemma3",
        sliding_window=32, layer_types=["sliding_attention", "full_attention"],
        query_pre_attn_scalar=64, final_logit_softcapping=30., attn_logit_softcapping=50.,
        rope_local_base_freq=10000., hidden_activation="gelu_pytorch_tanh")
    for target in ("gemma4", "gemma4_e", "gemma4_moe"):
        result[target] = checkpoint(target)
        result[target]["final_logit_softcapping"] = 30.
    result["gemma4_unified"] = {"architectures": ["Gemma4UnifiedForConditionalGeneration"],
        "model_type": "gemma4_unified", "text_config": checkpoint("gemma4")}
    lfm = _mini_lfm2_moe_config(hidden_size=256, num_attention_heads=4, head_dim=64,
                               vocab_size=512, intermediate_size=512, moe_intermediate_size=128)
    result["lfm2_moe"] = lfm
    result["lfm2"] = {k: v for k, v in lfm.items() if k not in (
        "num_experts", "num_experts_per_tok", "num_dense_layers", "moe_intermediate_size")}
    result["lfm2"].update(architectures=["Lfm2ForCausalLM"], model_type="lfm2", block_multiple_of=64)
    result["lfm2_vl"] = _mini_lfm2_vl_config(text_config=result["lfm2"])
    result["laguna"] = _mini_laguna_config(hidden_size=256, num_attention_heads=4,
        head_dim=64, intermediate_size=512, moe_intermediate_size=128, vocab_size=512)
    result["gpt_oss"] = base | dict(architectures=["GptOssForCausalLM"], model_type="gpt_oss",
        num_local_experts=4, num_experts_per_tok=2, intermediate_size=128, attention_bias=True,
        sliding_window=32, layer_types=["sliding_attention", "full_attention"])
    return result
