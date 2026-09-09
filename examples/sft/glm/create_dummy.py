"""Create a deterministic, tiny GLM-5.3-Flash checkpoint without downloading weights.

Run from the repository root:
    python examples/sft/glm/create_dummy.py --output models/dummy-glm-5.3-flash

The checkpoint uses the released model's names (including separate experts and
convolutions), so it tests the same weight importer as the 320B model. Its tiny
vision tower exists only to make it loadable by Transformers; the training
examples exercise the text tower.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def dummy_config() -> dict:
    return {
        "architectures": ["Glm5NextForConditionalGeneration"],
        "model_type": "glm5_next",
        "tie_word_embeddings": False,
        "text_config": {
            "model_type": "glm5_next_text",
            "vocab_size": 512,
            "hidden_size": 128,
            "intermediate_size": 256,
            "moe_intermediate_size": 128,
            "num_hidden_layers": 8,
            "num_attention_heads": 2,
            "num_key_value_heads": 2,
            "n_routed_experts": 4,
            "n_shared_experts": 1,
            "num_experts_per_tok": 2,
            "n_group": 1,
            "topk_group": 1,
            "norm_topk_prob": True,
            "routed_scaling_factor": 2.5,
            "first_k_dense_replace": 3,
            "q_lora_rank": 64,
            "kv_lora_rank": 64,
            "qk_nope_head_dim": 64,
            "qk_rope_head_dim": 0,
            "v_head_dim": 64,
            "linear_num_heads": 2,
            "linear_head_dim": 32,
            "linear_conv_kernel_dim": 4,
            "linear_lower_bound": -5.0,
            "hc_mult": 4,
            "hc_eps": 1e-6,
            "hc_sinkhorn_iters": 20,
            "index_topk": 256,
            "index_head_dim": 32,
            "index_n_heads": 2,
            "index_kpool": 4,
            "index_kpool_always_select_tail": True,
            "max_position_embeddings": 256,
            "rms_norm_eps": 1e-5,
            "swiglu_limit": 10.0,
            "pad_token_id": 0,
            "eos_token_id": 1,
            "use_cache": False,
        },
        "vision_config": {
            "model_type": "glm5_next_vision",
            "depth": 1,
            "hidden_size": 32,
            "num_heads": 2,
            "intermediate_size": 64,
            "out_hidden_size": 128,
            "projection_intermediate_size": 64,
            "image_size": 16,
            "patch_size": 4,
            "temporal_patch_size": 1,
            "spatial_merge_size": 2,
        },
        "image_token_id": 503,
        "video_token_id": 504,
        "image_start_token_id": 505,
        "image_end_token_id": 506,
        "video_start_token_id": 507,
        "video_end_token_id": 508,
    }


def checkpoint_weights(model):
    """Undo Transformers' in-memory renaming/fusion to match the released files."""
    for name, tensor in model.state_dict().items():
        name = name.replace(".self_attn.forget_gate.", ".self_attn.")
        for site in ("attn", "ffn"):
            name = name.replace(f".{site}_hc.", f".hc_{site}_")
        if name.endswith(".self_attn.conv1d.weight"):
            for part, value in zip(("q", "k", "v"), tensor.chunk(3, dim=0)):
                yield name.replace(".conv1d.weight", f".{part}_conv1d.weight"), value.contiguous()
        elif name.endswith(".mlp.experts.gate_up_proj"):
            for i, expert in enumerate(tensor):
                for part, value in zip(("gate", "up"), expert.chunk(2, dim=0)):
                    yield name.replace(".gate_up_proj", f".{i}.{part}_proj.weight"), value.contiguous()
        elif name.endswith(".mlp.experts.down_proj"):
            for i, value in enumerate(tensor):
                yield name.replace(".down_proj", f".{i}.down_proj.weight"), value.contiguous()
        else:
            yield name, tensor.contiguous()


def dummy_tokenizer():
    """Byte tokenizer with the released chat/tool protocol, within the 512 IDs."""
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    vocab = {
        token: i for i, token in enumerate(["<pad>", "<eos>", "<unk>"] + sorted(pre_tokenizers.ByteLevel.alphabet()))
    }
    backend = Tokenizer(models.BPE(vocab=vocab, merges=[], unk_token="<unk>"))
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    backend.decoder = decoders.ByteLevel()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend, pad_token="<pad>", eos_token="<eos>", unk_token="<unk>"
    )
    tokenizer.add_special_tokens(dict(additional_special_tokens=[
        "[gMASK]", "<sop>", "<|system|>", "<|user|>", "<|assistant|>", "<|observation|>",
        "<think>", "</think>", "<tool_call>", "</tool_call>",
        "<arg_key>", "</arg_key>", "<arg_value>", "</arg_value>",
    ]))
    tokenizer.chat_template = Path(__file__).with_name("chat_template.jinja").read_text()
    return tokenizer


def create_dummy(output: Path, seed: int = 42, index_topk: int = 256, max_sequence_length: int = 256):
    import torch
    from safetensors.torch import save_file
    from transformers import Glm5NextConfig, Glm5NextForConditionalGeneration

    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Output directory must be empty: {output}")
    if max_sequence_length <= 0:
        raise ValueError("max_sequence_length must be positive")
    output.mkdir(parents=True, exist_ok=True)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        config = dummy_config()
        config["text_config"]["index_topk"] = index_topk
        config["text_config"]["max_position_embeddings"] = max_sequence_length
        model = Glm5NextForConditionalGeneration(Glm5NextConfig(**config)).to(torch.bfloat16)
    # These small parameters are retained in FP32 by the training graph.
    for name, param in model.named_parameters():
        if name.endswith((".base", ".scale", ".A_log", ".dt_bias", ".e_score_correction_bias")):
            param.data = param.data.float()
    model.config.architectures = ["Glm5NextForConditionalGeneration"]
    model.config.save_pretrained(output)
    save_file(dict(checkpoint_weights(model)), str(output / "model.safetensors"), metadata={"format": "pt"})

    tokenizer = dummy_tokenizer()
    tokenizer.save_pretrained(output)
    (output / "generation_config.json").write_text(json.dumps(dict(
        eos_token_id=[tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|observation|>"),
                      tokenizer.convert_tokens_to_ids("<|user|>")], pad_token_id=tokenizer.pad_token_id)) + "\n")
    (output / "dummy_model.json").write_text(
        json.dumps(
            {
                "seed": seed,
                "parameters": sum(p.numel() for p in model.parameters()),
                "purpose": "Randomly initialized architecture test fixture, not a useful language model.",
            },
            indent=2,
        )
        + "\n"
    )
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("models/dummy-glm-5.3-flash"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--index-topk",
        type=int,
        default=256,
        help="Selected DSA tokens; use 32 to exercise sparsity with the 128-token examples",
    )
    parser.add_argument(
        "--max-sequence-length", type=int, default=256, help="Model context limit; increase this for long-rollout tests"
    )
    args = parser.parse_args()
    model = create_dummy(args.output, args.seed, args.index_topk, args.max_sequence_length)
    print(f"Saved {sum(p.numel() for p in model.parameters()):,} parameters to {args.output}")
