import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import argparse
import json
import math
import os
from collections import Counter
from transformers import AutoModelForCausalLM, AutoConfig

"""
Upcycle a dense Hugging Face transformer model into a Sparse Mixture of Experts (MoE) model (https://openreview.net/pdf?id=HDZ2GBwrWo).

- For Qwen3-0.6B, the 8-2 configuration (8 total experts, 2 active per token) is highly recommended
- Avoid High Top-k: Do not be tempted to increase the active experts (top-k) to 3 or 4. For Qwen3-0.6B, increasing k beyond 2 increases the active footprint but shows no consistent accuracy gain
- Qwen3-0.6B shows a unique "sweet spot" for Depth Upscaling at 20%.  If you have the computational overhead, applying a 20% DUS alongside MoE upcycling can provide small but repeatable performance bumps

Simply running the script is not enough; the model requires a lightweight fine-tuning stage to specialise the experts.

1. Dataset Size: Use a budget of approximately 150,000 supervised fine-tuning (SFT) samples
2. Training Duration: The sources suggest training for one epoch (roughly 1,056 updates)
3. Hardware & Time: This can be achieved on a single consumer-grade GPU (such as an NVIDIA RTX PRO 6000) in approximately 1.5 to 8 hours

IMPORTANT: Upcycled MoE training requires careful hyperparameter tuning:
- Use LOWER learning rate than dense models (1e-5 recommended, vs 2e-4 for dense)
- This script sets router_aux_loss_coef=0.01 (10x higher than pretrained MoE defaults)
- Monitor gradient norms - should stay below 0.4 during training
- Watch for router collapse: sudden loss increases after initial decrease indicate instability

python upcycle_script.py \
    --model_id "Qwen/Qwen3-0.6B-Instruct" \
    --num_experts 8 \
    --top_k 2
"""
class MoELayer(nn.Module):
    """
    Sparse MoE module: y = sum_{i=1}^k G(x)_i * FFN_i(x) [2].
    """
    def __init__(self, original_ffn, num_experts=4, top_k=2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k

        if not (1 <= self.top_k <= self.num_experts):
            raise ValueError(f"top_k must be in [1, num_experts]; got top_k={top_k}, num_experts={num_experts}")
        
        # Experts are initialized as copies of the pretrained FFN [2].
        self.experts = nn.ModuleList([
            copy.deepcopy(original_ffn) for _ in range(num_experts)
        ])
        
        # Router function G(x) = TopK(softmax(Wx)) [2].
        # We derive the hidden dimension from the original FFN's input layer.
        hidden_dim = self._get_input_dim(original_ffn)
        self.router = nn.Linear(hidden_dim, num_experts)

    def _get_input_dim(self, ffn):
        for module in ffn.modules():
            if isinstance(module, nn.Linear):
                return module.in_features
        raise ValueError("Hidden dimension not found in FFN.")

    def forward(self, x):
        # Eq. (3): G(x) = TopK(softmax(Wx))
        # x: [batch, seq, hidden]
        router_logits = self.router(x)
        router_probs = F.softmax(router_logits, dim=-1)

        # Select top-k experts per token.
        top_k_weights, top_k_indices = torch.topk(router_probs, self.top_k, dim=-1)  # both: [B, T, K]

        # Renormalize within the selected experts (common in TopK routing).
        top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True)

        # Eq. (2): y = sum_{j=1..K} G(x)_j * FFN_{idx_j}(x)
        bsz, seq_len, hidden = x.shape
        x_flat = x.reshape(bsz * seq_len, hidden)
        idx_flat = top_k_indices.reshape(bsz * seq_len, self.top_k)
        w_flat = top_k_weights.reshape(bsz * seq_len, self.top_k)

        out_flat = torch.zeros_like(x_flat)

        # Sparse dispatch: only run experts for tokens that actually selected them.
        for expert_idx, expert in enumerate(self.experts):
            token_pos, choice_pos = torch.where(idx_flat == expert_idx)
            if token_pos.numel() == 0:
                continue

            expert_in = x_flat.index_select(0, token_pos)

            # Most HF MLPs accept either [N, H] or [N, 1, H]. Handle both.
            expert_out = expert(expert_in)
            if expert_out.dim() == 3:
                expert_out = expert_out.squeeze(1)

            expert_w = w_flat[token_pos, choice_pos].to(expert_out.dtype).unsqueeze(-1)
            expert_out = expert_out * expert_w

            out_flat.index_add_(0, token_pos, expert_out)

        return out_flat.reshape(bsz, seq_len, hidden)


def _set_config_dtype_bf16(model: nn.Module) -> None:
    """Best-effort: ensure config reflects bfloat16 for later reloads."""
    if not hasattr(model, "config"):
        return
    # Transformers convention
    try:
        model.config.torch_dtype = "bfloat16"
    except Exception:
        pass
    # Some configs (e.g. Qwen3) also serialize a plain `dtype` field.
    if hasattr(model.config, "dtype"):
        try:
            model.config.dtype = "bfloat16"
        except Exception:
            pass


def _patch_saved_config_dtype_bf16(save_path: str) -> None:
    config_path = os.path.join(save_path, "config.json")
    if not os.path.exists(config_path):
        return
    try:
        with open(config_path, "r") as f:
            cfg = json.load(f)
        cfg["torch_dtype"] = "bfloat16"
        if "dtype" in cfg:
            cfg["dtype"] = "bfloat16"
        with open(config_path, "w") as f:
            json.dump(cfg, f, indent=2, sort_keys=True)
    except Exception:
        # Don't fail the run just because config patching failed.
        return


def _estimate_checkpoint_gib(model: nn.Module) -> float:
    """Rough on-disk estimate for weights only, in GiB."""
    params = sum(p.numel() for p in model.parameters())
    # We force bf16 below.
    bytes_per_param = 2
    return (params * bytes_per_param) / (1024**3)


def _get_layers_container(model: nn.Module):
    """Best-effort discovery of the decoder block ModuleList for common HF causal LMs."""
    candidates = [
        ("model", "layers"),  # Llama/Qwen style: model.layers
        ("transformer", "h"),  # GPT-2 style: transformer.h
        ("gpt_neox", "layers"),
        ("decoder", "layers"),
    ]
    for first, second in candidates:
        if hasattr(model, first):
            sub = getattr(model, first)
            if hasattr(sub, second):
                layers = getattr(sub, second)
                if isinstance(layers, nn.ModuleList):
                    return sub, second, layers

    # Fallback: search for a unique ModuleList with plausible name.
    found = []
    for name, module in model.named_modules():
        if isinstance(module, nn.ModuleList) and name.endswith(("layers", "h")) and len(module) > 0:
            found.append((name, module))
    if len(found) == 1:
        name, layers = found[0]
        parent_name = ".".join(name.split(".")[:-1])
        child_name = name.split(".")[-1]
        parent = model.get_submodule(parent_name) if parent_name else model
        return parent, child_name, layers

    raise ValueError(
        "Could not locate transformer layer stack (ModuleList). "
        "Tried common paths like model.layers / transformer.h."
    )


def apply_depth_upscaling(model: nn.Module, dus_pct: float) -> nn.Module:
    """Depth Upscaling (DUS): replicate layers by a fixed percentage (paper: 20/40/50%)."""
    if dus_pct is None:
        return model
    dus_pct = float(dus_pct)
    if dus_pct <= 0:
        return model

    parent, attr, layers = _get_layers_container(model)
    n = len(layers)
    new_n = int(math.ceil(n * (1.0 + dus_pct / 100.0)))
    dup = new_n - n
    if dup <= 0:
        return model

    # Evenly distribute duplicated layers across depth.
    # Example: if we need 2 duplicates in 10 layers, duplicate around layers 3 and 6.
    positions = [int(math.floor((i + 1) * n / (dup + 1))) for i in range(dup)]
    counts = Counter(positions)

    new_layers = []
    # Some configs (notably Qwen3) require layer_types length == num_hidden_layers.
    layer_types = getattr(model.config, "layer_types", None)
    new_layer_types = [] if isinstance(layer_types, list) else None
    for idx, layer in enumerate(layers):
        new_layers.append(layer)
        if new_layer_types is not None:
            new_layer_types.append(layer_types[idx])
        for _ in range(counts.get(idx, 0)):
            new_layers.append(copy.deepcopy(layer))
            if new_layer_types is not None:
                new_layer_types.append(layer_types[idx])

    setattr(parent, attr, nn.ModuleList(new_layers))

    # Update config fields when present.
    for key in ("num_hidden_layers", "n_layer", "num_layers"):
        if hasattr(model.config, key):
            setattr(model.config, key, len(new_layers))
    if new_layer_types is not None:
        model.config.layer_types = new_layer_types
        if hasattr(model.config, "max_window_layers"):
            model.config.max_window_layers = len(new_layer_types)
    return model


def convert_qwen3_dense_to_hf_moe(model: nn.Module, num_experts: int, top_k: int) -> nn.Module:
    """Convert a dense Qwen3ForCausalLM into a native HF Qwen3MoeForCausalLM checkpoint.

    This avoids the key-mismatch problem where a custom MoE layer would save router params as
    `...mlp.router.*` but HF Qwen3MoE expects `...mlp.gate.weight`.
    """
    if getattr(model.config, "model_type", None) != "qwen3":
        raise ValueError("convert_qwen3_dense_to_hf_moe expects a dense Qwen3 model (model_type='qwen3')")

    base_cfg_dict = model.config.to_dict()
    base_cfg_dict.pop("_name_or_path", None)
    base_cfg_dict.pop("architectures", None)
    base_cfg_dict.pop("model_type", None)

    # Qwen3MoE-specific config fields.
    base_cfg_dict["moe_intermediate_size"] = int(base_cfg_dict.get("intermediate_size"))
    base_cfg_dict["num_experts"] = int(num_experts)
    base_cfg_dict["num_experts_per_tok"] = int(top_k)
    base_cfg_dict["norm_topk_prob"] = True

    # Set higher router auxiliary loss for upcycled models (0.01 instead of default 0.001)
    # Upcycled models need stronger load balancing to prevent router collapse during fine-tuning
    base_cfg_dict["router_aux_loss_coef"] = 0.01

    # Qwen3 config validation: if layer_types exists, it must match depth.
    n_layers = int(base_cfg_dict.get("num_hidden_layers"))
    if "layer_types" in base_cfg_dict and isinstance(base_cfg_dict["layer_types"], list):
        if len(base_cfg_dict["layer_types"]) != n_layers:
            base_cfg_dict["layer_types"] = ["full_attention"] * n_layers
        base_cfg_dict["max_window_layers"] = n_layers

    moe_cfg = AutoConfig.for_model("qwen3_moe", **base_cfg_dict)
    moe_model = AutoModelForCausalLM.from_config(moe_cfg, trust_remote_code=True)

    # Keep everything in bf16 for size + consistency.
    moe_model = moe_model.to(dtype=torch.bfloat16)
    _set_config_dtype_bf16(moe_model)

    # Load all shared weights (embeddings, attention, norms, lm_head, etc.).
    moe_model.load_state_dict(model.state_dict(), strict=False)

    # Copy dense MLP weights into each expert.
    layers_container = moe_model.model.layers
    for layer_idx, moe_layer in enumerate(layers_container):
        dense_mlp = model.model.layers[layer_idx].mlp
        moe_block = moe_layer.mlp
        # Experts: copy projections from dense MLP.
        for expert in moe_block.experts:
            expert.gate_proj.weight.data.copy_(dense_mlp.gate_proj.weight.data)
            expert.up_proj.weight.data.copy_(dense_mlp.up_proj.weight.data)
            expert.down_proj.weight.data.copy_(dense_mlp.down_proj.weight.data)

        # Gate: initialize to a small random matrix so routing isn't identical for all tokens.
        # (Paper initializes experts from dense FFN; router is learned during the lightweight SFT stage.)
        with torch.no_grad():
            moe_block.gate.weight.zero_()
            moe_block.gate.weight.add_(0.01 * torch.randn_like(moe_block.gate.weight))

    return moe_model
    
def convert_to_moe(model, num_experts, top_k):
    """
    Identifies and replaces only the top-level FFN/MLP blocks.
    """
    # 1. Identify the specific FFN/MLP module type for the model
    # For Qwen3/Llama, we target the container, not the individual projections.
    # We use a set to keep track of replaced parents to avoid double-processing.
    processed_modules = set()

    for name, module in model.named_modules():
        # Target only the main MLP block (e.g., 'model.layers.0.mlp')
        # We check if 'mlp' is in the name and ensure we aren't targeting 
        # the sub-projs like 'mlp.gate_proj'
        if ("mlp" in name.lower() or "ffn" in name.lower()) and "." not in name.split('mlp')[-1]:
            parent_name = ".".join(name.split(".")[:-1])
            child_name = name.split(".")[-1]
            
            if parent_name and name not in processed_modules:
                parent = model.get_submodule(parent_name)
                
                print(f"Upcycling FFN Block: {name}")
                moe_module = MoELayer(module, num_experts=num_experts, top_k=top_k)
                setattr(parent, child_name, moe_module)
                
                # Mark this and all its children as processed
                processed_modules.add(name)
                for sub_name, _ in module.named_modules():
                    processed_modules.add(f"{name}.{sub_name}")
    
    return model


def _write_reloadable_hf_wrapper(save_path: str, base_config_dict: dict, *, num_experts: int, top_k: int, dus_pct: float):
    """Write minimal custom HF files into save_path so AutoModel can reload the upcycled model."""
    os.makedirs(save_path, exist_ok=True)

    config_py = r'''from transformers import PretrainedConfig


class UpcycledMoeConfig(PretrainedConfig):
    model_type = "surogate_upcycled_moe"

    def __init__(
        self,
        base_config=None,
        moe_num_experts=4,
        moe_top_k=2,
        dus_pct=0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.base_config = base_config or {}
        self.moe_num_experts = int(moe_num_experts)
        self.moe_top_k = int(moe_top_k)
        self.dus_pct = float(dus_pct)
'''

    modeling_py = r'''import copy
import math
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoConfig, AutoModelForCausalLM, PreTrainedModel

from .configuration_surogate_upcycled_moe import UpcycledMoeConfig


class MoELayer(nn.Module):
    def __init__(self, original_ffn, num_experts=4, top_k=2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        if not (1 <= self.top_k <= self.num_experts):
            raise ValueError(
                f"top_k must be in [1, num_experts]; got top_k={top_k}, num_experts={num_experts}"
            )
        self.experts = nn.ModuleList([copy.deepcopy(original_ffn) for _ in range(num_experts)])
        hidden_dim = self._get_input_dim(original_ffn)
        self.router = nn.Linear(hidden_dim, num_experts)

    def _get_input_dim(self, ffn):
        for module in ffn.modules():
            if isinstance(module, nn.Linear):
                return module.in_features
        raise ValueError("Hidden dimension not found in FFN.")

    def forward(self, x):
        router_logits = self.router(x)
        router_probs = F.softmax(router_logits, dim=-1)
        top_k_weights, top_k_indices = torch.topk(router_probs, self.top_k, dim=-1)
        top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True)

        bsz, seq_len, hidden = x.shape
        x_flat = x.reshape(bsz * seq_len, hidden)
        idx_flat = top_k_indices.reshape(bsz * seq_len, self.top_k)
        w_flat = top_k_weights.reshape(bsz * seq_len, self.top_k)
        out_flat = torch.zeros_like(x_flat)

        for expert_idx, expert in enumerate(self.experts):
            token_pos, choice_pos = torch.where(idx_flat == expert_idx)
            if token_pos.numel() == 0:
                continue
            expert_in = x_flat.index_select(0, token_pos)
            expert_out = expert(expert_in)
            if expert_out.dim() == 3:
                expert_out = expert_out.squeeze(1)
            expert_w = w_flat[token_pos, choice_pos].to(expert_out.dtype).unsqueeze(-1)
            out_flat.index_add_(0, token_pos, expert_out * expert_w)
        return out_flat.reshape(bsz, seq_len, hidden)


def _get_layers_container(model: nn.Module):
    candidates = [
        ("model", "layers"),
        ("transformer", "h"),
        ("gpt_neox", "layers"),
        ("decoder", "layers"),
    ]
    for first, second in candidates:
        if hasattr(model, first):
            sub = getattr(model, first)
            if hasattr(sub, second):
                layers = getattr(sub, second)
                if isinstance(layers, nn.ModuleList):
                    return sub, second, layers
    found = []
    for name, module in model.named_modules():
        if isinstance(module, nn.ModuleList) and name.endswith(("layers", "h")) and len(module) > 0:
            found.append((name, module))
    if len(found) == 1:
        name, layers = found[0]
        parent_name = ".".join(name.split(".")[:-1])
        child_name = name.split(".")[-1]
        parent = model.get_submodule(parent_name) if parent_name else model
        return parent, child_name, layers
    raise ValueError("Could not locate transformer layer stack (ModuleList).")


def apply_depth_upscaling(model: nn.Module, dus_pct: float) -> nn.Module:
    dus_pct = float(dus_pct or 0.0)
    if dus_pct <= 0:
        return model
    parent, attr, layers = _get_layers_container(model)
    n = len(layers)
    new_n = int(math.ceil(n * (1.0 + dus_pct / 100.0)))
    dup = new_n - n
    if dup <= 0:
        return model
    positions = [int(math.floor((i + 1) * n / (dup + 1))) for i in range(dup)]
    counts = Counter(positions)
    new_layers = []
    for idx, layer in enumerate(layers):
        new_layers.append(layer)
        for _ in range(counts.get(idx, 0)):
            new_layers.append(copy.deepcopy(layer))
    setattr(parent, attr, nn.ModuleList(new_layers))
    for key in ("num_hidden_layers", "n_layer", "num_layers"):
        if hasattr(model.config, key):
            setattr(model.config, key, len(new_layers))
    return model


def convert_to_moe(model, num_experts, top_k):
    processed_modules = set()
    for name, module in model.named_modules():
        if ("mlp" in name.lower() or "ffn" in name.lower()) and "." not in name.split('mlp')[-1]:
            parent_name = ".".join(name.split(".")[:-1])
            child_name = name.split(".")[-1]
            if parent_name and name not in processed_modules:
                parent = model.get_submodule(parent_name)
                moe_module = MoELayer(module, num_experts=num_experts, top_k=top_k)
                setattr(parent, child_name, moe_module)
                processed_modules.add(name)
                for sub_name, _ in module.named_modules():
                    processed_modules.add(f"{name}.{sub_name}")
    return model


class UpcycledMoeForCausalLM(PreTrainedModel):
    config_class = UpcycledMoeConfig

    def __init__(self, config: UpcycledMoeConfig):
        super().__init__(config)
        base_cfg_dict = dict(config.base_config or {})
        model_type = base_cfg_dict.pop("model_type", None)
        if model_type is None:
            raise ValueError("Missing 'model_type' in base_config; cannot reconstruct base model")
        base_cfg = AutoConfig.for_model(model_type, **base_cfg_dict)
        inner = AutoModelForCausalLM.from_config(base_cfg, trust_remote_code=True)
        inner = apply_depth_upscaling(inner, config.dus_pct)
        inner = convert_to_moe(inner, config.moe_num_experts, config.moe_top_k)

        # Transplant the inner model's module hierarchy onto this wrapper so Transformers
        # can load weights by key paths (e.g., 'lm_head.weight') during from_pretrained.
        self._inner_cls = inner.__class__
        self._modules = inner._modules
        self._parameters = inner._parameters
        self._buffers = inner._buffers
        self._non_persistent_buffers_set = inner._non_persistent_buffers_set
        for k, v in inner.__dict__.items():
            if k in {"_modules", "_parameters", "_buffers", "_non_persistent_buffers_set", "config"}:
                continue
            self.__dict__[k] = v
        # Keep the wrapper config (which also contains all base config fields).
        self.config = config

    def forward(self, *args, **kwargs):
        return self._inner_cls.forward(self, *args, **kwargs)

    def prepare_inputs_for_generation(self, *args, **kwargs):
        fn = getattr(self._inner_cls, "prepare_inputs_for_generation", None)
        if fn is None:
            raise AttributeError("Base model does not implement prepare_inputs_for_generation")
        return fn(self, *args, **kwargs)

    def _reorder_cache(self, *args, **kwargs):
        fn = getattr(self._inner_cls, "_reorder_cache", None)
        if fn is None:
            raise AttributeError("Base model does not implement _reorder_cache")
        return fn(self, *args, **kwargs)
'''

    with open(os.path.join(save_path, "configuration_surogate_upcycled_moe.py"), "w") as f:
        f.write(config_py)
    with open(os.path.join(save_path, "modeling_surogate_upcycled_moe.py"), "w") as f:
        f.write(modeling_py)

    # Update config.json to point Auto* to these local files.
    config_path = os.path.join(save_path, "config.json")
    with open(config_path, "r") as f:
        cfg = json.load(f)
    cfg["auto_map"] = {
        "AutoConfig": "configuration_surogate_upcycled_moe.UpcycledMoeConfig",
        "AutoModelForCausalLM": "modeling_surogate_upcycled_moe.UpcycledMoeForCausalLM",
    }
    cfg["architectures"] = ["UpcycledMoeForCausalLM"]
    cfg["model_type"] = "surogate_upcycled_moe"
    cfg["base_config"] = base_config_dict
    # Avoid collisions with generation settings (Transformers treats top_k as a sampling flag).
    cfg.pop("top_k", None)
    cfg.pop("num_experts", None)
    cfg["moe_num_experts"] = int(num_experts)
    cfg["moe_top_k"] = int(top_k)
    cfg["dus_pct"] = float(dus_pct or 0.0)
    with open(config_path, "w") as f:
        json.dump(cfg, f, indent=2, sort_keys=True)


def main():
    # Outside sources: argparse and AutoModel usage are standard HF/Python utilities.
    parser = argparse.ArgumentParser(description="HF MoE Upcycling Script")
    parser.add_argument("--model_id", type=str, required=True, 
                        help="Hugging Face model ID (e.g., 'Qwen/Qwen3-0.6B')")
    parser.add_argument("--num_experts", type=int, default=8, 
                        help="Total number of experts [8]")
    parser.add_argument("--top_k", type=int, default=2, 
                        help="Active experts per token [2]")
    parser.add_argument("--dus_pct", type=float, default=20,
                        help="Depth upscaling percentage (e.g. 20, 40, 50). Layer replication.")
    parser.add_argument("--save_path", type=str, required=True, help="Directory to save the MoE model")
    parser.add_argument("--max_shard_size", type=str, default="5GB",
                        help="Shard size for save_pretrained (does not reduce total size). Example: '2GB', '10GB'.")
    parser.add_argument("--make_reloadable", action="store_true",
                        help="Write custom HF wrapper files + config auto_map for trust_remote_code reloads")
    parser.add_argument("--legacy_custom_moe", action="store_true",
                        help="Force the legacy custom MoE layer replacement even for Qwen3 (produces a non-native checkpoint unless --make_reloadable is set)")
    parser.add_argument("--force", action="store_true",
                        help="Allow running on models that already appear to be MoE (not recommended)")
    
    args = parser.parse_args()

    print(f"Loading dense model: {args.model_id}...")
    # Force bf16 early to keep checkpoint size manageable.
    # Loading a sub-billion parameter model as recommended [1, 5].
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    model = model.to(dtype=torch.bfloat16)
    _set_config_dtype_bf16(model)

    looks_like_moe = bool(
        getattr(model.config, "num_experts", None)
        and getattr(model.config, "num_experts_per_tok", None)
    )
    if looks_like_moe and not args.force:
        raise ValueError(
            "Input model already appears to be MoE (has num_experts/num_experts_per_tok). "
            "This script is for upcycling a dense model into MoE. "
            "Use a dense base model, or pass --force to override."
        )

    # Qwen3MoE uses `moe_intermediate_size` to size expert FFNs (defaults to 768). If it's
    # missing/zero but the dense intermediate_size is set, align them to avoid load-time mismatches.
    if getattr(model.config, "model_type", None) == "qwen3_moe":
        if getattr(model.config, "moe_intermediate_size", None) in (None, 0):
            if getattr(model.config, "intermediate_size", None):
                model.config.moe_intermediate_size = model.config.intermediate_size
    base_params = sum(p.numel() for p in model.parameters())

    if args.dus_pct and args.dus_pct > 0:
        print(f"Applying Depth Upscaling (DUS): {args.dus_pct:.1f}%...")
        model = apply_depth_upscaling(model, args.dus_pct)
    
    print(f"Upcycling to MoE {args.num_experts}-{args.top_k}...")
    if getattr(model.config, "model_type", None) == "qwen3" and not args.legacy_custom_moe and not args.make_reloadable:
        print("Detected dense Qwen3; exporting a native HF Qwen3MoE checkpoint...")
        model = convert_qwen3_dense_to_hf_moe(model, args.num_experts, args.top_k)
    else:
        model = convert_to_moe(model, args.num_experts, args.top_k)

    # Ensure final dtype is bf16.
    model = model.to(dtype=torch.bfloat16)
    _set_config_dtype_bf16(model)
    
    # Calculate approximate active vs total parameters [6].
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Transformation complete.")
    
    print(f"Total Parameter Count: {total_params / 1e9:.2f}B")
    print(f"Number of Activated Experts (top-k): {args.top_k}")
    
    # Approximate activation params as (FFN params scale by top-k/num_experts) + all non-expert params.
    # This is only a rough estimate; exact active params depend on architecture details.
    router_params = sum(p.numel() for n, p in model.named_parameters() if ".router." in n)
    activated_params = (base_params * (args.top_k / args.num_experts)) + router_params
    print(f"Number of Activated Params: {activated_params / 1e9:.2f}B")
    
    est_gib = _estimate_checkpoint_gib(model)
    print(f"Estimated checkpoint size (weights only, bf16): ~{est_gib:.1f} GiB")
    print(f"Saving upcycled model to {args.save_path} (max_shard_size={args.max_shard_size})...")
    model.save_pretrained(args.save_path, max_shard_size=args.max_shard_size)
    _patch_saved_config_dtype_bf16(args.save_path)

    if args.make_reloadable:
        print("Writing reloadable Hugging Face wrapper (trust_remote_code=True)...")
        base_config_dict = model.config.to_dict()
        _write_reloadable_hf_wrapper(
            args.save_path,
            base_config_dict,
            num_experts=args.num_experts,
            top_k=args.top_k,
            dus_pct=args.dus_pct,
        )

    print("\n" + "="*80)
    print("✓ Upcycling complete!")
    print("="*80)
    print("\nNext step: Lightweight fine-tuning with ~150k samples.")
    print("\nCRITICAL: Upcycled MoE models require special training hyperparameters:")
    print("  • Learning rate: 1e-5 (MUCH LOWER than dense models)")
    print("  • Router aux loss: 0.01 (already set in config.json)")
    print("  • Gradient clipping: max_grad_norm=0.5")
    print("  • Warmup ratio: 0.15-0.2 (longer than dense models)")
    print("\nMonitor training for router collapse:")
    print("  ✓ Healthy: gradient norm < 0.4, steady loss decrease")
    print("  ✗ Failing: gradient norm > 0.8, loss suddenly increases")
    print("\nSee docs/getting-started/quickstart-sft.md for detailed guidance.")
    print("="*80)
    
    
if __name__ == "__main__":
    main()