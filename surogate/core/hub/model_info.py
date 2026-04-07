"""Resolve model metadata from config files.

Works with two sources:
  - LakeFS (Local Hub): reads config.json / generation_config.json from repo
  - Hugging Face Hub: fetches the same files via huggingface_hub SDK

Returns a dict of fields suitable for merging into DeployedModel creation.
"""

import json
import math
import re
from typing import Any, Optional

from huggingface_hub import hf_hub_download, model_info
from huggingface_hub.utils import EntryNotFoundError, RepositoryNotFoundError
from lakefs_sdk import ApiClient

from surogate.core.hub import lakefs
from surogate.utils.logger import get_logger

logger = get_logger()


# ── Helpers ─────────────────────────────────────────────────────────


def _format_params(n: int | float) -> str:
    """Format a raw parameter count, rounded up to the nearest 100M."""
    rounded = math.ceil(n / 1e8) * 1e8
    if rounded >= 1e9:
        return f"{rounded / 1e9:g}B"
    if rounded >= 1e6:
        return f"{rounded / 1e6:g}M"
    return f"{rounded / 1e3:g}K"


def _estimate_params(config: dict[str, Any]) -> int | None:
    """Estimate parameter count from transformer config dimensions.

    Per layer:
      - Attention: 4 * d * d_head * (n_heads + 2 * n_kv_heads)  (Q, K, V, O projections)
      - MLP:       3 * d * intermediate_size  (gate, up, down projections for SwiGLU)
              or   2 * d * intermediate_size  (non-gated FFN fallback)
      - Norms:     2 * d
    Plus embeddings: V * d (+ V * d for lm_head if not tied)
    """
    d = config.get("hidden_size")
    n_layers = config.get("num_hidden_layers")
    vocab = config.get("vocab_size")
    if not all(isinstance(v, int) and v > 0 for v in (d, n_layers, vocab)):
        return None

    # Attention
    n_heads = config.get("num_attention_heads", 0)
    n_kv_heads = config.get("num_key_value_heads", n_heads)
    head_dim = config.get("head_dim", d // n_heads if n_heads else d)
    attn = d * head_dim * (n_heads + 2 * n_kv_heads) + n_heads * head_dim * d

    # MLP
    ffn_dim = config.get("intermediate_size", 4 * d)
    has_gate = config.get("hidden_act", "") in ("silu", "swiglu", "gelu_new")
    mlp = (3 if has_gate else 2) * d * ffn_dim

    # Layer norms (2 per layer: pre-attn + pre-mlp)
    norms = 2 * d

    per_layer = attn + mlp + norms

    # Embeddings + lm_head
    tie_weights = config.get("tie_word_embeddings", True)
    embed = vocab * d * (1 if tie_weights else 2)

    return n_layers * per_layer + embed


def _parse_params_from_name(name: str) -> str | None:
    """Try to extract param count from a repo/model name like 'Llama-3.1-8B-Instruct'."""
    m = re.search(r"(\d+(?:\.\d+)?)\s*([BMK])\b", name, re.IGNORECASE)
    if not m:
        return None
    num = float(m.group(1))
    suffix = m.group(2).upper()
    return f"{num:g}{suffix}"


# ── Common extraction ───────────────────────────────────────────────


def _extract_model_info(
    config: dict[str, Any],
    generation_config: Optional[dict[str, Any]],
    repo_name: str = "",
) -> dict[str, Any]:
    """Extract canonical fields from HF-style config dicts."""
    info: dict[str, Any] = {}

    # family — from architectures[0] or model_type
    archs = config.get("architectures")
    if archs and isinstance(archs, list) and len(archs) > 0:
        info["family"] = archs[0]
    elif config.get("model_type"):
        info["family"] = config["model_type"]

    # param_count — calculate from config, parse from name as fallback
    num_params = (
        config.get("num_parameters")
        or config.get("num_params")
        or _estimate_params(config)
    )
    if num_params and isinstance(num_params, (int, float)) and num_params > 0:
        info["param_count"] = _format_params(num_params)
    elif repo_name:
        parsed = _parse_params_from_name(repo_name)
        if parsed:
            info["param_count"] = parsed

    # quantization
    qconfig = config.get("quantization_config")
    if isinstance(qconfig, dict):
        method = qconfig.get("quant_method")
        bits = qconfig.get("bits")
        if method:
            info["quantization"] = f"{method}" + (f"-{bits}bit" if bits else "")

    # context_window
    for key in (
        "max_position_embeddings",
        "max_sequence_length",
        "n_positions",
        "seq_length",
        "sliding_window",
    ):
        val = config.get(key)
        if isinstance(val, int) and val > 0:
            info["context_window"] = val
            break

    # generation_defaults
    if generation_config:
        gd: dict[str, Any] = {}
        mapping = {
            "temperature": "temperature",
            "top_p": "top_p",
            "top_k": "top_k",
            "max_new_tokens": "max_new_tokens",
            "max_length": "max_length",
            "repetition_penalty": "repetition_penalty",
        }
        for src_key, dst_key in mapping.items():
            val = generation_config.get(src_key)
            if val is not None:
                gd[dst_key] = val
        if gd:
            info["generation_defaults"] = gd

    return info


# ── LakeFS source ──────────────────────────────────────────────────


async def _read_lakefs_json(
    client: ApiClient, repo: str, ref: str, path: str,
) -> Optional[dict[str, Any]]:
    data = await lakefs.get_object_content(client, repo, ref, path)
    if data is None:
        return None
    try:
        return json.loads(data)
    except (json.JSONDecodeError, UnicodeDecodeError):
        logger.warning(f"Failed to parse {path} from {repo}@{ref}")
        return None


async def resolve_from_lakefs(
    client: ApiClient, hub_ref: str,
) -> dict[str, Any]:
    """Resolve model info from a LakeFS hub_ref like 'repo@branch'."""
    parts = hub_ref.split("@", 1)
    if len(parts) != 2:
        return {}
    repo, ref = parts

    config = await _read_lakefs_json(client, repo, ref, "config.json")
    if config is None:
        return {}

    gen_config = await _read_lakefs_json(client, repo, ref, "generation_config.json")
    return _extract_model_info(config, gen_config, repo_name=repo)


# ── Hugging Face source ────────────────────────────────────────────


def _download_hf_json(
    repo_id: str, filename: str, token: Optional[str] = None,
) -> Optional[dict[str, Any]]:
    try:
        path = hf_hub_download(repo_id=repo_id, filename=filename, token=token)
        with open(path) as f:
            return json.load(f)
    except (EntryNotFoundError, RepositoryNotFoundError):
        return None
    except Exception:
        logger.warning(f"Failed to fetch {filename} from HF {repo_id}", exc_info=True)
        return None


def resolve_from_huggingface(
    repo_id: str, token: Optional[str] = None,
) -> dict[str, Any]:
    """Resolve model info from a Hugging Face repo.

    Handles GGUF references like ``owner/repo:file.gguf`` by stripping the
    filename and fetching config.json from the base ``owner/repo``.
    """
    # GGUF refs use colon separator: owner/repo:file.gguf
    if ":" in repo_id and repo_id.split(":")[-1].endswith(".gguf"):
        repo_id = repo_id.split(":")[0]

    config = _download_hf_json(repo_id, "config.json", token)
    if config is None:
        return {}

    # Inject safetensors param count from HF model_info if available
    if not config.get("num_parameters"):
        try:
            info = model_info(repo_id, token=token)
            st = getattr(info, "safetensors", None)
            if st is not None:
                # SafeTensorsInfo object — .total or .parameters dict
                total = getattr(st, "total", None)
                if total is None and hasattr(st, "parameters"):
                    params = st.parameters
                    if isinstance(params, dict):
                        total = sum(params.values())
                if isinstance(total, int) and total > 0:
                    config["num_parameters"] = total
        except Exception:
            pass

    gen_config = _download_hf_json(repo_id, "generation_config.json", token)
    return _extract_model_info(config, gen_config, repo_name=repo_id)
