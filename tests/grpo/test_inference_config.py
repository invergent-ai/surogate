"""GRPOInferenceConfig -> vLLM namespace mapping for the newly plumbed scalars."""

from surogate.core.config.grpo_inference_config import GRPOInferenceConfig
from surogate.utils.dict import DictDefault


def test_max_num_seqs_and_kv_cache_dtype_reach_vllm():
    cfg = GRPOInferenceConfig(DictDefault({"model": "m", "max_num_seqs": 16, "kv_cache_dtype": "fp8"}))
    assert cfg.max_num_seqs == 16
    assert cfg.kv_cache_dtype == "fp8"

    ns = cfg.to_vllm()
    assert ns.max_num_seqs == 16
    assert ns.kv_cache_dtype == "fp8"


def test_unset_scalars_are_dropped_so_vllm_defaults_apply():
    """None is not a valid value for either flag — the attribute must be absent."""
    cfg = GRPOInferenceConfig(DictDefault({"model": "m"}))
    ns = cfg.to_vllm()
    assert not hasattr(ns, "max_num_seqs")
    assert not hasattr(ns, "kv_cache_dtype")
