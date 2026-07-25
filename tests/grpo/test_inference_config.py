from surogate.core.config.grpo_inference_config import GRPOInferenceConfig
from surogate.utils.dict import DictDefault


def test_attention_backend_is_forwarded_to_vllm() -> None:
    config = GRPOInferenceConfig(
        DictDefault(
            {
                "model": "local-model",
                "attention_backend": "TORCH_SDPA",
                "enforce_eager": True,
                "enable_prefix_caching": False,
            }
        )
    )

    args = config.to_vllm()

    assert args.attention_backend == "TORCH_SDPA"
    assert args.enforce_eager is True
    assert args.enable_prefix_caching is False
