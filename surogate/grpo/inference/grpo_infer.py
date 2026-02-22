import os

from surogate.core.config.grpo_inference_config import GRPOInferenceConfig


def setup_vllm_env(config: GRPOInferenceConfig):
    """Set vLLM environment variables based on config. Must be called before importing vLLM."""

    # spawn is more robust in vLLM nightlies and Qwen3-VL (fork can deadlock with multithreaded processes)
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

    if config.enable_lora:
        os.environ["VLLM_ALLOW_RUNTIME_LORA_UPDATING"] = "True"

def grpo_infer(config: GRPOInferenceConfig):
    setup_vllm_env(config)
    
    # We import here to be able to set environment variables before importing vLLM
    from surogate.grpo.inference.vllm.server import server  # pyright: ignore
    server(config)
