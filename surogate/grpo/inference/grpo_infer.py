from surogate.core.config.grpo_inference_config import GRPOInferenceConfig


def grpo_infer(config: GRPOInferenceConfig):
    """Serve GRPO rollouts from this repository's own engine.

    A separate process that speaks the OpenAI chat surface plus the token-in and
    tokenize routes the multi-turn client needs. It replaces this process (see
    `surogate_engine`).
    """
    from surogate.grpo.inference.surogate_engine import server

    server(config)
