"""Director: a faithful replica of the Sakana Fugu learned LLM orchestrator.

Fugu (Trinity-based, latency variant): a frozen small-LM backbone is used purely
as a feature extractor; a lightweight head reads its hidden state and selects which
frontier worker model should answer each query. Trained by KL to a soft reward
distribution (single-step tasks), then refined with sep-CMA-ES on end-to-end reward.
"""

__version__ = "0.0.1"
