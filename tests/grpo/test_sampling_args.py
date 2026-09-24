"""What `get_sampling_args` is allowed to put in a rollout request.

A seed is the one field here that cannot mean anything useful per rollout. The
engine resolves a request seed ahead of its own server setting
(`serve/translate.cpp:62-67`), so a seed in the sampling config pins every
rollout of a group to the same sample, and it does so even when the server was
started without `--seed`. Identical rollouts score identically, and a group with
no reward spread has an advantage of exactly zero under both functions in
`orchestrator/advantage.py`, so the step trains on nothing while every log line
looks healthy. Colocate is no safer: `shared_model.py` seeds its per-request RNG
from the same value.

This was measured, not reasoned about: the same engine, prompt and temperature
gives 8/8 distinct completions with no seed in the body and 1/8 with one.
"""

from __future__ import annotations

from surogate.core.config.grpo_orch_config import GRPOSamplingConfig
from surogate.grpo.orchestrator.utils import get_sampling_args
from surogate.utils.dict import DictDefault


def _args(cfg: dict) -> dict:
    return get_sampling_args(GRPOSamplingConfig(DictDefault(cfg)), temperature=1.0)


def test_a_configured_seed_never_reaches_a_rollout_request():
    assert "seed" not in _args({"seed": 1234, "max_tokens": 32})


def test_a_seed_hidden_under_extra_body_does_not_get_through_either():
    """`extra_body` is rebuilt from the config, so it needs its own pop.

    The clients merge `extra_body` into the top level of the request body, so a
    seed there is read by the engine exactly as a top-level one is. Nothing sets
    it today, which is why it is worth a test rather than a comment.
    """
    args = _args({"extra_body": {"seed": 1234}, "max_tokens": 32})
    assert "seed" not in args
    assert "seed" not in args["extra_body"]


def test_the_rest_of_the_sampling_config_still_reaches_the_request():
    """The seed is dropped on its own; nothing else may go missing with it."""
    args = _args({"seed": 1234, "max_tokens": 32, "top_p": 0.9})
    assert args["max_tokens"] == 32
    assert args["top_p"] == 0.9
    assert args["temperature"] == 1.0
