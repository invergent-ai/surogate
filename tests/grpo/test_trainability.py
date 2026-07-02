from surogate.core.config.grpo_orch_config import GRPOAdvantageConfig
from surogate.grpo.orchestrator.advantage import compute_advantages
from surogate.grpo.orchestrator.trainability import exclude_non_trainable_rollouts

METRIC = "ultra_valid_for_training"


def _rollout(reward: float, trainable: bool, n_tokens: int = 3) -> dict:
    return {
        "reward": reward,
        "metrics": {METRIC: 1.0 if trainable else 0.0},
        "trajectory": [{"tokens": {"completion_mask": [1] * n_tokens}}],
    }


def _mask(rollout: dict) -> list[int]:
    return rollout["trajectory"][0]["tokens"]["completion_mask"]


def test_all_trainable_is_noop():
    rollouts = [_rollout(1.0, True), _rollout(0.0, True)]
    rewards = [r["reward"] for r in rollouts]
    adv, metrics = exclude_non_trainable_rollouts(rollouts, rewards, 2, METRIC)
    assert adv == [1.0, 0.0]
    assert metrics["trainability/non_trainable_count"] == 0.0
    assert _mask(rollouts[0]) == [1, 1, 1]
    assert _mask(rollouts[1]) == [1, 1, 1]


def test_infra_failure_masked_and_substituted():
    # group of 2: one valid-correct (trainable), one infra failure (not trainable)
    rollouts = [_rollout(1.0, True), _rollout(0.0, False)]
    rewards = [r["reward"] for r in rollouts]
    adv, metrics = exclude_non_trainable_rollouts(rollouts, rewards, 2, METRIC)
    # infra reward substituted with the trainable mean (1.0) for the baseline
    assert adv == [1.0, 1.0]
    # infra rollout masked out of the loss; trainable rollout untouched
    assert _mask(rollouts[1]) == [0, 0, 0]
    assert _mask(rollouts[0]) == [1, 1, 1]
    # true rewards preserved for logging
    assert rewards == [1.0, 0.0]
    assert rollouts[1]["reward"] == 0.0
    assert metrics["trainability/non_trainable_count"] == 1.0
    # only one trainable rollout -> no contrast -> degenerate group
    assert metrics["trainability/degenerate_group_count"] == 1.0


def test_advantage_is_unbiased_vs_naive_baseline():
    # group of 4: rewards [1.0, 0.5, 0.0(infra), 0.0(infra)]
    rollouts = [_rollout(1.0, True), _rollout(0.5, True), _rollout(0.0, False), _rollout(0.0, False)]
    rewards = [r["reward"] for r in rollouts]
    adv_rewards, _ = exclude_non_trainable_rollouts(rollouts, rewards, 4, METRIC)
    # trainable mean = 0.75; infra entries substituted to 0.75
    assert adv_rewards == [1.0, 0.5, 0.75, 0.75]
    advs = compute_advantages(adv_rewards, [1, 1, 1, 1], 4, GRPOAdvantageConfig({}))
    # baseline = mean(adv_rewards) = 0.75 = trainable mean (unbiased)
    assert abs(advs[0] - 0.25) < 1e-6
    assert abs(advs[1] + 0.25) < 1e-6
    assert abs(advs[2]) < 1e-6
    assert abs(advs[3]) < 1e-6
    # Contrast with the naive (buggy) baseline that keeps infra-0s in the mean:
    naive = compute_advantages(rewards, [1, 1, 1, 1], 4, GRPOAdvantageConfig({}))
    assert naive[0] > advs[0]  # infra-0s inflate the trainable rollout's advantage


def test_missing_metric_defaults_trainable():
    rollouts = [
        {"reward": 1.0, "metrics": {}, "trajectory": []},
        {"reward": 0.0, "metrics": {}, "trajectory": []},
    ]
    adv, metrics = exclude_non_trainable_rollouts(rollouts, [1.0, 0.0], 2, METRIC)
    assert adv == [1.0, 0.0]
    assert metrics["trainability/non_trainable_count"] == 0.0
