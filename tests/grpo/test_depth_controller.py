"""Adaptive rollout-depth budgeting (TurnOPD arXiv:2607.05804 §5.1, coverage arm)."""

import asyncio

import pytest

from surogate.grpo.orchestrator.depth_controller import (
    DepthObservation,
    RolloutDepthController,
    _quantile,
)


def obs(depths, successes=None):
    if successes is None:
        successes = [True] * len(depths)
    return [DepthObservation(depth=d, success=s) for d, s in zip(depths, successes)]


class TestQuantile:
    def test_endpoints(self):
        v = [1, 2, 3, 4, 5]
        assert _quantile(v, 0.0) == 1
        assert _quantile(v, 1.0) == 5

    def test_interpolates(self):
        assert _quantile([0, 10], 0.5) == pytest.approx(5.0)

    def test_singleton(self):
        assert _quantile([7], 0.8) == 7.0

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            _quantile([], 0.5)


class TestDisabled:
    def test_never_caps(self):
        c = RolloutDepthController(enabled=False, max_depth=12)
        c.observe(0, obs([2, 2, 2] * 10))
        assert c.cap_for_step(0) is None
        assert c.cap_for_step(100) is None


class TestProbeCadence:
    def test_warmup_steps_are_uncapped(self):
        c = RolloutDepthController(enabled=True, max_depth=12, warmup_steps=3, probe_interval=8)
        assert all(c.is_probe_step(s) for s in (0, 1, 2))

    def test_probe_every_interval(self):
        c = RolloutDepthController(enabled=True, max_depth=12, warmup_steps=0, probe_interval=4)
        assert [s for s in range(12) if c.is_probe_step(s)] == [0, 4, 8]

    def test_probe_steps_run_uncapped_even_once_a_cap_exists(self):
        c = RolloutDepthController(
            enabled=True, max_depth=12, warmup_steps=0, probe_interval=4, min_observations=1
        )
        c.observe(0, obs([3, 3, 3, 3]))
        assert c.cap is not None
        assert c.cap_for_step(4) is None, "probe step must run to the env's full depth"
        assert c.cap_for_step(5) == c.cap


class TestCapEstimation:
    def test_covers_the_requested_quantile_of_successes(self):
        c = RolloutDepthController(
            enabled=True, max_depth=20, warmup_steps=0, probe_interval=1, quantile=0.8, ema=1.0, min_observations=1
        )
        # 10 successes at depths 1..10 -> p80 = 8.2 -> 8
        c.observe(0, obs(list(range(1, 11))))
        assert c.cap == 8

    def test_ignores_failed_rollouts_when_successes_exist(self):
        c = RolloutDepthController(
            enabled=True, max_depth=20, warmup_steps=0, probe_interval=1, quantile=0.8, ema=1.0, min_observations=1
        )
        # Successes are shallow; failures stall at the horizon and must not inflate the cap.
        depths = [2, 2, 3, 3] + [20] * 8
        success = [True] * 4 + [False] * 8
        c.observe(0, obs(depths, success))
        assert c.cap <= 3
        assert c.metrics()["depth/success_conditioned"] == 1.0

    def test_falls_back_to_full_population_without_successes(self):
        c = RolloutDepthController(
            enabled=True, max_depth=20, warmup_steps=0, probe_interval=1, quantile=0.8, ema=1.0, min_observations=1
        )
        c.observe(0, obs([4, 5, 6, 7], [False] * 4))
        assert c.cap is not None
        assert c.metrics()["depth/success_conditioned"] == 0.0

    def test_clipped_to_bounds(self):
        c = RolloutDepthController(
            enabled=True, max_depth=5, min_depth=3, warmup_steps=0, probe_interval=1, ema=1.0, min_observations=1
        )
        c.observe(0, obs([50] * 8))
        assert c.cap == 5
        c2 = RolloutDepthController(
            enabled=True, max_depth=20, min_depth=4, warmup_steps=0, probe_interval=1, ema=1.0, min_observations=1
        )
        c2.observe(0, obs([1] * 8))
        assert c2.cap == 4

    def test_no_cap_before_min_observations(self):
        c = RolloutDepthController(
            enabled=True, max_depth=12, warmup_steps=0, probe_interval=1, min_observations=10
        )
        c.observe(0, obs([3] * 4))
        assert c.cap is None
        c.observe(1, obs([3] * 8))
        assert c.cap is not None

    def test_ema_smooths_across_probes(self):
        c = RolloutDepthController(
            enabled=True, max_depth=20, warmup_steps=0, probe_interval=1, quantile=1.0, ema=0.5, min_observations=1
        )
        c.observe(0, obs([4] * 8))
        assert c.cap == 4
        c.observe(1, obs([12] * 8))
        # 0.5*4 + 0.5*12 = 8, not a jump straight to 12
        assert c.cap == 8


class TestBiasTrap:
    """A cap estimated from capped rollouts can only ratchet down."""

    def test_capped_steps_do_not_feed_statistics(self):
        c = RolloutDepthController(
            enabled=True, max_depth=20, warmup_steps=0, probe_interval=5, quantile=1.0, ema=1.0, min_observations=1
        )
        c.observe(0, obs([10] * 8))  # probe
        assert c.cap == 10

        # Non-probe steps observe truncated depths; they must be discarded.
        for step in (1, 2, 3, 4):
            c.observe(step, obs([10] * 8))
        assert c.cap == 10, "censored depths must not shrink the horizon"

    def test_cap_can_grow_back_from_a_probe(self):
        c = RolloutDepthController(
            enabled=True, max_depth=20, warmup_steps=0, probe_interval=2, quantile=1.0, ema=1.0, min_observations=1
        )
        c.observe(0, obs([4] * 8))
        assert c.cap == 4
        c.observe(2, obs([15] * 8))  # probe sees deeper trajectories
        assert c.cap == 15


class TestCapInjection:
    """The cap must ride in `info` so it survives to a remote env server."""

    def _scheduler_stub(self, cap):
        from surogate.grpo.orchestrator.scheduler import Scheduler

        stub = Scheduler.__new__(Scheduler)
        stub.depth_cap = cap
        return stub

    def test_no_cap_returns_example_unchanged(self):
        from surogate.grpo.orchestrator.scheduler import Scheduler

        stub = self._scheduler_stub(None)
        example = {"task": "t", "info": {"a": 1}}
        assert Scheduler._example_with_depth_cap(stub, example) is example

    def test_cap_lands_in_info_without_mutating_the_source(self):
        from surogate.grpo.orchestrator.patches import ROLLOUT_DEPTH_CAP_KEY
        from surogate.grpo.orchestrator.scheduler import Scheduler

        stub = self._scheduler_stub(6)
        example = {"task": "t", "info": {"a": 1}}
        out = Scheduler._example_with_depth_cap(stub, example)

        assert out["info"][ROLLOUT_DEPTH_CAP_KEY] == 6
        assert out["info"]["a"] == 1
        assert ROLLOUT_DEPTH_CAP_KEY not in example["info"], "buffer's example must not be mutated"

    def test_handles_missing_info(self):
        from surogate.grpo.orchestrator.patches import ROLLOUT_DEPTH_CAP_KEY
        from surogate.grpo.orchestrator.scheduler import Scheduler

        stub = self._scheduler_stub(3)
        out = Scheduler._example_with_depth_cap(stub, {"task": "t"})
        assert out["info"][ROLLOUT_DEPTH_CAP_KEY] == 3


class TestEnvPatch:
    @staticmethod
    def _env(max_turns):
        """A concrete MultiTurnEnv instance without running __init__.

        MultiTurnEnv is abstract (env_response), so subclass it minimally; skip
        __init__ because we only exercise the stop condition.
        """
        import verifiers as vf

        class _Env(vf.MultiTurnEnv):
            async def env_response(self, messages, state, **kwargs):
                return []

        env = _Env.__new__(_Env)
        env.max_turns = max_turns
        return env

    def test_patch_preserves_stop_marker_and_is_idempotent(self):
        import verifiers as vf

        from surogate.grpo.orchestrator.patches import monkey_patch_multiturn_env_depth_cap

        monkey_patch_multiturn_env_depth_cap()
        patched = vf.MultiTurnEnv.max_turns_reached
        assert hasattr(patched, "stop"), "must still be discoverable as a stop condition"
        assert patched.__name__ == "max_turns_reached"

        monkey_patch_multiturn_env_depth_cap()
        assert vf.MultiTurnEnv.max_turns_reached is patched, "re-patching must be a no-op"

    def test_cap_stops_the_rollout_at_the_requested_depth(self):
        import verifiers as vf

        from surogate.grpo.orchestrator.patches import ROLLOUT_DEPTH_CAP_KEY, monkey_patch_multiturn_env_depth_cap

        monkey_patch_multiturn_env_depth_cap()

        env = self._env(12)
        reached = vf.MultiTurnEnv.max_turns_reached

        state = {"info": {ROLLOUT_DEPTH_CAP_KEY: 3}, "trajectory": [None, None]}
        assert asyncio.run(reached(env, state)) is False
        state["trajectory"].append(None)
        assert asyncio.run(reached(env, state)) is True

    def test_without_cap_falls_through_to_env_max_turns(self):
        import verifiers as vf

        from surogate.grpo.orchestrator.patches import monkey_patch_multiturn_env_depth_cap

        monkey_patch_multiturn_env_depth_cap()

        env = self._env(3)
        reached = vf.MultiTurnEnv.max_turns_reached

        state = {"info": {}, "trajectory": [None, None]}
        assert asyncio.run(reached(env, state)) is False
        state["trajectory"].append(None)
        assert asyncio.run(reached(env, state)) is True

    def test_cap_deeper_than_env_max_turns_still_respects_env(self):
        """The cap tightens the horizon; it must never extend it."""
        import verifiers as vf

        from surogate.grpo.orchestrator.patches import ROLLOUT_DEPTH_CAP_KEY, monkey_patch_multiturn_env_depth_cap

        monkey_patch_multiturn_env_depth_cap()

        env = self._env(3)
        reached = vf.MultiTurnEnv.max_turns_reached

        state = {"info": {ROLLOUT_DEPTH_CAP_KEY: 99}, "trajectory": [None, None, None]}
        assert asyncio.run(reached(env, state)) is True
