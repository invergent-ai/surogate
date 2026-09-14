"""A group-scored env's rollouts must be scored wherever they are generated.

A group-level rubric ranks trajectories against each other, so those rollouts are
generated with ``score_rollouts=False`` and the group is scored once complete.
Validation generated against the same env and skipped that step, so every val
reward kept its default and every val statistic read as a confident 0.0 next to a
healthy training reward.

Scoring takes a group that is still a group. Re-deriving groups from a flat batch
by ``example_id`` would be wrong: the val buffer samples with replacement, so two
draws of one example would merge into a single judged group of the wrong size.
"""

import asyncio

from surogate.grpo.orchestrator.vf_utils import (
    group_scoring_failed,
    score_group_best_effort,
    score_group_if_deferred,
)


class _Rubric:
    def __init__(self):
        self.groups_seen = []

    async def score_group(self, group):
        self.groups_seen.append(len(group))
        for rollout in group:
            rollout["reward"] = 1.0


class _Env:
    """Stands in for the env group: one rubric, looked up by task name."""

    def __init__(self):
        self.rubric = _Rubric()

    def get_env_for_name(self, name):
        return self


def _group(task, n, *, legacy=False):
    """Build rollouts the way verifiers actually routes them.

    verifiers >= 0.2 dropped the top-level ``task`` column and routes via
    ``info["env_id"]``; ``get_task`` only falls back to ``task`` for pre-upgrade
    entries. Building the fake the legacy way would exercise the fallback and
    prove nothing about a live run.
    """
    out = []
    for _ in range(n):
        rollout = {"example_id": "ex0", "reward": 0.0}
        rollout.update({"task": task} if legacy else {"info": {"env_id": task}})
        out.append(rollout)
    return out


def test_a_deferred_group_is_scored_whole():
    env, group = _Env(), _group("ruler_task", 4)
    asyncio.run(score_group_if_deferred(env, group, {"ruler_task"}))

    assert env.rubric.groups_seen == [4], "ranked as one group, not per rollout"
    assert all(r["reward"] == 1.0 for r in group)


def test_a_task_that_scores_inline_is_left_alone():
    env, group = _Env(), _group("gsm8k", 4)
    asyncio.run(score_group_if_deferred(env, group, {"ruler_task"}))

    assert env.rubric.groups_seen == []
    assert all(r["reward"] == 0.0 for r in group)


def test_nothing_happens_without_deferred_tasks_or_rollouts():
    env = _Env()
    asyncio.run(score_group_if_deferred(env, _group("ruler_task", 4), set()))
    asyncio.run(score_group_if_deferred(env, [], {"ruler_task"}))

    assert env.rubric.groups_seen == []


def test_a_group_of_one_is_still_passed_to_the_rubric():
    """The scorer does not second-guess the rubric on group size.

    ``RulerRubric.score_rollout`` forwards a one-element group to ``score_group``
    itself, so refusing here would contradict it. The group size is kept sane at
    startup instead, where the fix can raise it rather than drop the rollout.
    """
    env, group = _Env(), _group("ruler_task", 1)
    asyncio.run(score_group_if_deferred(env, group, {"ruler_task"}))

    assert env.rubric.groups_seen == [1]


def test_the_legacy_task_key_still_routes():
    """Pre-upgrade WAL and checkpoint entries carry `task` instead of `info`."""
    env, group = _Env(), _group("ruler_task", 2, legacy=True)
    asyncio.run(score_group_if_deferred(env, group, {"ruler_task"}))

    assert env.rubric.groups_seen == [2]


def test_a_rubric_that_raises_does_not_take_the_run_with_it():
    """Validation must not become the one place a judge failure is fatal.

    `generate()` gathers groups without `return_exceptions`, and the val task is
    awaited unshielded, so a rubric raise would propagate out and kill the run. The
    training path drops the offending group and carries on.
    """
    class _Boom(_Rubric):
        async def score_group(self, group):
            raise RuntimeError("judge exploded")

    env = _Env()
    env.rubric = _Boom()
    group = _group("ruler_task", 2)

    assert asyncio.run(score_group_best_effort(env, group, {"ruler_task"})) == []


def test_the_strict_helper_still_propagates():
    """The scheduler wants the failure: it drops the group and records the task."""
    class _Boom(_Rubric):
        async def score_group(self, group):
            raise RuntimeError("judge exploded")

    env = _Env()
    env.rubric = _Boom()
    try:
        asyncio.run(score_group_if_deferred(env, _group("ruler_task", 2), {"ruler_task"}))
    except RuntimeError:
        return
    raise AssertionError("score_group_if_deferred must not swallow")


def test_a_swallowed_judge_failure_is_dropped_too():
    """The shipped default does not raise; it flags and returns zeros.

    `ruler.swallow_exceptions` defaults to true, so `ruler_score` catches the judge
    failure, sets `ruler_judge_failed` on every state and returns `[0.0] * n`
    normally. Catching exceptions alone would let exactly the confident zero this
    branch removes back in through the one path the default actually takes.
    """
    class _SwallowingJudge(_Rubric):
        async def score_group(self, group):
            for rollout in group:
                rollout["reward"] = 0.0
                rollout["metrics"] = {"ruler_judge_failed": 1.0}

    env = _Env()
    env.rubric = _SwallowingJudge()

    assert asyncio.run(score_group_best_effort(env, _group("ruler_task", 2), {"ruler_task"})) == []


def test_a_genuine_zero_is_still_reported():
    """Dropping must key on the failure flag, not on the reward being zero."""
    class _ScoresZero(_Rubric):
        async def score_group(self, group):
            for rollout in group:
                rollout["reward"] = 0.0
                rollout["metrics"] = {"ruler_judge_failed": 0.0}

    env = _Env()
    env.rubric = _ScoresZero()
    group = _group("ruler_task", 2)

    assert asyncio.run(score_group_best_effort(env, group, {"ruler_task"})) == group


def test_group_scoring_failed_ignores_unrelated_metrics():
    assert not group_scoring_failed([{"metrics": {"ruler_judge_latency_ms": 42.0}}])
    assert not group_scoring_failed([{"reward": 0.0}])
    assert group_scoring_failed([{"metrics": {"ruler_judge_failed": 1.0}}])


def test_a_non_numeric_metric_does_not_kill_the_run():
    """The failure check reads metrics from every rubric in the group.

    `generate()` gathers groups without `return_exceptions`, so anything raising
    out of `score_group_best_effort` propagates through the unshielded val task and
    ends the run. Nothing enforces that a metric is a float, so the check has to be
    inside the guard, not after it.
    """
    class _WritesJunk(_Rubric):
        async def score_group(self, group):
            for rollout in group:
                rollout["reward"] = 1.0
                rollout["metrics"] = {"some_judge_failed": None}

    env = _Env()
    env.rubric = _WritesJunk()

    assert asyncio.run(score_group_best_effort(env, _group("ruler_task", 2), {"ruler_task"})) == []


def test_a_second_rubrics_judge_failure_is_also_caught():
    """A RubricGroup can carry more than one judge-backed rubric."""
    assert group_scoring_failed([{"metrics": {"otherjudge_judge_failed": 1.0}}])
