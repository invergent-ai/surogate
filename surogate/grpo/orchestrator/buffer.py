import hashlib
import json
import random
from collections import defaultdict
from functools import partial
from pathlib import Path
from typing import cast

import verifiers as vf
from datasets import Dataset
from verifiers.utils.save_utils import make_serializable

from surogate.core.config.grpo_orch_config import GRPOBufferConfig
from statistics import pstdev

from surogate.grpo.utils.utils import format_num, mean, mean_normalize
from surogate.utils.logger import get_logger

logger = get_logger()


class Buffer:
    """A buffer for storing rollouts and metadata."""

    POOLS = ["easy", "normal", "hard"]

    def __init__(
        self,
        dataset: Dataset,
        env_names: list[str],
        buffer_config: GRPOBufferConfig,
    ):
        self.dataset = dataset
        self.env_names = env_names
        self.config = buffer_config

        if self.config.seed is not None:
            random.seed(self.config.seed)

        # Basic assertions
        assert "example_id" in self.dataset.column_names, "The dataset must contain a `example_id` column."
        assert "prompt" in self.dataset.column_names, "The dataset must contain a `prompt` column."
        assert "task" in self.dataset.column_names, "The dataset must contain a `task` column."
        assert len(self.dataset) > 0, "The dataset must contain at least one example."
        assert isinstance(self.dataset["example_id"][0], int), "The `example_id` column must be of type int."
        assert len(set(self.dataset["example_id"])) == len(self.dataset), "The `example_id` column must be unique."
        assert set(self.dataset["task"]) == set(self.env_names), "The `task` column must contain all environment names."

        # Initialize example buffer (env_name -> (example_id -> example))
        self.example_buffer: dict[str, dict[int, dict]] = defaultdict(dict)
        for example in map(partial(cast, dict), self.dataset):
            self.example_buffer[example["task"]][example["example_id"]] = example
        assert len(self.example_buffer) == len(self.env_names)
        logger.debug(
            f"Initialized buffer with {format_num(len(self.dataset), precision=0)} example(s) in {len(self.env_names)} environment(s)"
        )

        if self.config.env_ratios is not None:
            # Convert ratios to probabilities
            env_ratio = mean_normalize(self.config.env_ratios)
            self.env_probs = {env_name: ratio for env_name, ratio in zip(self.env_names, env_ratio)}
            logger.debug(
                f"Sampling buffer according to provided environment ratios ({', '.join(f'{k}={v:.2f}' for k, v in self.env_probs.items())})"
            )
        else:
            # Count examples per environment to sample according to natural env distribution
            env_counts = [len(self.example_buffer[env_name]) for env_name in self.env_names]
            env_ratio = mean_normalize(env_counts)
            self.env_probs = {env_name: ratio for env_name, ratio in zip(self.env_names, env_ratio)}
            logger.debug(
                f"Sampling buffer according to natural environment distribution ({', '.join(f'{k}={v:.2f}' for k, v in self.env_probs.items())})"
            )

        # Initialize buffers for easy/ hard examples
        self.easy_examples: list[dict] = []
        self.hard_examples: list[dict] = []
        # Zero-variance groups: every rollout scored identically, so the
        # group-centered advantage is 0 for all members — the task taught
        # nothing THIS time. Cooldown-pool it (with recycling) rather than
        # retiring it: a task that is flat today can become variant as the
        # policy improves. Mean-based easy/hard pools cannot see the
        # all-0.5 flat class, which measured ~20% of groups (2026-08-16).
        self.flat_examples: list[dict] = []
        self.consumed_examples: list[dict] = []

        # Initialize rollout buffer (flat list of rollouts)
        self.rollout_buffer: list[vf.RolloutOutput] = []

        # Rollout write-ahead log: completed rollouts become durable the
        # moment update() ingests them, so an orchestrator restart replays
        # them instead of regenerating (measured cost of mid-step bounces
        # before this existed: 128-192 rollouts / 2-3h each, 2026-08-20/21).
        # Lifecycle: attach_wal() arms it; update() appends delivered
        # rollouts; save() truncates (the checkpoint owns them from then on);
        # replay_wal() restores the post-checkpoint delta after load().
        # NOTE for step purges: deleting a poisoned step's rollouts must also
        # delete the live_spool dir, or replay resurrects them.
        self._wal_path: Path | None = None

        # Per-task difficulty memory for frontier-biased sampling: EMA of
        # group-mean reward keyed "env:example_id", updated on every completed
        # group (including flat/easy/hard-pooled ones — a 0.5-carpet's 0.5 EMA
        # correctly lands it off-band). Persisted in checkpoints; advisory
        # only, so absent history (fresh tasks, old checkpoints) is fine.
        self.example_reward_ema: dict[str, float] = {}
        # Per-task DISPERSION memory: EMA of the group's reward std. This —
        # not the mean — is the frontier signal, because GRPO divides
        # advantages by the group std, so a group's training value IS its
        # dispersion. Measured 2026-08-22: a FACET group of 61x0.5 + 2x0.0
        # has mean 0.48 (would be boosted by a mean-band rule) but std 0.088,
        # versus 0.39 for a genuine 12-win hard-task group.
        self.example_std_ema: dict[str, float] = {}
        self._long_draw_streak: int = 0

        self.reset_step_metrics()

    def _pick_example(self, env_name: str) -> dict:
        """Within-env draw; frontier-biased when midband_sampling_boost is set.

        The frontier signal is the task's recent reward DISPERSION, not its
        mean: boost tasks whose group std >= gradient_std_high (they produce
        real advantage spread), demote carpets below gradient_std_low (95% of
        rollouts identical => ~no gradient however good the mean looks), and
        leave unseen tasks at 1.0 so fresh material still gets explored.
        Demoted tasks stay drawable — difficulty migrates as the policy
        improves and the EMA keeps tracking them.
        """
        examples = list(self.example_buffer[env_name].values())
        boost = self.config.midband_sampling_boost
        if not boost or boost <= 1.0:
            return random.choice(examples)
        hi = self.config.gradient_std_high if self.config.gradient_std_high is not None else 0.15
        lo = self.config.gradient_std_low if self.config.gradient_std_low is not None else 0.08
        weights = []
        for ex in examples:
            std = self.example_std_ema.get(f"{env_name}:{ex['example_id']}")
            if std is None:
                weights.append(1.0)          # unseen task: explore uniformly
            elif std >= hi:
                weights.append(boost)        # real dispersion = real gradient
            elif std < lo:
                weights.append(1.0 / boost)  # carpet (uniform outcome): demote
            else:
                weights.append(1.0)
        return random.choices(examples, weights=weights, k=1)[0]

    def attach_wal(self, spool_dir: Path) -> None:
        """Arms the rollout write-ahead log under the given directory."""
        spool_dir.mkdir(parents=True, exist_ok=True)
        self._wal_path = spool_dir / "rollout_wal.jsonl"

    def _wal_append(self, rollouts: list[vf.RolloutOutput]) -> None:
        if self._wal_path is None:
            return
        with open(self._wal_path, "a") as f:
            for r in rollouts:
                f.write(json.dumps(r, default=make_serializable) + "\n")
            f.flush()

    def _wal_truncate(self) -> None:
        if self._wal_path is not None and self._wal_path.exists():
            self._wal_path.unlink()

    def replay_wal(self) -> int:
        """Restores post-checkpoint completed rollouts after a restart.

        Idempotent: entries already present in rollout_buffer (full-record
        hash) are skipped, so replay-after-clean-checkpoint is a no-op.
        Returns the number of rollouts restored.
        """
        if self._wal_path is None or not self._wal_path.exists():
            return 0
        seen = {
            hashlib.sha256(json.dumps(r, sort_keys=True, default=make_serializable).encode()).hexdigest()
            for r in self.rollout_buffer
        }
        restored = 0
        with open(self._wal_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                key = hashlib.sha256(json.dumps(r, sort_keys=True, default=make_serializable).encode()).hexdigest()
                if key in seen:
                    continue
                seen.add(key)
                self.rollout_buffer.append(cast(vf.RolloutOutput, r))
                restored += 1
        if restored:
            logger.info(
                f"WAL replay: restored {restored} completed rollout(s) generated after the "
                f"last checkpoint — they will not be regenerated"
            )
        return restored

    def get_example_hash(self, example: dict) -> str:
        """Returns a hash of the example based on hash keys."""
        hash_keys = [key for key in self.config.hash_keys if key in example]
        assert hash_keys, "No hashable keys found in example."
        return hashlib.sha256(json.dumps([example[key] for key in hash_keys]).encode()).hexdigest()

    def save(self, path: Path) -> None:
        """Saves pool assignments and rollout buffer."""
        path.mkdir(parents=True, exist_ok=True)

        def write_jsonl(lst: list, path: Path) -> None:
            with open(path, "w") as f:
                for item in lst:
                    f.write(json.dumps(item, default=make_serializable) + "\n")

        write_jsonl(self.easy_examples, path / "easy_examples.jsonl")
        write_jsonl(self.hard_examples, path / "hard_examples.jsonl")
        write_jsonl(self.flat_examples, path / "flat_examples.jsonl")
        write_jsonl(self.consumed_examples, path / "consumed_examples.jsonl")
        write_jsonl(self.rollout_buffer, path / "rollout_buffer.jsonl")
        with open(path / "reward_ema.json", "w") as f:
            json.dump(self.example_reward_ema, f)
        with open(path / "reward_std_ema.json", "w") as f:
            json.dump(self.example_std_ema, f)
        # The checkpoint now owns every delivered rollout; the WAL restarts empty.
        self._wal_truncate()

    def load(self, path: Path) -> None:
        """Loads pool assignments and rollouts."""

        def read_jsonl(path: Path) -> list[dict]:
            with open(path) as f:
                return [json.loads(line) for line in f]

        saved_easy_examples = read_jsonl(path / "easy_examples.jsonl")
        saved_hard_examples = read_jsonl(path / "hard_examples.jsonl")
        # Checkpoints written before the flat pool existed have no file;
        # treat absent as empty (forward-compatible resume).
        flat_path = path / "flat_examples.jsonl"
        saved_flat_examples = read_jsonl(flat_path) if flat_path.exists() else []
        consumed_path = path / "consumed_examples.jsonl"
        if self.config.sample_without_replacement and not consumed_path.is_file():
            raise ValueError(
                "one-use buffer checkpoint has no consumed_examples.jsonl ledger"
            )
        saved_consumed_examples = (
            read_jsonl(consumed_path) if consumed_path.is_file() else []
        )
        saved_rollout_buffer = cast(list[vf.RolloutOutput], read_jsonl(path / "rollout_buffer.jsonl"))
        # Difficulty EMA: advisory memory; absent on pre-feature checkpoints.
        ema_path = path / "reward_ema.json"
        if ema_path.exists():
            with open(ema_path) as f:
                self.example_reward_ema = json.load(f)
        std_path = path / "reward_std_ema.json"
        if std_path.exists():
            with open(std_path) as f:
                self.example_std_ema = json.load(f)

        if (
            any(saved_easy_examples)
            or any(saved_hard_examples)
            or any(saved_consumed_examples)
            or any(saved_rollout_buffer)
        ):
            # Build hash lookup for example buffer (env -> (example_hash -> example_id))
            example_hash_lookup = defaultdict(dict)
            all_hashes = set()
            for env in self.example_buffer:
                for example_id, example in self.example_buffer[env].items():
                    example_hash = self.get_example_hash(example)
                    if example_hash in all_hashes:
                        logger.warning(
                            f"Duplicate example hash found based on hash_keys={self.config.hash_keys}. Overwriting with latest example. This may cause unexpected behavior when resuming the buffer."
                        )
                    example_hash_lookup[env][example_hash] = example_id
                    all_hashes.add(example_hash)

            def move_saved_pool(saved_examples: list[dict], target_pool: list[dict]) -> int:
                """Moves saved examples to the target pool from example buffer based on hash lookup."""
                num_moved = 0
                for example in saved_examples:
                    example_hash = self.get_example_hash(example)
                    for env in example_hash_lookup:
                        if example_hash in example_hash_lookup[env]:
                            example_id = example_hash_lookup[env][example_hash]
                            example = self.example_buffer[env].pop(example_id, None)
                            if example is not None:
                                target_pool.append(example)
                                num_moved += 1
                                break
                return num_moved

            if any(saved_flat_examples):
                move_saved_pool(saved_flat_examples, self.flat_examples)
            if any(saved_easy_examples):
                num_moved = move_saved_pool(saved_easy_examples, self.easy_examples)
                logger.debug(f"Loaded {num_moved}/{len(saved_easy_examples)} example(s) to easy pool from checkpoint.")
                if num_moved != len(saved_easy_examples):
                    num_not_moved = len(saved_easy_examples) - num_moved
                    logger.warning(
                        f"Could not move {num_not_moved} example(s) from checkpoint to easy pool. This usually means you resumed with an env mix that does not contain all previous examples."
                    )

            if any(saved_hard_examples):
                num_moved = move_saved_pool(saved_hard_examples, self.hard_examples)
                logger.debug(f"Moved {num_moved}/{len(saved_hard_examples)} example(s) to hard pool from checkpoint.")
                if num_moved != len(saved_hard_examples):
                    num_not_moved = len(saved_hard_examples) - num_moved
                    logger.warning(
                        f"Could not move {num_not_moved} example(s) from checkpoint to hard pool. This usually means you resumed with an env mix that does not contain all previous examples."
                    )

            if any(saved_consumed_examples):
                num_moved = move_saved_pool(
                    saved_consumed_examples,
                    self.consumed_examples,
                )
                if num_moved != len(saved_consumed_examples):
                    raise ValueError(
                        "one-use buffer checkpoint does not match the current dataset"
                    )
                logger.debug(
                    f"Restored {num_moved} consumed one-use example(s) from checkpoint."
                )

            if any(saved_rollout_buffer):
                # Extend rollout buffer, but only include rollouts for which the example still exists in the example buffer
                valid_saved_rollouts = [
                    rollout for rollout in saved_rollout_buffer if rollout["task"] in self.env_names
                ]
                self.rollout_buffer.extend(valid_saved_rollouts)
                logger.debug(f"Loaded {len(valid_saved_rollouts)} rollout(s) from checkpoint.")

            # Load rollouts, filtering out removed environments and problems
            num_easy_examples = len(self.easy_examples)
            num_moved = self._move_examples_to_normal(self.easy_examples, self.config.easy_fraction)
            logger.debug(f"Converted {num_moved}/{num_easy_examples} example(s) back to normal from easy pool.")
            num_hard_examples = len(self.hard_examples)
            num_moved = self._move_examples_to_normal(self.hard_examples, self.config.hard_fraction)
            logger.debug(f"Converted {num_moved}/{num_hard_examples} example(s) back to normal from hard pool.")
        else:
            logger.debug("No easy/ hard examples or rollouts found in checkpoint")

    def _normal_example_count(self) -> int:
        return sum(len(examples) for examples in self.example_buffer.values())

    def _move_examples_to_normal(self, examples: list[dict], fraction: float | None) -> int:
        """Moves a fraction of examples from an easy/hard pool back to normal."""
        if fraction is None or fraction <= 0.0 or not examples:
            return 0
        num_moved = round(len(examples) * fraction)
        if num_moved <= 0:
            num_moved = 1
        num_moved = min(num_moved, len(examples))
        for _ in range(num_moved):
            example = random.choice(examples)
            env_name = example["task"]
            example_id = example["example_id"]
            examples.remove(example)
            self.example_buffer[env_name][example_id] = example
        return num_moved

    def _recycle_examples_if_needed(self) -> None:
        min_normal = self.config.normal_pool_min_examples or 0
        if self._normal_example_count() <= min_normal:
            moved_easy = self._move_examples_to_normal(self.easy_examples, self.config.recycle_easy_fraction)
            moved_hard = self._move_examples_to_normal(self.hard_examples, self.config.recycle_hard_fraction)
            moved_flat = self._move_examples_to_normal(
                self.flat_examples, self.config.recycle_flat_fraction)
            self.recycled_examples_per_step["hard"] += moved_flat
            self.recycled_examples_per_step["easy"] += moved_easy
            self.recycled_examples_per_step["hard"] += moved_hard

            if moved_easy or moved_hard:
                logger.info(
                    "Recycled %d easy and %d hard example(s) into the normal pool "
                    "(normal=%d, easy=%d, hard=%d)",
                    moved_easy,
                    moved_hard,
                    self._normal_example_count(),
                    len(self.easy_examples),
                    len(self.hard_examples),
                )
        # Per-env starvation guard: sample_examples() deals only envs whose
        # NORMAL pool is non-empty, and the global floor above cannot trip
        # while other envs stay full — so an env whose every example was
        # classified easy/hard is silently never dealt again (an env can get
        # there through an outage that scores its whole registry under
        # hard_threshold). Return ALL of that env's easy/hard examples to
        # normal; difficulty re-classifies as fresh results arrive.
        for env_name, prob in self.env_probs.items():
            if prob <= 0 or self.example_buffer.get(env_name):
                continue
            starved = [e for e in self.easy_examples if e["task"] == env_name] + [
                e for e in self.hard_examples if e["task"] == env_name
            ]
            for example in starved:
                if example in self.easy_examples:
                    self.easy_examples.remove(example)
                else:
                    self.hard_examples.remove(example)
                self.example_buffer.setdefault(env_name, {})[example["example_id"]] = example
                self.recycled_examples_per_step["hard"] += 1
            if starved:
                logger.info(
                    "Env %s normal pool was empty while weighted %.3f — recycled "
                    "%d of its easy/hard example(s) back to normal.",
                    env_name,
                    prob,
                    len(starved),
                )

    def sample_examples(self, n: int) -> list[dict]:
        """Samples n examples from the buffer, respecting env ratios."""

        self._recycle_examples_if_needed()
        if self.config.sample_without_replacement:
            sampled_examples = []
            for _ in range(n):
                non_empty_envs = [
                    env for env, examples in self.example_buffer.items() if examples
                ]
                if not non_empty_envs:
                    raise ValueError("No environments left with examples.")
                sampled_env = random.choices(
                    non_empty_envs,
                    weights=[self.env_probs[env] for env in non_empty_envs],
                    k=1,
                )[0]
                sampled_example = self._pick_example(sampled_env)
                self.example_buffer[sampled_env].pop(sampled_example["example_id"])
                self.consumed_examples.append(sampled_example)
                sampled_examples.append(sampled_example)
            return sampled_examples

        non_empty_envs = [env for env, examples in self.example_buffer.items() if examples]

        if not non_empty_envs:
            raise ValueError("No environments left with examples.")

        non_empty_env_probs = [self.env_probs[env] for env in non_empty_envs]
        sampled_examples = []
        for sampled_env in random.choices(non_empty_envs, weights=non_empty_env_probs, k=n):
            sampled_examples.append(self._pick_example(sampled_env))

        self._ensure_short_prompt_example(sampled_examples, non_empty_envs)
        return sampled_examples

    @staticmethod
    def _prompt_chars(example: dict) -> int:
        prompt = example.get("prompt")
        if isinstance(prompt, str):
            return len(prompt)
        if isinstance(prompt, list):
            return sum(len(str(m.get("content", ""))) for m in prompt if isinstance(m, dict))
        return 0

    def _ensure_short_prompt_example(self, sampled: list[dict], envs: list[str]) -> None:
        """Guarantee >=1 short-prompt example per STEP (vtc rescue, issue #74).

        Chunked GRPO takes the step's ValidTokenCount from the LAST micro's
        CHUNK 0. If every sample's prompt is longer than the chunk, chunk 0
        contains no completion tokens, vtc reads 0 and the WHOLE STEP is
        discarded (measured: step 103, 0/256 samples under 1792 tokens, ~2h
        of generation lost; healthy step 102 had 64/256). The trainer's
        reorder guard can only pin the best micro last — it cannot create one.
        So the fix belongs here: swap a drawn example for a short-prompt one
        when a whole step's worth of draws has produced none.

        The window is what makes this safe. The orchestrator draws ONE example
        per group, so a per-draw guarantee silently becomes "every group must
        be short" — which collapsed step 104 into an all-CRM batch pinned to a
        single worker (~24h ETA, two GPUs idle) before this was caught. Forcing
        at most `need` swaps per `vtc_rescue_window` issued examples keeps the
        real guarantee (any window of `window` consecutive groups holds a short
        one, so every step has its valid chunk-0 micro) while costing ~1 group
        in `window` of env-ratio distortion instead of the entire mix.
        """
        need = self.config.vtc_min_short_prompt_examples
        if not need:
            return
        max_chars = self.config.vtc_short_prompt_max_chars or 6000
        short_n = sum(1 for e in sampled if self._prompt_chars(e) <= max_chars)
        if short_n >= need:
            self._long_draw_streak = 0
            return
        # This draw is all-long. A later draw in the same step can still supply
        # the short micro, so only intervene once a full window has gone by.
        self._long_draw_streak += len(sampled)
        window = self.config.vtc_rescue_window or 4
        if self._long_draw_streak < window:
            return
        candidates = [
            e for env in envs for e in self.example_buffer[env].values()
            if self._prompt_chars(e) <= max_chars
        ]
        if not candidates:
            logger.warning(
                "vtc rescue: batch has %d/%d short-prompt examples (<=%d chars) and the "
                "buffer holds NONE — the step may be discarded (issue #74)",
                short_n, need, max_chars)
            return
        for slot in range(len(sampled)):
            if short_n >= need:
                break
            if self._prompt_chars(sampled[slot]) <= max_chars:
                continue
            sampled[slot] = random.choice(candidates)
            short_n += 1
            self._long_draw_streak = 0
            logger.info(
                "vtc rescue: swapped a long-prompt example for a short one (<=%d chars) "
                "after %d all-long draws so the step keeps a valid chunk-0 micro",
                max_chars, window)

    def update(self, rollouts: list[vf.RolloutOutput]):
        """Updates the buffer state with completed rollouts."""

        rollouts_by_example = defaultdict(list)
        for rollout in rollouts:
            rollouts_by_example[rollout["example_id"]].append(rollout)

        for example_id, example_rollouts in rollouts_by_example.items():
            rewards = [r["reward"] for r in example_rollouts]
            avg_reward = mean(rewards)
            env_name = example_rollouts[0]["task"]

            ema_key = f"{env_name}:{example_id}"
            prev_ema = self.example_reward_ema.get(ema_key)
            self.example_reward_ema[ema_key] = (
                avg_reward if prev_ema is None else 0.7 * prev_ema + 0.3 * avg_reward
            )
            group_std = pstdev(rewards) if len(rewards) > 1 else 0.0
            prev_std = self.example_std_ema.get(ema_key)
            self.example_std_ema[ema_key] = (
                group_std if prev_std is None else 0.7 * prev_std + 0.3 * group_std
            )

            reward_spread = max(rewards) - min(rewards)
            if (
                self.config.flat_group_filtering
                and len(example_rollouts) > 1
                and (
                    len(set(rewards)) == 1
                    or reward_spread <= (self.config.flat_group_epsilon or 0.0)
                    or (
                        self.config.flat_group_std_min is not None
                        and len(rewards) > 1
                        and pstdev(rewards) < self.config.flat_group_std_min
                    )
                )
            ):
                # Zero- OR near-zero-variance group: exclude its rollouts from
                # the batch and cool the task down. The exact-match trigger
                # alone misses the dominant live pattern (measured run 4 steps
                # 13-18: terminal groups at 61-63/64 identical 0.5s with one
                # 0.4 outlier, std 0.012-0.09 — one outlier defeats set()==1
                # while the group still carries ~no gradient and holds a
                # 0.4-weight lane slot).
                logger.info(
                    f"[{env_name}] FLAT-DROP example={example_id} "
                    f"n={len(example_rollouts)} rewards(min/max/mean)="
                    f"{min(rewards):.2f}/{max(rewards):.2f}/{avg_reward:.2f} "
                    f"spread={reward_spread:.3f} — excluded from batch, task cooled down"
                )
                if example_id in self.example_buffer[env_name]:
                    example = self.example_buffer[env_name].pop(example_id)
                    self.flat_examples.append(example)
                self.num_examples_per_step[env_name]["hard"] += 1
                self.num_rollouts_per_step[env_name]["hard"] += len(example_rollouts)
                continue

            if self.config.easy_threshold is not None and avg_reward >= self.config.easy_threshold:
                pool = "easy"
            elif self.config.hard_threshold is not None and avg_reward <= self.config.hard_threshold:
                pool = "hard"
            else:
                pool = "normal"

            if pool != "normal" and example_id in self.example_buffer[env_name]:
                example = self.example_buffer[env_name].pop(example_id)
                target_pool = self.easy_examples if pool == "easy" else self.hard_examples
                target_pool.append(example)

            self.num_examples_per_step[env_name][pool] += 1
            if self.config.online_difficulty_filtering:
                if avg_reward == 0.0:
                    self.num_rollouts_per_step[env_name]["hard"] += len(example_rollouts)
                    continue
                elif avg_reward == 1.0:
                    self.num_rollouts_per_step[env_name]["easy"] += len(example_rollouts)
                    continue

            self.num_rollouts_per_step[env_name]["normal"] += len(example_rollouts)
            self.rollout_buffer.extend(example_rollouts)
            self._wal_append(example_rollouts)

    def sample_rollouts(self, n: int) -> list[vf.RolloutOutput]:
        """Samples the latest n rollouts from the buffer."""
        n = min(n, len(self.rollout_buffer))
        sampled_rollouts = self.rollout_buffer[-n:]
        self.rollout_buffer = self.rollout_buffer[:-n]
        return sampled_rollouts

    def reset_step_metrics(self) -> None:
        """Reset per-step metrics (called after get_metrics)."""
        zero_per_pool = lambda: {p: 0 for p in self.POOLS}
        # num examples per env per step per pool (env_name -> (pool -> num_examples))
        self.num_examples_per_step = {env: zero_per_pool() for env in self.env_names}
        # num rollouts per env per step per pool (env_name -> (pool -> num_rollouts))
        self.num_rollouts_per_step = {env: zero_per_pool() for env in self.env_names}
        self.recycled_examples_per_step = {"easy": 0, "hard": 0}

    def get_metrics(self) -> dict[str, float]:
        """Returns the buffer metrics for the current step."""

        metrics = {}

        # sum over envs (e.g. log globally)
        num_examples_per_step_per_pool = {
            pool: sum(self.num_examples_per_step[env][pool] for env in self.env_names) for pool in self.POOLS
        }
        num_rollouts_per_step_per_pool = {
            pool: sum(self.num_rollouts_per_step[env][pool] for env in self.env_names) for pool in self.POOLS
        }
        num_examples_per_step = sum(num_examples_per_step_per_pool.values())
        num_rollouts_per_step = sum(num_rollouts_per_step_per_pool.values())

        for pool in ["easy", "hard"]:
            if num_examples_per_step:
                metrics[f"evicted_examples/{pool}"] = num_examples_per_step_per_pool[pool] / num_examples_per_step
            if num_rollouts_per_step:
                metrics[f"filtered_rollouts/{pool}"] = num_rollouts_per_step_per_pool[pool] / num_rollouts_per_step

        total_normal = sum(len(self.example_buffer[env]) for env in self.env_names)
        pool_counts = [len(self.easy_examples), total_normal, len(self.hard_examples)]
        pool_ratios = mean_normalize(pool_counts)
        for pool, pool_ratio in zip(self.POOLS, pool_ratios):
            metrics[f"pool/{pool}"] = pool_ratio

        for pool in ["easy", "hard"]:
            metrics[f"recycled_examples/{pool}"] = self.recycled_examples_per_step[pool]

        self.reset_step_metrics()

        return metrics
