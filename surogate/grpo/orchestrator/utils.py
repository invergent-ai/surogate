import asyncio
import time
from collections import defaultdict
from dataclasses import dataclass
from itertools import cycle
from pathlib import Path
from typing import Any, AsyncContextManager

import pandas as pd
import verifiers as vf
from openai.types.chat.chat_completion import ChatCompletion, Choice
from openai.types.completion_usage import CompletionUsage
from rich.console import Console
from rich.table import Table
from verifiers.utils.async_utils import maybe_semaphore
from verifiers.utils.client_utils import setup_openai_client

from surogate.core.config.grpo_orch_config import GRPOSamplingConfig
from surogate.grpo.transport import TrainingSample
from surogate.grpo.utils.client import InferencePool
from surogate.grpo.utils.utils import (
    format_num,
    format_time,
    get_broadcast_dir,
    get_ckpt_dir,
    get_step_path,
)
from surogate.utils.logger import get_logger

logger = get_logger()

SEMAPHORE: AsyncContextManager | None = None


async def set_semaphore(limit: int):
    global SEMAPHORE
    SEMAPHORE = await maybe_semaphore(limit)


async def get_semaphore() -> AsyncContextManager:
    global SEMAPHORE
    assert SEMAPHORE is not None, "Semaphore not set"
    return SEMAPHORE


def get_sampling_args(sampling_config: GRPOSamplingConfig, temperature: float, routing_replay: bool = False) -> dict:
    # Convert SamplingConfig to the OpenAI-compatible sampling args the server accepts
    sampling_args = dict(sampling_config.__dict__)
    sampling_args.pop("temp_scheduler", None)
    # A seed reaching the request body pins every rollout of a group to the same
    # sample: the engine resolves a request seed ahead of its own server setting
    # (`serve/translate.cpp:62-67`), so it does this even when the server passes
    # no `--seed` at all. Identical rollouts score identically, and a group with
    # no reward spread has an advantage of exactly zero, so the step trains on
    # nothing. Colocate is no safer: `shared_model.py` seeds its per-request RNG
    # from the same value. A fixed seed has no per-rollout meaning, so it is
    # dropped rather than honoured.
    #
    # Both doors have to be shut. `extra_body` is rebuilt from the config below
    # rather than carried through `sampling_args`, so this pop cannot reach a
    # seed nested in it, and the clients merge `extra_body` into the top level of
    # the request where the engine reads it just the same.
    seed_pinned = sampling_args.pop("seed", None) is not None
    sampling_args["temperature"] = temperature
    if sampling_args.get("top_p") is None:
        sampling_args["top_p"] = 1.0
    sampling_args["logprobs"] = True
    # Convert extra_body to plain dicts (DictDefault/addict breaks Pydantic serialization)
    import json

    plain_extra_body = (
        json.loads(json.dumps(dict(sampling_config.extra_body), default=str)) if sampling_config.extra_body else {}
    )
    sampling_args["extra_body"] = {
        **plain_extra_body,
        "return_token_ids": True,  # Always return token IDs
        "top_k": -1,
        "min_p": 0.0,
    }
    sampling_args["extra_body"]["min_tokens"] = sampling_args.pop("min_tokens")
    sampling_args["extra_body"]["repetition_penalty"] = sampling_args.pop("repetition_penalty")
    if routing_replay:
        # Rollout Routing Replay: ask the engine for the experts it picked per token; verifiers keeps the
        # `routed_experts` payload on each trajectory step (see surogate/grpo/routing_replay.py).
        sampling_args["extra_body"]["return_routed_experts"] = True
    seed_pinned |= sampling_args["extra_body"].pop("seed", None) is not None
    if seed_pinned:
        logger.warning_once(
            "a sampling seed is ignored for rollouts: one seed across a group makes every "
            "rollout in it identical, which gives an advantage of zero and trains on "
            "nothing. Remove `seed` from the orchestrator config's sampling section, "
            "including from `sampling.extra_body`."
        )
    return sampling_args


def parse_num_completion_tokens(responses: list[list[ChatCompletion]]) -> list[int]:
    """Parses the number of tokens from a list of chat completions returned by OAI API."""
    all_num_completion_tokens = []
    for response in responses:
        num_completion_tokens = 0
        for chat_completion in response:
            assert isinstance(chat_completion, ChatCompletion)
            assert chat_completion.usage is not None, "Usage should be present in the response"
            usage = chat_completion.usage
            assert isinstance(usage, CompletionUsage)
            num_completion_tokens += usage.completion_tokens
        all_num_completion_tokens.append(num_completion_tokens)
    assert len(all_num_completion_tokens) == len(responses), (
        "Number of completion tokens should be the same as the number of responses"
    )
    return all_num_completion_tokens


def parse_is_truncated_completions(responses: list[list[ChatCompletion]]) -> list[bool]:
    """Parses whether the completions were truncated from a list of (multi-turn) OAI chat completions"""
    all_is_truncated = []
    for response in responses:
        is_truncated = False
        for chat_completion in response:
            assert isinstance(chat_completion, ChatCompletion)
            assert len(chat_completion.choices) == 1, "Response should always have one choice"
            choice = chat_completion.choices[0]
            assert isinstance(choice, Choice)
            if choice.finish_reason == "length":
                is_truncated = True
        all_is_truncated.append(is_truncated)
    return all_is_truncated


def print_benchmark(history: dict[str, list[Any]]) -> None:
    """
    Print benchmark results as rich table. Shows formatted values for the
    inference throughput and overall step time. First first N rows show the
    per-step values, and the last row shows the mean, std, min, and max values.
    """
    history.pop("step")
    assert all(len(v) for v in history.values()), "All metrics must have logged the same number of steps"

    # Turn metric history into pd.DataFrame
    df = pd.DataFrame(dict(history.items()))
    columns = {
        "perf/throughput": "Throughput",
        "time/step": "Step Time",
    }
    df = df.rename(columns=columns)
    df = df[list(columns.values())]
    df = df.iloc[1:]  # Exclude first row

    # Setup console
    console = Console()
    table = Table(title="Benchmark")

    # Add columns
    table.add_column("Step", justify="right")
    for col in df.columns:
        table.add_column(col, justify="center", style="magenta")

    # Add formatted rows
    formatted_df = pd.DataFrame(columns=df.columns)
    formatted_df["Step Time"] = df["Step Time"].apply(format_time)
    formatted_df["Throughput"] = df["Throughput"].apply(format_num, precision=2)
    for step, row in formatted_df.iterrows():
        table.add_row(*([str(step)] + [str(x) for x in row]))

    # Separator
    num_table_columns = 1 + len(df.columns)
    table.add_row(*([""] * num_table_columns))

    # Add row for formatted, aggregated statistics
    mean_df = df.describe().loc[["mean", "std", "min", "max"], :]
    formatted_mean_df = pd.DataFrame(columns=mean_df.columns)
    formatted_mean_df["Step Time"] = mean_df["Step Time"].apply(format_time)
    formatted_mean_df["Throughput"] = mean_df["Throughput"].apply(format_num, precision=2)
    mean_row = ["Overall"] + formatted_mean_df.T.apply(
        lambda row: f"{row['mean']} ± {row['std']} [{row['min']}, {row['max']}]", axis=1
    ).tolist()
    table.add_row(*mean_row)

    # Display table
    console.print(table)


@dataclass(frozen=True)
class TeacherRule:
    """A rule's tokens, ready to splice into a ChatML token sequence that only the teacher scores.

    Built with the student's tokenizer, which the teacher shares: it scores the student's exact tokens plus these.
    """

    im_start_id: int
    im_end_id: int
    system_ids: tuple[int, ...]  # the role name after <|im_start|> that opens a system message
    appended_ids: tuple[int, ...]  # "\n\n" + rule, for the end of the prompt's own system message
    block_ids: tuple[int, ...]  # a whole system message holding the rule, for prompts without one

    @classmethod
    def from_tokenizer(cls, tokenizer: Any, rule: str) -> "TeacherRule":
        im_start_id = tokenizer.convert_tokens_to_ids("<|im_start|>")
        im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
        if any(token_id is None or token_id == tokenizer.unk_token_id for token_id in (im_start_id, im_end_id)):
            raise ValueError("teacher_model.rule needs a ChatML tokenizer, with <|im_start|> and <|im_end|> tokens")

        def encode(text: str) -> tuple[int, ...]:
            return tuple(tokenizer.encode(text, add_special_tokens=False))

        rule = rule.strip()
        return cls(
            im_start_id=im_start_id,
            im_end_id=im_end_id,
            system_ids=encode("system"),
            appended_ids=encode("\n\n" + rule),
            block_ids=(im_start_id, *encode("system\n" + rule), im_end_id, *encode("\n")),
        )

    def insert(self, ids: list[int]) -> tuple[list[int], int, int]:
        """The teacher's token sequence for `ids`, with the start and length of the rule span inside it.

        A sequence that opens with a system message gets the rule at that message's end, just before its
        <|im_end|>; any other gets a system message of its own at position 0. The student's tokens are kept as
        they are, so student position t is teacher position t before the span and t + length after it.
        """
        header = [self.im_start_id, *self.system_ids]
        if ids[: len(header)] == header and self.im_end_id in ids:
            start, inserted = ids.index(self.im_end_id), self.appended_ids
        else:
            start, inserted = 0, self.block_ids
        return ids[:start] + list(inserted) + ids[start:], start, len(inserted)


def actual_token_logprobs(tokens: list[int], prompt_logprobs: Any) -> list[float]:
    """Each position's log-probability of the token actually at that position, from a `prompt_logprobs` array.

    A position's candidates are looked up by token id, never by order: the engine returns them as a JSON object
    whose keys come back sorted, so with alternatives requested the actual token is often not the first entry.
    Position 0 has no context and scores 0.0. Anywhere else a missing actual token is an error, not a guess.
    """
    if not isinstance(prompt_logprobs, list) or len(prompt_logprobs) != len(tokens):
        returned = f"{len(prompt_logprobs)} entries" if isinstance(prompt_logprobs, list) else "no prompt_logprobs"
        raise ValueError(
            f"The teacher returned {returned} for {len(tokens)} tokens; it must score every token it is sent."
        )
    logprobs = []
    for position, (token, entry) in enumerate(zip(tokens, prompt_logprobs)):
        if position == 0:
            logprobs.append(0.0)
            continue
        candidate = (entry.get(str(token)) or entry.get(token)) if isinstance(entry, dict) else None
        if not isinstance(candidate, dict) or candidate.get("logprob") is None:
            raise ValueError(
                f"The teacher's prompt_logprobs at position {position} do not score the token there ({token}). "
                "A teacher must return each token it is sent, keyed by token id."
            )
        logprobs.append(float(candidate["logprob"]))
    return logprobs


async def compute_teacher_logprobs(
    clients: list[vf.ClientConfig],
    model_name: str,
    samples: list[TrainingSample],
    rule: TeacherRule | None = None,
) -> list[list[float]]:
    """Compute teacher model logprobs for a batch of training samples via prefill.

    One value per sample token (prompt + completion), which is the length the packer requires. With a rule the
    teacher scores the sequence with the rule spliced in, and the rule's own positions are dropped again.
    """

    async def _compute_single(client_config: vf.ClientConfig, sample: TrainingSample) -> list[float]:
        client = setup_openai_client(client_config)
        tokens = sample.prompt_ids + sample.completion_ids
        start, inserted = 0, 0
        if rule is not None:
            tokens, start, inserted = rule.insert(tokens)

        async with await get_semaphore():
            response = await client.post(
                "/chat/completions/tokens",
                body={
                    "model": model_name,
                    "messages": [{"role": "user", "content": ""}],
                    "tokens": tokens,
                    "max_tokens": 1,
                    "temperature": 1.0,
                    "top_p": 1.0,
                    "skip_special_tokens": False,
                    # An integer: the engine refuses a boolean (HTTP 400). 0 returns the actual token alone.
                    "prompt_logprobs": 0,
                },
                cast_to=ChatCompletion,
            )
        logprobs = actual_token_logprobs(tokens, getattr(response, "prompt_logprobs", None))
        del logprobs[start : start + inserted]
        return logprobs

    return await asyncio.gather(*[_compute_single(client, sample) for client, sample in zip(cycle(clients), samples)])


@dataclass(frozen=True)
class TeacherRoute:
    """Where a training env's samples are scored: a teacher inference pool, its model name and an optional rule."""

    pool: InferencePool
    model_name: str
    rule: TeacherRule | None = None


async def compute_routed_teacher_logprobs(
    samples: list[TrainingSample],
    sample_envs: list[str | None],
    routes: dict[str, TeacherRoute],
) -> list[list[float]]:
    """Teacher logprobs for samples of several training envs, each sample scored by its env's route.

    `sample_envs` names each sample's env and `routes` maps env names to routes. Samples whose envs share a
    route (pool, model and rule) are scored together, and the results come back in sample order.
    """
    assert len(sample_envs) == len(samples), "every sample needs its env name"
    groups: dict[TeacherRoute, list[int]] = defaultdict(list)
    for index, env_name in enumerate(sample_envs):
        if env_name not in routes:
            raise ValueError(f"A training sample of env {env_name!r} has no teacher (teachers: {sorted(routes)})")
        groups[routes[env_name]].append(index)
    scored = await asyncio.gather(
        *[
            compute_teacher_logprobs(
                route.pool.clients, route.model_name, [samples[index] for index in indices], rule=route.rule
            )
            for route, indices in groups.items()
        ]
    )
    logprobs: list[list[float]] = [[] for _ in samples]
    for indices, group_logprobs in zip(groups.values(), scored):
        for index, sample_logprobs in zip(indices, group_logprobs):
            logprobs[index] = sample_logprobs
    return logprobs


def get_weight_dir(output_dir: Path, step: int, check_exists: bool = True, wait_timeout: int | None = None) -> Path:
    """Get the weight directory for a given checkpoint step.

    Args:
        output_dir: The output directory for the run.
        step: The checkpoint step.
        check_exists: If True, raises FileNotFoundError if no weight directory exists.
            If False, returns the broadcast directory path without checking existence.
        wait_timeout: Maximum time in seconds to wait for a stable directory to appear.
            If None, no waiting is performed.
    """
    ckpt_weight_dir = get_step_path(get_ckpt_dir(output_dir), step) / "weight"
    broadcast_weight_dir = get_step_path(get_broadcast_dir(output_dir), step)

    def find_stable_dir() -> Path | None:
        # For checkpoint weights, check STABLE file in parent directory (checkpoints/step_{step}/STABLE)
        ckpt_step_dir = get_step_path(get_ckpt_dir(output_dir), step)
        if (ckpt_step_dir / "STABLE").exists() and ckpt_weight_dir.exists():
            return ckpt_weight_dir

        # For broadcast weights, check STABLE file in the broadcast directory itself
        if (broadcast_weight_dir / "STABLE").exists() and broadcast_weight_dir.exists():
            return broadcast_weight_dir

        return None

    # Check immediately, then wait if needed
    result = find_stable_dir()
    if result is None and wait_timeout:
        start_time = time.time()
        while time.time() - start_time < wait_timeout:
            time.sleep(1)
            result = find_stable_dir()
            if result:
                break

    if result:
        return result
    if not check_exists:
        return broadcast_weight_dir

    raise FileNotFoundError(f"No weight directory found for checkpoint step {step}")
