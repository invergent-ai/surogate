"""Export retained Fugu GRPO batches as backbone-neutral distillation data."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import msgspec
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from surogate.grpo.transport.types import TrainingBatch


SCHEMA_VERSION = "fugu_paid_rollout_distillation_v1"
SPLIT_SALT = "fugu-paid-rollout-distillation-v1"
EXPECTED_STEPS = range(200)
CURRENT_POOL_FIRST_STEP = 166
MIN_TOPOLOGY_SAMPLES = 2
MIN_REWARD_MARGIN = 0.2
MIN_CHOSEN_MEAN_REWARD = 0.75
IM_START = "<|im_start|>"
IM_END = "<|im_end|>"


class DistillationCorpusError(ValueError):
    """The retained data cannot satisfy the distillation corpus contract."""


@dataclass(frozen=True)
class SourceBatch:
    step: int
    path: Path
    pool_epoch: str


@dataclass(frozen=True)
class ParsedPrompt:
    messages: tuple[dict[str, str], ...]
    task_text: str
    capability_profiles: tuple[dict[str, Any], ...]
    assistant_prefix: str


@dataclass(frozen=True)
class ParsedWorkflow:
    steps: tuple[dict[str, Any], ...]
    topology_signature: str
    workflow_signature: str


def stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _jsonl_write(path: Path, rows: Iterable[Mapping[str, Any]]) -> tuple[int, str]:
    count = 0
    digest = hashlib.sha256()
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            line = stable_json(row) + "\n"
            handle.write(line)
            digest.update(line.encode("ascii"))
            count += 1
    return count, digest.hexdigest()


def _source_path(root: Path, step: int) -> Path:
    if step <= 144:
        bases = (
            root / "output/fugu_ultra_paper/run_default/rollouts_archive",
            root / "output/fugu_ultra_paper/run_default/rollouts",
        )
    else:
        bases = (
            root / "output/fugu_ultra_stage2/run_default/rollouts_archive",
            root / "output/fugu_ultra_stage2/run_default/rollouts",
        )
    found = [base / f"step_{step}/rollouts.bin" for base in bases]
    found = [path for path in found if path.is_file()]
    if len(found) != 1:
        raise DistillationCorpusError(
            f"expected one canonical batch for step {step}, found {len(found)}: {found}"
        )
    return found[0]


def discover_source_batches(root: Path) -> list[SourceBatch]:
    batches = [
        SourceBatch(
            step=step,
            path=_source_path(root, step),
            pool_epoch=("current_pool" if step >= CURRENT_POOL_FIRST_STEP else "legacy_pool"),
        )
        for step in EXPECTED_STEPS
    ]
    if [batch.step for batch in batches] != list(EXPECTED_STEPS):
        raise DistillationCorpusError("canonical batch sequence is not exactly steps 0-199")
    return batches


def _decode_segments(
    token_ids: Sequence[int], tokenizer: PreTrainedTokenizerBase
) -> list[tuple[str, str]]:
    text = tokenizer.decode(
        token_ids,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    segments: list[tuple[str, str]] = []
    cursor = 0
    while cursor < len(text):
        start = text.find(IM_START, cursor)
        if start < 0:
            if text[cursor:].strip():
                raise DistillationCorpusError("nonempty text precedes or follows chat segments")
            break
        if text[cursor:start].strip():
            raise DistillationCorpusError("nonempty text between chat segments")
        body_start = start + len(IM_START)
        newline = text.find("\n", body_start)
        if newline < 0:
            raise DistillationCorpusError("chat segment has no role delimiter")
        role = text[body_start:newline].strip()
        end = text.find(IM_END, newline + 1)
        if end < 0:
            content = text[newline + 1 :]
            cursor = len(text)
        else:
            content = text[newline + 1 : end]
            cursor = end + len(IM_END)
        segments.append((role, content))
    return segments


def _parse_capability_profiles(user_content: str) -> tuple[str, tuple[dict[str, Any], ...]]:
    marker = "\n\nAVAILABLE LANGUAGE MODELS:\n"
    if marker not in user_content:
        raise DistillationCorpusError("prompt has no anonymous capability-profile section")
    task_text, raw_profiles = user_content.rsplit(marker, 1)
    profiles: list[dict[str, Any]] = []
    for line in raw_profiles.strip().splitlines():
        match = re.fullmatch(r"Model\s+(\d+):\s+roles=(.+)", line.strip())
        if match is None:
            raise DistillationCorpusError(f"invalid capability profile line: {line!r}")
        worker_id = int(match.group(1))
        roles = [role.strip() for role in match.group(2).split(",")]
        if worker_id != len(profiles) or not roles or any(not role for role in roles):
            raise DistillationCorpusError(f"invalid capability profile ordering: {line!r}")
        profiles.append(
            {
                "worker_id": worker_id,
                "capability_tags": roles,
            }
        )
    if not profiles:
        raise DistillationCorpusError("prompt has no capability profiles")
    return task_text, tuple(profiles)


def parse_prompt(
    token_ids: Sequence[int], tokenizer: PreTrainedTokenizerBase
) -> ParsedPrompt:
    segments = _decode_segments(token_ids, tokenizer)
    if len(segments) != 3 or [role for role, _ in segments] != [
        "system",
        "user",
        "assistant",
    ]:
        raise DistillationCorpusError(
            f"expected system/user/assistant-prefix prompt, got {[role for role, _ in segments]}"
        )
    task_text, profiles = _parse_capability_profiles(segments[1][1])
    return ParsedPrompt(
        messages=(
            {"role": "system", "content": segments[0][1]},
            {"role": "user", "content": segments[1][1]},
        ),
        task_text=task_text,
        capability_profiles=profiles,
        assistant_prefix=segments[2][1],
    )


def decode_completion(
    token_ids: Sequence[int], tokenizer: PreTrainedTokenizerBase
) -> str:
    text = tokenizer.decode(
        token_ids,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    while text.endswith(IM_END):
        text = text[: -len(IM_END)]
    eos = tokenizer.eos_token
    if eos:
        while text.endswith(eos):
            text = text[: -len(eos)]
    return text.strip()


def _balanced_list(text: str, start: int) -> str | None:
    depth = 0
    quote: str | None = None
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if escaped:
            escaped = False
            continue
        if char == "\\":
            escaped = True
            continue
        if quote is not None:
            if char == quote:
                quote = None
            continue
        if char in "\"'":
            quote = char
        elif char == "[":
            depth += 1
        elif char == "]":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    return None


def _literal_list(text: str, pattern: str) -> list[Any] | None:
    matches = list(re.finditer(pattern + r"\s*=\s*(?=\[)", text, flags=re.IGNORECASE))
    if not matches:
        return None
    raw = _balanced_list(text, matches[-1].end())
    if raw is None:
        return None
    try:
        value = ast.literal_eval(raw)
    except (SyntaxError, ValueError):
        return None
    return value if isinstance(value, list) else None


def parse_workflow(completion: str, *, worker_count: int) -> ParsedWorkflow:
    worker_ids = _literal_list(completion, r"model[_ ]?id")
    subtasks = _literal_list(completion, r"subtasks?")
    access_list = _literal_list(completion, r"access[_ ]?list")
    if worker_ids is None or subtasks is None or access_list is None:
        raise DistillationCorpusError("completion does not contain three literal lists")
    if not (len(worker_ids) == len(subtasks) == len(access_list)):
        raise DistillationCorpusError("workflow lists have different lengths")
    if not 1 <= len(worker_ids) <= 5:
        raise DistillationCorpusError("workflow must contain 1-5 steps")

    steps: list[dict[str, Any]] = []
    for position, (worker_id, subtask, raw_access) in enumerate(
        zip(worker_ids, subtasks, access_list, strict=True)
    ):
        if (
            isinstance(worker_id, bool)
            or not isinstance(worker_id, int)
            or not 0 <= worker_id < worker_count
        ):
            raise DistillationCorpusError(f"step {position} has invalid worker ID")
        if not isinstance(subtask, str) or not subtask.strip():
            raise DistillationCorpusError(f"step {position} has empty subtask")
        if isinstance(raw_access, str):
            raw_access = [raw_access]
        if not isinstance(raw_access, list):
            raise DistillationCorpusError(f"step {position} access is not a list")
        if any(
            isinstance(value, str) and value.strip().lower() == "all"
            for value in raw_access
        ):
            access = list(range(position))
        else:
            if any(
                isinstance(value, bool)
                or not isinstance(value, int)
                or not 0 <= value < position
                for value in raw_access
            ):
                raise DistillationCorpusError(f"step {position} access is not backward-only")
            access = list(raw_access)
        if len(access) != len(set(access)):
            raise DistillationCorpusError(f"step {position} contains duplicate access")
        steps.append(
            {
                "position_id": position,
                "worker_id": worker_id,
                "subtask": subtask.strip(),
                "access": access,
            }
        )

    topology = {
        "steps": [
            {
                "position_id": step["position_id"],
                "worker_id": step["worker_id"],
                "access": step["access"],
            }
            for step in steps
        ]
    }
    workflow = {"steps": steps}
    return ParsedWorkflow(
        steps=tuple(steps),
        topology_signature=sha256_bytes(stable_json(topology).encode("ascii")),
        workflow_signature=sha256_bytes(stable_json(workflow).encode("ascii")),
    )


def _split(prompt_id: str) -> str:
    digest = hashlib.sha256(f"{SPLIT_SALT}\0{prompt_id}".encode("ascii")).digest()
    return "holdout" if int.from_bytes(digest[:8], "big") % 5 == 0 else "train"


def _prompt_id(prompt: ParsedPrompt) -> str:
    surface = {
        "messages": prompt.messages,
        "capability_profiles": prompt.capability_profiles,
    }
    return sha256_bytes(stable_json(surface).encode("ascii"))


def _candidate_id(
    *, prompt_id: str, completion: str, step: int, sample_index: int
) -> str:
    value = f"{prompt_id}\0{completion}\0{step}\0{sample_index}".encode("utf-8")
    return sha256_bytes(value)


def _representative(
    rows: Sequence[Mapping[str, Any]], *, chosen: bool
) -> Mapping[str, Any]:
    ordered = sorted(
        rows,
        key=lambda row: (
            -float(row["reward"]) if chosen else float(row["reward"]),
            -float(row["advantage"]),
            row["candidate_id"],
        ),
    )
    return ordered[0]


def _preference_rows(
    candidates: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in candidates:
        if row["valid"]:
            grouped[(str(row["prompt_id"]), str(row["pool_epoch"]))].append(row)

    preferences: list[dict[str, Any]] = []
    exclusions: Counter[str] = Counter()
    for (prompt_id, pool_epoch), rows in sorted(grouped.items()):
        by_topology: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for row in rows:
            by_topology[str(row["topology_signature"])].append(row)
        eligible = []
        for signature, topology_rows in by_topology.items():
            if len(topology_rows) < MIN_TOPOLOGY_SAMPLES:
                continue
            eligible.append(
                {
                    "signature": signature,
                    "rows": topology_rows,
                    "count": len(topology_rows),
                    "mean_reward": sum(float(row["reward"]) for row in topology_rows)
                    / len(topology_rows),
                }
            )
        if len(eligible) < 2:
            exclusions["fewer_than_two_repeated_topologies"] += 1
            continue
        eligible.sort(key=lambda item: (item["mean_reward"], item["count"], item["signature"]))
        rejected_topology = eligible[0]
        chosen_topology = eligible[-1]
        margin = float(chosen_topology["mean_reward"]) - float(
            rejected_topology["mean_reward"]
        )
        if margin < MIN_REWARD_MARGIN:
            exclusions["reward_margin_below_threshold"] += 1
            continue
        if float(chosen_topology["mean_reward"]) < MIN_CHOSEN_MEAN_REWARD:
            exclusions["chosen_topology_lacks_repeated_success"] += 1
            continue
        chosen = _representative(chosen_topology["rows"], chosen=True)
        rejected = _representative(rejected_topology["rows"], chosen=False)
        preference_id = sha256_bytes(
            f"{prompt_id}\0{pool_epoch}\0{chosen_topology['signature']}\0"
            f"{rejected_topology['signature']}".encode("ascii")
        )
        preferences.append(
            {
                "schema_version": SCHEMA_VERSION,
                "preference_id": preference_id,
                "prompt_id": prompt_id,
                "split": chosen["split"],
                "pool_epoch": pool_epoch,
                "chosen_candidate_id": chosen["candidate_id"],
                "rejected_candidate_id": rejected["candidate_id"],
                "chosen_topology_signature": chosen_topology["signature"],
                "rejected_topology_signature": rejected_topology["signature"],
                "chosen_topology_count": chosen_topology["count"],
                "rejected_topology_count": rejected_topology["count"],
                "chosen_topology_mean_reward": chosen_topology["mean_reward"],
                "rejected_topology_mean_reward": rejected_topology["mean_reward"],
                "reward_margin": margin,
            }
        )
    return preferences, dict(sorted(exclusions.items()))


def build_corpus(
    *, root: Path, source_tokenizer: str | Path, output_dir: Path
) -> dict[str, Any]:
    root = root.resolve()
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise DistillationCorpusError(f"output directory already exists: {output_dir}")
    batches = discover_source_batches(root)
    tokenizer = AutoTokenizer.from_pretrained(
        str(source_tokenizer),
        local_files_only=True,
        trust_remote_code=False,
    )
    decoder = msgspec.msgpack.Decoder(type=TrainingBatch)

    prompts: dict[str, dict[str, Any]] = {}
    candidates: list[dict[str, Any]] = []
    reward_counts: Counter[str] = Counter()
    invalid_reasons: Counter[str] = Counter()
    epoch_counts: Counter[str] = Counter()
    source_manifest: list[dict[str, Any]] = []
    total_prompt_tokens = 0
    total_completion_tokens = 0

    for source in batches:
        raw_batch = source.path.read_bytes()
        batch = decoder.decode(raw_batch)
        if batch.step != source.step:
            raise DistillationCorpusError(
                f"batch step mismatch at {source.path}: encoded {batch.step}, expected {source.step}"
            )
        source_manifest.append(
            {
                "step": source.step,
                "path": source.path.relative_to(root).as_posix(),
                "sha256": sha256_bytes(raw_batch),
                "pool_epoch": source.pool_epoch,
                "examples": len(batch.examples),
            }
        )
        for sample_index, sample in enumerate(batch.examples):
            parsed_prompt = parse_prompt(sample.prompt_ids, tokenizer)
            prompt_id = _prompt_id(parsed_prompt)
            split = _split(prompt_id)
            existing = prompts.get(prompt_id)
            prompt_row = {
                "schema_version": SCHEMA_VERSION,
                "prompt_id": prompt_id,
                "split": split,
                "messages": list(parsed_prompt.messages),
                "task_text": parsed_prompt.task_text,
                "capability_profiles": list(parsed_prompt.capability_profiles),
                "source_assistant_prefix": parsed_prompt.assistant_prefix,
            }
            if existing is not None and stable_json(existing) != stable_json(prompt_row):
                raise DistillationCorpusError(f"prompt hash collision: {prompt_id}")
            prompts[prompt_id] = prompt_row

            completion = decode_completion(sample.completion_ids, tokenizer)
            valid = True
            validation_error: str | None = None
            workflow: dict[str, Any] | None = None
            topology_signature: str | None = None
            workflow_signature: str | None = None
            try:
                parsed_workflow = parse_workflow(
                    completion,
                    worker_count=len(parsed_prompt.capability_profiles),
                )
                workflow = {"steps": list(parsed_workflow.steps)}
                topology_signature = parsed_workflow.topology_signature
                workflow_signature = parsed_workflow.workflow_signature
            except DistillationCorpusError as exc:
                valid = False
                validation_error = str(exc)
                invalid_reasons[validation_error] += 1

            reward = None if sample.reward is None else float(sample.reward)
            advantage = None if sample.advantage is None else float(sample.advantage)
            if reward is None or advantage is None:
                raise DistillationCorpusError(
                    f"step {source.step} sample {sample_index} lacks reward or advantage"
                )
            candidate_id = _candidate_id(
                prompt_id=prompt_id,
                completion=completion,
                step=source.step,
                sample_index=sample_index,
            )
            candidates.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "candidate_id": candidate_id,
                    "prompt_id": prompt_id,
                    "split": split,
                    "pool_epoch": source.pool_epoch,
                    "source_step": source.step,
                    "source_sample_index": sample_index,
                    "reward": reward,
                    "advantage": advantage,
                    "completion": completion,
                    "valid": valid,
                    "validation_error": validation_error,
                    "workflow": workflow,
                    "topology_signature": topology_signature,
                    "workflow_signature": workflow_signature,
                }
            )
            reward_counts[str(reward)] += 1
            epoch_counts[source.pool_epoch] += 1
            total_prompt_tokens += len(sample.prompt_ids)
            total_completion_tokens += len(sample.completion_ids)

    preferences, preference_exclusions = _preference_rows(candidates)
    prompt_rows = [prompts[key] for key in sorted(prompts)]
    candidates.sort(key=lambda row: (row["source_step"], row["source_sample_index"]))
    preferences.sort(key=lambda row: (row["pool_epoch"], row["prompt_id"]))

    output_dir.mkdir(parents=True)
    try:
        prompt_count, prompt_hash = _jsonl_write(output_dir / "prompts.jsonl", prompt_rows)
        candidate_count, candidate_hash = _jsonl_write(
            output_dir / "candidates.jsonl", candidates
        )
        preference_count, preference_hash = _jsonl_write(
            output_dir / "preferences.jsonl", preferences
        )
        split_counts = Counter(row["split"] for row in prompt_rows)
        preference_counts = Counter(
            f"{row['pool_epoch']}:{row['split']}" for row in preferences
        )
        report = {
            "schema_version": SCHEMA_VERSION,
            "status": "ready_for_offline_distillation",
            "external_calls": 0,
            "paid_calls": 0,
            "source": {
                "expected_steps": [0, 199],
                "step_count": len(batches),
                "current_pool_first_step": CURRENT_POOL_FIRST_STEP,
                "tokenizer": str(source_tokenizer),
                "batches": source_manifest,
            },
            "counts": {
                "prompts": prompt_count,
                "candidates": candidate_count,
                "valid_candidates": sum(bool(row["valid"]) for row in candidates),
                "invalid_candidates": sum(not bool(row["valid"]) for row in candidates),
                "preferences": preference_count,
                "prompt_tokens": total_prompt_tokens,
                "completion_tokens": total_completion_tokens,
                "prompt_splits": dict(sorted(split_counts.items())),
                "candidate_pool_epochs": dict(sorted(epoch_counts.items())),
                "preference_pool_epoch_splits": dict(sorted(preference_counts.items())),
                "rewards": dict(sorted(reward_counts.items())),
            },
            "preference_policy": {
                "aggregate_within_prompt_and_pool_epoch": True,
                "group_by": "anonymous worker/access topology",
                "min_samples_per_topology": MIN_TOPOLOGY_SAMPLES,
                "min_mean_reward_margin": MIN_REWARD_MARGIN,
                "min_chosen_mean_reward": MIN_CHOSEN_MEAN_REWARD,
                "one_pair_per_prompt_and_pool_epoch": True,
            },
            "invalid_candidate_reasons": dict(sorted(invalid_reasons.items())),
            "preference_exclusions": preference_exclusions,
            "artifacts": {
                "prompts.jsonl": {"rows": prompt_count, "sha256": prompt_hash},
                "candidates.jsonl": {"rows": candidate_count, "sha256": candidate_hash},
                "preferences.jsonl": {
                    "rows": preference_count,
                    "sha256": preference_hash,
                },
            },
            "training_contract": {
                "source_policy_logprobs_are_not_valid_for_ornith": True,
                "legacy_and_current_pool_rewards_must_not_be_aggregated": True,
                "current_pool_preferences_receive_priority": True,
                "holdout_prompt_groups_must_not_enter_training": True,
                "fresh_paid_collection_authorized": False,
            },
        }
        report_text = json.dumps(report, indent=2, sort_keys=True, ensure_ascii=True) + "\n"
        (output_dir / "report.json").write_text(report_text, encoding="utf-8")
    except BaseException:
        shutil.rmtree(output_dir, ignore_errors=True)
        raise
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--source-tokenizer", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    report = build_corpus(
        root=args.root,
        source_tokenizer=args.source_tokenizer,
        output_dir=args.output_dir,
    )
    print(json.dumps({"output_dir": str(args.output_dir), **report["counts"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
