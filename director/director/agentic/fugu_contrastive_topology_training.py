"""Prepare retained-data contrastive topology training for the Fugu conductor."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import msgspec
import numpy as np
import yaml
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from surogate import _surogate
from director.agentic.fugu_structural_outcome_training import (
    StructuralTrainingError,
    _action_content,
    _chosen_action,
    _initial_state,
    _read_json,
    _read_jsonl,
    _rotate,
    _token_count,
    _verify_distillation_corpus,
    _write_jsonl,
    sha256_file,
)
from surogate.core.config.loader import load_config
from surogate.grpo.batch import prepare_sample
from surogate.grpo.config import GRPOTrainConfig
from surogate.grpo.data import microbatch_to_numpy
from surogate.grpo.trainer import GRPOTrainer
from surogate.grpo.transport import (
    FileSystemTrainingBatchSender,
    TrainingBatch,
    TrainingSample,
)
from ultra.live_control import build_control_action_messages


SCHEMA_VERSION = "fugu_ornith_contrastive_topology_v1"
SEQUENCE_LEN = 2_816
PAIRS_PER_STEP = 8
REPLAY_PER_ACTION = 9
STRUCTURAL_PATTERNS = (
    re.compile(r'"worker_id"\s*:\s*\d+'),
    re.compile(r'"access"\s*:\s*\[[^\]]*\]'),
)
REPLAY_PATTERNS = (
    re.compile(r'"action"\s*:\s*"(?:continue|handoff|replan|complete)"'),
    re.compile(r'"target_position_id"\s*:\s*\d+'),
    *STRUCTURAL_PATTERNS,
)
PLANNER_PATTERNS = (
    re.compile(r"model_id\s*=\s*\[[^\n]*\]"),
    re.compile(r"access_list\s*=\s*\[[^\n]*\]"),
)
# Typed capability contract (unified_capability_action_v2): structure lives in
# anonymous profile selectors and integer access positions, not worker IDs.
TYPED_STRUCTURAL_PATTERNS = (
    re.compile(r'"profile_ref"\s*:\s*"[^"]*"'),
    re.compile(r'"access_positions"\s*:\s*\[[^\]]*\]'),
)
TYPED_REPLAY_PATTERNS = (
    re.compile(r'"action"\s*:\s*"(?:continue|handoff|replan|complete)"'),
    re.compile(r'"target_position_id"\s*:\s*\d+'),
    *TYPED_STRUCTURAL_PATTERNS,
)
FORBIDDEN_IDENTITIES = (
    "gpt-5.6-sol",
    "gemini-3.5-flash",
    "gpt-5.6-terra",
    "grok-4.5",
    "yunwu.ai",
)
REPLAY_MASK_MODES = frozenset({"replay", "planner_replay", "typed_replay"})


def stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def is_replay_mask_mode(mask_mode: object) -> bool:
    return mask_mode in REPLAY_MASK_MODES


def _row_hash(row: Mapping[str, Any]) -> str:
    return hashlib.sha256(str(row["row_id"]).encode("utf-8")).hexdigest()


def _load_needed_candidates(corpus: Path, candidate_ids: set[str]) -> dict[str, dict[str, Any]]:
    candidates: dict[str, dict[str, Any]] = {}
    with (corpus / "candidates.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            candidate_id = row.get("candidate_id")
            if candidate_id in candidate_ids:
                candidates[str(candidate_id)] = row
    if set(candidates) != candidate_ids:
        raise StructuralTrainingError("one or more preference candidates are missing")
    return candidates


def _assert_identity_free(messages: Sequence[Mapping[str, Any]], row_id: str) -> None:
    learned_surface = stable_json(messages).lower()
    leaked = [identity for identity in FORBIDDEN_IDENTITIES if identity in learned_surface]
    if leaked:
        raise StructuralTrainingError(f"row {row_id} leaks concrete identities: {leaked}")


def _preference_rows(
    *, corpus: Path, tokenizer: PreTrainedTokenizerBase
) -> list[dict[str, Any]]:
    prompts = {row["prompt_id"]: row for row in _read_jsonl(corpus / "prompts.jsonl")}
    preferences = sorted(
        (
            row
            for row in _read_jsonl(corpus / "preferences.jsonl")
            if row.get("split") == "train" and row.get("pool_epoch") == "current_pool"
        ),
        key=lambda row: str(row["preference_id"]),
    )
    if len(preferences) != 16:
        raise StructuralTrainingError(
            f"expected 16 current-pool train preferences, found {len(preferences)}"
        )
    candidate_ids = {
        str(row[key])
        for row in preferences
        for key in ("chosen_candidate_id", "rejected_candidate_id")
    }
    candidates = _load_needed_candidates(corpus, candidate_ids)

    rows: list[dict[str, Any]] = []
    pair_index = 0
    for preference in preferences:
        prompt = prompts.get(preference["prompt_id"])
        if prompt is None:
            raise StructuralTrainingError("preference references a missing prompt")
        state = _initial_state(prompt)
        margin = float(preference["reward_margin"])
        if not 0.0 < margin <= 1.0:
            raise StructuralTrainingError("preference reward margin is outside (0, 1]")
        actions = {
            "chosen": _chosen_action(candidates[str(preference["chosen_candidate_id"])]),
            "rejected": _chosen_action(candidates[str(preference["rejected_candidate_id"])]),
        }
        rewards = {
            "chosen": float(preference["chosen_topology_mean_reward"]),
            "rejected": float(preference["rejected_topology_mean_reward"]),
        }
        for rotation in range(len(state.workers)):
            rotated: dict[str, tuple[Any, Any]] = {
                branch: _rotate(state, action, rotation)
                for branch, action in actions.items()
            }
            chosen_state = rotated["chosen"][0]
            if rotated["rejected"][0] != chosen_state:
                raise StructuralTrainingError("pair rotation produced unequal prompt states")
            messages, prompt_tokens, compacted = build_control_action_messages(
                chosen_state,
                prompt_token_counter=lambda value: _token_count(tokenizer, value),
                max_input_tokens=2_400,
            )
            step = pair_index // PAIRS_PER_STEP
            pair_id = f"{preference['preference_id']}::rotation-{rotation}"
            for branch, sign in (("chosen", 1.0), ("rejected", -1.0)):
                content = _action_content(rotated[branch][1])
                row_id = f"{pair_id}::{branch}"
                row_messages = [*messages, {"role": "assistant", "content": content}]
                _assert_identity_free(row_messages, row_id)
                rows.append(
                    {
                        "row_id": row_id,
                        "source": "retained_current_pool_preference",
                        "mask_mode": "topology",
                        "step": step,
                        "pair_id": pair_id,
                        "preference_id": preference["preference_id"],
                        "prompt_id": preference["prompt_id"],
                        "rotation": rotation,
                        "branch": branch,
                        "advantage": sign * margin,
                        "reward": rewards[branch],
                        "reward_margin": margin,
                        "prompt_tokens": prompt_tokens,
                        "prompt_compacted": compacted,
                        "messages": row_messages,
                    }
                )
            pair_index += 1
    if pair_index != 64:
        raise StructuralTrainingError(f"expected 64 rotated preference pairs, found {pair_index}")
    return rows


def _rotation(row_id: str) -> int:
    match = re.search(r"::rotation-(\d+)(?:::|$)", row_id)
    return int(match.group(1)) if match else -1


def _select_replay_rows(path: Path, *, steps: int) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in _read_jsonl(path):
        if row.get("source") == "accepted_unified_replay":
            grouped[str(row.get("action"))].append(row)
    expected_actions = {"continue", "handoff", "replan", "complete"}
    if set(grouped) != expected_actions:
        raise StructuralTrainingError("replay corpus does not cover every control action")

    selected: list[dict[str, Any]] = []
    for action in sorted(expected_actions):
        by_rotation: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for row in sorted(grouped[action], key=_row_hash):
            by_rotation[_rotation(str(row["row_id"]))].append(row)
        action_rows: list[dict[str, Any]] = []
        while len(action_rows) < REPLAY_PER_ACTION:
            progressed = False
            for rotation in range(4):
                if by_rotation[rotation]:
                    action_rows.append(by_rotation[rotation].pop(0))
                    progressed = True
                    if len(action_rows) == REPLAY_PER_ACTION:
                        break
            if not progressed:
                raise StructuralTrainingError(f"insufficient replay rows for {action}")
        selected.extend(action_rows)

    selected.sort(key=lambda row: (str(row["action"]), _row_hash(row)))
    emitted: list[dict[str, Any]] = []
    for index, row in enumerate(selected):
        emitted_row = dict(row)
        emitted_row.update(
            {
                "source": "accepted_unified_replay",
                "mask_mode": "replay",
                "step": index % steps,
                "advantage": 0.0,
                "reward": 0.0,
            }
        )
        _assert_identity_free(emitted_row["messages"], str(emitted_row["row_id"]))
        emitted.append(emitted_row)
    return emitted


def _patterns(mode: str) -> tuple[re.Pattern[str], ...]:
    if mode == "topology":
        return STRUCTURAL_PATTERNS
    if mode == "replay":
        return REPLAY_PATTERNS
    if mode in {"planner_topology", "planner_replay"}:
        return PLANNER_PATTERNS
    if mode == "typed_topology":
        return TYPED_STRUCTURAL_PATTERNS
    if mode == "typed_replay":
        return TYPED_REPLAY_PATTERNS
    raise StructuralTrainingError(f"unknown mask mode: {mode}")


def _token_ids(value: Any) -> list[int]:
    if isinstance(value, Mapping):
        value = value.get("input_ids")
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, list) and len(value) == 1 and isinstance(value[0], list):
        value = value[0]
    if not isinstance(value, list) or any(
        not isinstance(token, int) or isinstance(token, bool) for token in value
    ):
        raise StructuralTrainingError("tokenizer did not return one integer token sequence")
    return value


def _encode_row(
    row: Mapping[str, Any], tokenizer: PreTrainedTokenizerBase
) -> tuple[list[int], list[int], list[bool]]:
    messages = row["messages"]
    prompt_messages = messages[:-1]
    content = messages[-1]["content"]
    prefix = tokenizer.apply_chat_template(
        prompt_messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    rendered = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=False,
    )
    if not rendered.startswith(prefix) or not rendered[len(prefix) :].startswith(content):
        raise StructuralTrainingError("chat template does not preserve assistant content")
    prompt_ids = _token_ids(
        tokenizer.apply_chat_template(
            prompt_messages,
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    )
    full = tokenizer(rendered, add_special_tokens=False, return_offsets_mapping=True)
    full_ids = _token_ids(full["input_ids"])
    if full_ids[: len(prompt_ids)] != prompt_ids:
        raise StructuralTrainingError("assistant target does not extend exact prompt tokens")
    if len(full_ids) > SEQUENCE_LEN:
        raise StructuralTrainingError(
            f"row {row['row_id']} has {len(full_ids)} tokens, above {SEQUENCE_LEN}"
        )
    spans = [
        (len(prefix) + match.start(), len(prefix) + match.end())
        for pattern in _patterns(str(row["mask_mode"]))
        for match in pattern.finditer(content)
    ]
    offsets = full["offset_mapping"]
    full_mask = [
        end > start
        and any(start < span_end and end > span_start for span_start, span_end in spans)
        for start, end in offsets
    ]
    completion_ids = full_ids[len(prompt_ids) :]
    completion_mask = full_mask[len(prompt_ids) :]
    if not completion_ids or not any(completion_mask):
        raise StructuralTrainingError(f"row {row['row_id']} has no masked completion tokens")
    return prompt_ids, completion_ids, completion_mask


def _training_config(*, model: Path, parent_adapter: Path, output_dir: Path, steps: int) -> dict[str, Any]:
    model_config = _read_json(model / "config.json")
    adapter_config = _read_json(parent_adapter / "adapter_config.json")
    num_experts = int(model_config.get("num_experts") or 0)
    config: dict[str, Any] = {
        "model": str(model),
        "adapter_path": str(parent_adapter),
        "adapter_init_mode": "trainable",
        "output_dir": str(output_dir),
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 1,
        "sequence_len": SEQUENCE_LEN,
        "max_steps": steps,
        "logging_steps": 1,
        "learning_rate": 1.0e-6,
        "lr_scheduler_type": "constant",
        "warmup_steps": 0,
        "max_grad_norm": 1.0,
        "weight_decay": 0.0,
        "optimizer": "adamw",
        "gpus": 2 if num_experts else 1,
        "recipe": "fp8-hybrid" if num_experts else "bf16",
        "recompute": True,
        "cpu_training": True,
        "offload_residual": True,
        "lmhead_chunks": 8,
        "lora": True,
        "lora_rank": int(adapter_config["r"]),
        "lora_alpha": float(adapter_config["lora_alpha"]),
        "lora_dtype": "bf16",
        "train_router": False,
        "lora_target_modules": list(adapter_config["target_modules"]),
        "loss": {
            "ipo_mask_low": 0.2,
            "ipo_mask_high": 0.2,
            "adv_tau": 0.1,
            "teacher_tau": 0.0,
            "opd_tau": 0.0,
            "opd_beta": 1.0,
            "replay_tau": 0.05,
            "kl_tau": 0.001,
        },
        "save_steps": 0,
        "checkpoint_dir": str(output_dir),
        "resume_from_checkpoint": False,
        "report_to": ["surogate"],
        "surogate_metrics_path": str(output_dir / "surogate_metrics.jsonl"),
    }
    if num_experts:
        config.update(
            {
                "ep_size": 2,
                "qlora_fp8": True,
                "qlora_offload_experts": False,
            }
        )
    return config


def build_stage(
    *,
    corpus: Path,
    replay_rows: Path,
    model: Path,
    parent_adapter: Path,
    output_dir: Path,
    training_output: Path,
) -> dict[str, Any]:
    paths = [corpus, replay_rows, model, parent_adapter, output_dir, training_output]
    corpus, replay_rows, model, parent_adapter, output_dir, training_output = (
        path.expanduser().resolve() for path in paths
    )
    if output_dir.exists() or training_output.exists():
        raise StructuralTrainingError("stage or training output already exists")
    corpus_report = _verify_distillation_corpus(corpus)
    for name in ("adapter_config.json", "adapter_model.safetensors"):
        if not (parent_adapter / name).is_file():
            raise StructuralTrainingError(f"parent adapter is missing {name}")
    tokenizer = AutoTokenizer.from_pretrained(
        str(model), local_files_only=True, trust_remote_code=False
    )
    preference_rows = _preference_rows(corpus=corpus, tokenizer=tokenizer)
    steps = math.ceil((len(preference_rows) // 2) / PAIRS_PER_STEP)
    replay = _select_replay_rows(replay_rows, steps=steps)
    rows = [*preference_rows, *replay]

    encoded_counts: list[int] = []
    target_counts: list[int] = []
    for row in rows:
        prompt_ids, completion_ids, completion_mask = _encode_row(row, tokenizer)
        encoded_counts.append(len(prompt_ids) + len(completion_ids))
        target_counts.append(sum(completion_mask))
    pair_counts = Counter(str(row.get("branch")) for row in preference_rows)
    if pair_counts != {"chosen": 64, "rejected": 64}:
        raise StructuralTrainingError("preference branch counts are not symmetric")
    step_counts = Counter(int(row["step"]) for row in rows)
    if set(step_counts) != set(range(steps)):
        raise StructuralTrainingError("training steps are not contiguous")

    output_dir.mkdir(parents=True)
    try:
        row_count, rows_hash = _write_jsonl(output_dir / "rows.jsonl", rows)
        config = _training_config(
            model=model,
            parent_adapter=parent_adapter,
            output_dir=training_output,
            steps=steps,
        )
        config_path = output_dir / "train.yaml"
        config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        report = {
            "schema_version": SCHEMA_VERSION,
            "status": "ready_for_native_parent_scoring",
            "external_calls": 0,
            "paid_calls": 0,
            "fresh_worker_trajectories": 0,
            "inputs": {
                "distillation_report_sha256": sha256_file(corpus / "report.json"),
                "distillation_preferences_sha256": corpus_report["artifacts"]["preferences.jsonl"]["sha256"],
                "replay_rows": str(replay_rows),
                "replay_rows_sha256": sha256_file(replay_rows),
                "model": str(model),
                "parent_adapter": str(parent_adapter),
                "parent_adapter_sha256": sha256_file(parent_adapter / "adapter_model.safetensors"),
            },
            "counts": {
                "rows": row_count,
                "preference_pairs": len(preference_rows) // 2,
                "preference_rows": len(preference_rows),
                "replay_rows": len(replay),
                "replay_actions": dict(sorted(Counter(row["action"] for row in replay).items())),
                "optimizer_steps": steps,
                "rows_per_step": dict(sorted(step_counts.items())),
                "unpacked_tokens": sum(encoded_counts),
                "target_tokens": sum(target_counts),
                "min_row_tokens": min(encoded_counts),
                "max_row_tokens": max(encoded_counts),
            },
            "objective": {
                "chosen_advantage": "+reward_margin",
                "rejected_advantage": "-reward_margin",
                "preference_mask": ["worker_id", "access"],
                "replay_mask": ["action", "target_position_id", "worker_id", "access"],
                "subtask_tokens_directly_optimized": False,
                "adv_tau": 0.1,
                "replay_tau": 0.05,
                "kl_tau": 0.001,
            },
            "artifacts": {
                "rows.jsonl": {"sha256": rows_hash, "rows": row_count},
                "train.yaml": {"sha256": sha256_file(config_path)},
            },
            "training_output": str(training_output),
            "promotion_requires_frozen_local_generation_gate": True,
        }
        (output_dir / "build_report.json").write_text(
            json.dumps(report, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
            encoding="utf-8",
        )
    except BaseException:
        shutil.rmtree(output_dir, ignore_errors=True)
        raise
    return report


def _score_reference(
    trainer: GRPOTrainer,
    prompt_ids: list[int],
    completion_ids: list[int],
    completion_mask: list[bool],
) -> list[float]:
    logical_ids = prompt_ids + completion_ids
    if len(logical_ids) > SEQUENCE_LEN:
        raise StructuralTrainingError("native reference sample exceeds sequence length")
    input_ids = np.zeros((1, SEQUENCE_LEN), dtype=np.int32)
    input_ids[0, : len(logical_ids)] = logical_ids
    targets = np.full((1, SEQUENCE_LEN), -100, dtype=np.int32)
    prompt_len = len(prompt_ids)
    for offset, selected in enumerate(completion_mask):
        logical_index = prompt_len + offset
        if selected:
            targets[0, logical_index - 1] = input_ids[0, logical_index]
    position_ids = np.arange(SEQUENCE_LEN, dtype=np.int32).reshape(1, -1)
    temperatures = np.ones(input_ids.shape, dtype=np.float32)
    scored = np.asarray(
        trainer.trainer.forward_for_grpo(
            input_ids,
            targets,
            position_ids,
            temperatures,
        )
    ).reshape(-1)
    # Ornith's hybrid blocks require the ordinary saved-activation path. A zero
    # backward closes that native micro-step without changing any weight.
    trainer.trainer.backward_grpo(np.zeros((1, SEQUENCE_LEN), dtype=np.float32))
    result = [0.0] * len(completion_ids)
    for offset, selected in enumerate(completion_mask):
        if selected:
            value = float(scored[prompt_len + offset - 1])
            if not math.isfinite(value) or value > 1.0e-6:
                raise StructuralTrainingError(
                    "native parent returned an invalid log-probability "
                    f"at completion offset {offset}, logical index {prompt_len + offset}, "
                    f"token {completion_ids[offset]}: {value}"
                )
            result[offset] = value
    return result


def _pad_sample_to_sequence(sample: TrainingSample) -> None:
    padding = SEQUENCE_LEN - len(sample.prompt_ids) - len(sample.completion_ids)
    if padding < 0:
        raise StructuralTrainingError("training sample exceeds the configured sequence")
    if padding == 0:
        return
    sample.completion_ids.extend([0] * padding)
    sample.completion_mask.extend([False] * padding)
    sample.completion_logprobs.extend([0.0] * padding)
    sample.completion_temperatures.extend([1.0] * padding)
    if sample.replay_mask is not None:
        sample.replay_mask.extend([False] * padding)


def _control_config(*, run_dir: Path, model: str, steps: int, max_batch: int) -> dict[str, Any]:
    return {
        "batch_size": max_batch + max_batch % 2,
        "env": [],
        "max_async_level": 0,
        "max_steps": steps,
        "model": {"lora_adapter": "ornith-contrastive-parent", "name": model},
        "num_train_workers": 1,
        "output_dir": str(run_dir),
        "rollout_transport": {"type": "filesystem"},
        "rollouts_per_example": 2,
        "sequence_len": SEQUENCE_LEN,
        "strict_async_level": True,
        "verification": {"enabled": True},
        "weight_broadcast": {"type": "filesystem"},
    }


def prepare_native_batches(
    *, stage_dir: Path, schema_version: str = SCHEMA_VERSION
) -> dict[str, Any]:
    stage_dir = stage_dir.expanduser().resolve()
    build_report = _read_json(stage_dir / "build_report.json")
    if build_report.get("schema_version") != schema_version:
        raise StructuralTrainingError("stage build report has the wrong schema")
    if sha256_file(stage_dir / "rows.jsonl") != build_report["artifacts"]["rows.jsonl"]["sha256"]:
        raise StructuralTrainingError("stage rows changed after build")
    config_path = stage_dir / "train.yaml"
    if sha256_file(config_path) != build_report["artifacts"]["train.yaml"]["sha256"]:
        raise StructuralTrainingError("training config changed after build")
    raw_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    output_dir = Path(raw_config["output_dir"])
    if output_dir.exists():
        raise StructuralTrainingError(f"training output already exists: {output_dir}")
    model = str(raw_config["model"])
    tokenizer = AutoTokenizer.from_pretrained(
        model, local_files_only=True, trust_remote_code=False
    )
    rows = _read_jsonl(stage_dir / "rows.jsonl")
    by_step: dict[int, list[TrainingSample]] = defaultdict(list)
    rows_by_step: dict[int, list[dict[str, Any]]] = defaultdict(list)
    score_records: list[dict[str, Any]] = []

    index_by_row_id = {str(row["row_id"]): index for index, row in enumerate(rows)}
    for row in rows:
        prompt_ids, completion_ids, completion_mask = _encode_row(row, tokenizer)
        full_replay_mask = None
        if is_replay_mask_mode(row.get("mask_mode")):
            full_replay_mask = [False] * len(prompt_ids) + completion_mask
        sample = TrainingSample(
            prompt_ids=prompt_ids,
            prompt_mask=[False] * len(prompt_ids),
            completion_ids=completion_ids,
            completion_mask=completion_mask,
            completion_logprobs=[0.0] * len(completion_ids),
            completion_temperatures=[1.0] * len(completion_ids),
            advantage=float(row["advantage"]),
            reward=float(row["reward"]),
            replay_mask=full_replay_mask,
        )
        _pad_sample_to_sequence(sample)
        prepare_sample(sample, seq_len=SEQUENCE_LEN)
        step = int(row["step"])
        by_step[step].append(sample)
        rows_by_step[step].append(row)

    steps = int(raw_config["max_steps"])
    if set(by_step) != set(range(steps)):
        raise StructuralTrainingError("native batches do not cover every configured step")

    config = load_config(GRPOTrainConfig, str(config_path))
    trainer = GRPOTrainer(config)
    try:
        for step in range(steps):
            trainer.trainer.set_grad_accumulation(len(by_step[step]))
            for row, sample in zip(rows_by_step[step], by_step[step]):
                reference = _score_reference(
                    trainer,
                    sample.prompt_ids,
                    sample.completion_ids,
                    sample.completion_mask,
                )
                sample.completion_logprobs = reference
                selected_scores = [
                    value
                    for value, keep in zip(reference, sample.completion_mask)
                    if keep
                ]
                score_records.append(
                    {
                        "row_id": row["row_id"],
                        "index": index_by_row_id[str(row["row_id"])],
                        "step": step,
                        "masked_tokens": len(selected_scores),
                        "mean_parent_logprob": sum(selected_scores) / len(selected_scores),
                        "reference_sha256": hashlib.sha256(
                            np.asarray(reference, dtype=np.float32).tobytes()
                        ).hexdigest(),
                    }
                )
    finally:
        trainer.close()
    run_dir = output_dir / "run_default"
    control_path = run_dir / "control/orch.yaml"
    control_path.parent.mkdir(parents=True, exist_ok=True)
    control_path.write_text(
        yaml.safe_dump(
            _control_config(
                run_dir=run_dir,
                model=model,
                steps=steps,
                max_batch=max(len(samples) for samples in by_step.values()),
            ),
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    sender = FileSystemTrainingBatchSender(run_dir)
    batch_hashes: dict[str, str] = {}
    for step in range(steps):
        sender.send(TrainingBatch(examples=by_step[step], step=step))
        batch_path = run_dir / f"rollouts/step_{step}/rollouts.bin"
        restored = msgspec.msgpack.decode(batch_path.read_bytes(), type=TrainingBatch)
        if len(restored.examples) != len(by_step[step]):
            raise StructuralTrainingError("serialized native batch lost examples")
        batch_hashes[str(step)] = sha256_file(batch_path)

    reference_rows, reference_hash = _write_jsonl(
        stage_dir / "reference_scores.jsonl", score_records
    )
    report = {
        "schema_version": schema_version,
        "status": "native_parent_rescored_isolated_batches_ready",
        "external_calls": 0,
        "paid_calls": 0,
        "fresh_worker_trajectories": 0,
        "parent_adapter_sha256": build_report["inputs"]["parent_adapter_sha256"],
        "rows_sha256": build_report["artifacts"]["rows.jsonl"]["sha256"],
        "train_config_sha256": build_report["artifacts"]["train.yaml"]["sha256"],
        "reference_rows": reference_rows,
        "zero_gradient_reference_microsteps": reference_rows,
        "optimizer_steps_taken_during_scoring": 0,
        "reference_scores_sha256": reference_hash,
        "isolated_reference_rows": reference_rows,
        "isolated_reference_scores_sha256": reference_hash,
        "reference_microstep_reset_per_optimizer_batch": True,
        "one_sample_per_microbatch": True,
        "control_sha256": sha256_file(control_path),
        "batch_sha256": batch_hashes,
        "optimizer_steps_authorized": steps,
        "promotion_authorized": False,
    }
    (stage_dir / "reference_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    return report


def pad_existing_native_batches(*, stage_dir: Path) -> dict[str, Any]:
    stage_dir = stage_dir.expanduser().resolve()
    report_path = stage_dir / "reference_report.json"
    report = _read_json(report_path)
    if report.get("status") != "native_parent_scored_batches_ready":
        raise StructuralTrainingError("native reference batches are not ready for padding")
    raw_config = yaml.safe_load((stage_dir / "train.yaml").read_text(encoding="utf-8"))
    output_dir = Path(raw_config["output_dir"])
    if (output_dir / "final_adapter").exists():
        raise StructuralTrainingError("refusing to rewrite batches after a completed training run")
    run_dir = output_dir / "run_default"
    sender = FileSystemTrainingBatchSender(run_dir)
    new_hashes: dict[str, str] = {}
    sample_count = 0
    for step_text, expected_hash in sorted(
        report["batch_sha256"].items(), key=lambda item: int(item[0])
    ):
        step = int(step_text)
        batch_path = run_dir / f"rollouts/step_{step}/rollouts.bin"
        if sha256_file(batch_path) != expected_hash:
            raise StructuralTrainingError(f"native batch {step} changed before padding")
        batch = msgspec.msgpack.decode(batch_path.read_bytes(), type=TrainingBatch)
        for sample in batch.examples:
            _pad_sample_to_sequence(sample)
            if len(sample.prompt_ids) + len(sample.completion_ids) != SEQUENCE_LEN:
                raise StructuralTrainingError("native sample padding failed")
            sample_count += 1
        sender.send(batch)
        new_hashes[step_text] = sha256_file(batch_path)
    report.update(
        {
            "status": "native_parent_scored_isolated_batches_ready",
            "batch_sha256": new_hashes,
            "training_samples": sample_count,
            "one_sample_per_microbatch": True,
            "hybrid_recurrent_state_isolated": True,
        }
    )
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    return report


def rescore_isolated_native_batches(*, stage_dir: Path) -> dict[str, Any]:
    stage_dir = stage_dir.expanduser().resolve()
    report_path = stage_dir / "reference_report.json"
    report = _read_json(report_path)
    if report.get("status") != "native_parent_scored_isolated_batches_ready":
        raise StructuralTrainingError("isolated native batches are not ready for rescoring")
    config_path = stage_dir / "train.yaml"
    raw_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    output_dir = Path(raw_config["output_dir"])
    if (output_dir / "final_adapter").exists():
        raise StructuralTrainingError("refusing to rescore after a completed training run")
    run_dir = output_dir / "run_default"
    batches: list[TrainingBatch] = []
    for step_text, expected_hash in sorted(
        report["batch_sha256"].items(), key=lambda item: int(item[0])
    ):
        batch_path = run_dir / f"rollouts/step_{step_text}/rollouts.bin"
        if sha256_file(batch_path) != expected_hash:
            raise StructuralTrainingError(f"isolated batch {step_text} changed before rescoring")
        batch = msgspec.msgpack.decode(batch_path.read_bytes(), type=TrainingBatch)
        if any(
            len(sample.prompt_ids) + len(sample.completion_ids) != SEQUENCE_LEN
            for sample in batch.examples
        ):
            raise StructuralTrainingError("rescore requires one padded sample per microbatch")
        batches.append(batch)

    config = load_config(GRPOTrainConfig, str(config_path))
    trainer = GRPOTrainer(config)
    records: list[dict[str, Any]] = []
    try:
        for batch in batches:
            trainer.trainer.set_grad_accumulation(len(batch.examples))
            for sample_index, sample in enumerate(batch.examples):
                logical_ids = sample.prompt_ids + sample.completion_ids
                loss_mask = sample.prompt_mask + sample.completion_mask
                input_ids = np.asarray([logical_ids], dtype=np.int32)
                targets = np.full((1, SEQUENCE_LEN), -100, dtype=np.int32)
                for logical_index, selected in enumerate(loss_mask):
                    if selected:
                        targets[0, logical_index - 1] = input_ids[0, logical_index]
                positions = np.arange(SEQUENCE_LEN, dtype=np.int32).reshape(1, -1)
                temperatures = np.ones((1, SEQUENCE_LEN), dtype=np.float32)
                scored = np.asarray(
                    trainer.trainer.forward_for_grpo(
                        input_ids, targets, positions, temperatures
                    )
                ).reshape(-1)
                trainer.trainer.backward_grpo(
                    np.zeros((1, SEQUENCE_LEN), dtype=np.float32)
                )
                selected_scores: list[float] = []
                for completion_index, selected in enumerate(sample.completion_mask):
                    if not selected:
                        sample.completion_logprobs[completion_index] = 0.0
                        continue
                    logical_index = len(sample.prompt_ids) + completion_index
                    value = float(scored[logical_index - 1])
                    if not math.isfinite(value) or value > 1.0e-6:
                        raise StructuralTrainingError(
                            "isolated native rescore returned an invalid log-probability"
                        )
                    sample.completion_logprobs[completion_index] = value
                    selected_scores.append(value)
                records.append(
                    {
                        "step": batch.step,
                        "sample_index": sample_index,
                        "masked_tokens": len(selected_scores),
                        "mean_parent_logprob": sum(selected_scores) / len(selected_scores),
                        "reference_sha256": hashlib.sha256(
                            np.asarray(sample.completion_logprobs, dtype=np.float32).tobytes()
                        ).hexdigest(),
                    }
                )
    finally:
        trainer.close()

    sender = FileSystemTrainingBatchSender(run_dir)
    hashes: dict[str, str] = {}
    for batch in batches:
        sender.send(batch)
        batch_path = run_dir / f"rollouts/step_{batch.step}/rollouts.bin"
        hashes[str(batch.step)] = sha256_file(batch_path)
    record_count, record_hash = _write_jsonl(
        stage_dir / "isolated_reference_scores.jsonl", records
    )
    report.update(
        {
            "status": "native_parent_rescored_isolated_batches_ready",
            "batch_sha256": hashes,
            "isolated_reference_rows": record_count,
            "isolated_reference_scores_sha256": record_hash,
            "reference_microstep_reset_per_optimizer_batch": True,
            "optimizer_steps_taken_during_isolated_rescoring": 0,
        }
    )
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    return report


def audit_native_alignment(
    *, stage_dir: Path, step: int, schema_version: str = SCHEMA_VERSION
) -> dict[str, Any]:
    stage_dir = stage_dir.expanduser().resolve()
    report = _read_json(stage_dir / "reference_report.json")
    if report.get("status") != "native_parent_rescored_isolated_batches_ready":
        raise StructuralTrainingError("isolated native batches are not ready")
    config_path = stage_dir / "train.yaml"
    raw_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    batch_path = Path(raw_config["output_dir"]) / f"run_default/rollouts/step_{step}/rollouts.bin"
    if sha256_file(batch_path) != report["batch_sha256"].get(str(step)):
        raise StructuralTrainingError("alignment-audit batch hash mismatch")
    batch = msgspec.msgpack.decode(batch_path.read_bytes(), type=TrainingBatch)
    config = load_config(GRPOTrainConfig, str(config_path))
    trainer = GRPOTrainer(config)
    log_ratios: list[float] = []
    probability_differences: list[float] = []
    try:
        trainer.trainer.set_grad_accumulation(len(batch.examples))
        for sample in batch.examples:
            logical_ids = sample.prompt_ids + sample.completion_ids
            loss_mask = sample.prompt_mask + sample.completion_mask
            references = [0.0] * len(sample.prompt_ids) + sample.completion_logprobs
            input_ids = np.asarray([logical_ids], dtype=np.int32)
            targets = np.full((1, SEQUENCE_LEN), -100, dtype=np.int32)
            for logical_index, selected in enumerate(loss_mask):
                if selected:
                    targets[0, logical_index - 1] = input_ids[0, logical_index]
            positions = np.arange(SEQUENCE_LEN, dtype=np.int32).reshape(1, -1)
            temperatures = np.ones((1, SEQUENCE_LEN), dtype=np.float32)
            scored = np.asarray(
                trainer.trainer.forward_for_grpo(
                    input_ids, targets, positions, temperatures
                )
            ).reshape(-1)
            trainer.trainer.backward_grpo(
                np.zeros((1, SEQUENCE_LEN), dtype=np.float32)
            )
            for logical_index, selected in enumerate(loss_mask):
                if not selected:
                    continue
                trainer_logprob = float(scored[logical_index - 1])
                reference = float(references[logical_index])
                log_ratios.append(trainer_logprob - reference)
                probability_differences.append(
                    math.exp(trainer_logprob) - math.exp(reference)
                )
    finally:
        trainer.close()
    mismatch = [math.exp(value) - value - 1.0 for value in log_ratios]
    mean_abs_delta = sum(abs(value) for value in log_ratios) / len(log_ratios)
    mean_mismatch_kl = sum(mismatch) / len(mismatch)
    ipo_masked_fraction = sum(
        abs(value) > 0.2 for value in probability_differences
    ) / len(probability_differences)
    aligned = (
        mean_abs_delta <= 0.04
        and mean_mismatch_kl <= 0.002
        and ipo_masked_fraction == 0.0
    )
    result = {
        "schema_version": schema_version,
        "status": (
            "native_parent_alignment_pass"
            if aligned
            else "native_parent_alignment_fail"
        ),
        "step": step,
        "samples": len(batch.examples),
        "masked_tokens": len(log_ratios),
        "max_abs_logprob_delta": max(abs(value) for value in log_ratios),
        "mean_abs_logprob_delta": mean_abs_delta,
        "mean_mismatch_kl": mean_mismatch_kl,
        "ipo_masked_fraction": ipo_masked_fraction,
        "alignment_thresholds": {
            "mean_abs_logprob_delta_max": 0.04,
            "mean_mismatch_kl_max": 0.002,
            "ipo_masked_fraction_max": 0.0,
        },
        "optimizer_steps": 0,
        "external_calls": 0,
        "paid_calls": 0,
        "batch_sha256": sha256_file(batch_path),
        "parent_adapter_sha256": report["parent_adapter_sha256"],
    }
    audit_path = stage_dir / f"alignment_step_{step}.json"
    audit_path.write_text(
        json.dumps(result, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    return result


def diagnose_native_gradients(
    *, stage_dir: Path, step: int, adapter_path: Path | None
) -> dict[str, Any]:
    """Measure each sample's gradient contribution without changing weights."""

    stage_dir = stage_dir.expanduser().resolve()
    report = _read_json(stage_dir / "reference_report.json")
    if report.get("status") != "native_parent_rescored_isolated_batches_ready":
        raise StructuralTrainingError("isolated native batches are not ready")
    config_path = stage_dir / "train.yaml"
    raw_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    output_dir = Path(raw_config["output_dir"])
    batch_path = output_dir / f"run_default/rollouts/step_{step}/rollouts.bin"
    if sha256_file(batch_path) != report["batch_sha256"].get(str(step)):
        raise StructuralTrainingError("gradient-diagnostic batch hash mismatch")
    batch = msgspec.msgpack.decode(batch_path.read_bytes(), type=TrainingBatch)
    rows = [row for row in _read_jsonl(stage_dir / "rows.jsonl") if row["step"] == step]
    if len(rows) != len(batch.examples):
        raise StructuralTrainingError("gradient-diagnostic row/sample count mismatch")

    config = load_config(GRPOTrainConfig, str(config_path))
    if adapter_path is not None:
        adapter_path = adapter_path.expanduser().resolve()
        if not (adapter_path / "adapter_model.safetensors").is_file():
            raise StructuralTrainingError(f"diagnostic adapter is incomplete: {adapter_path}")
        config.adapter_path = str(adapter_path)

    loss_scale = sum(
        sum(sample.prompt_mask) + sum(sample.completion_mask)
        for sample in batch.examples
    )
    trainer = GRPOTrainer(config)
    records: list[dict[str, Any]] = []
    try:
        for sample_index, (row, sample) in enumerate(zip(rows, batch.examples)):
            mb = microbatch_to_numpy(prepare_sample(sample, seq_len=SEQUENCE_LEN))
            input_ids = mb["input_ids"]
            targets = mb["targets"].copy()
            loss_mask = mb["loss_mask"].flatten()
            shifted_mask = np.zeros_like(loss_mask)
            shifted_mask[:-1] = loss_mask[1:]
            targets[0, ~shifted_mask] = -100
            positions = mb["position_ids"]
            temperatures = mb["temperatures"]
            if config.gpus > 1:
                input_ids = np.tile(input_ids, (config.gpus, 1))
                targets = np.tile(targets, (config.gpus, 1))
                positions = np.tile(positions, (config.gpus, 1))
                temperatures = np.tile(temperatures, (config.gpus, 1))

            inference_logprobs = mb["inference_logprobs"].flatten()
            advantages = mb["advantages"].flatten()
            loss_mask_u8 = loss_mask.astype(np.uint8)
            sample_starts = np.asarray([0], dtype=np.int32)
            sample_ends = np.asarray([SEQUENCE_LEN], dtype=np.int32)
            teacher_logprobs = (
                None
                if mb["teacher_logprobs"] is None
                else mb["teacher_logprobs"].flatten()
            )
            opd_reference_logprobs = mb["opd_reference_logprobs"].flatten()
            hindsight_logprobs = mb["hindsight_logprobs"].flatten()
            hindsight_mask = mb["hindsight_mask"].flatten().astype(np.uint8)
            replay_mask = mb["replay_mask"].flatten().astype(np.uint8)

            # Ornith's recurrent QLoRA path is compiled for the production
            # optimizer batch width. Keep that width even though this audit
            # advances after one measured microstep.
            trainer.trainer.set_grad_accumulation(len(batch.examples))
            trainer.trainer.step_grpo_native(
                input_ids,
                targets,
                inference_logprobs,
                advantages,
                loss_mask_u8,
                sample_starts,
                sample_ends,
                position_ids=positions,
                temperatures=temperatures,
                teacher_logprobs=teacher_logprobs,
                opd_reference_logprobs=opd_reference_logprobs,
                hindsight_logprobs=hindsight_logprobs,
                hindsight_mask=hindsight_mask,
                replay_mask=replay_mask,
                loss_scale=float(loss_scale),
                ipo_mask_low=float(config.loss.ipo_mask_low),
                ipo_mask_high=float(config.loss.ipo_mask_high),
                adv_tau=float(config.loss.adv_tau),
                teacher_tau=float(config.loss.teacher_tau),
                opd_tau=float(config.loss.opd_tau),
                opd_beta=float(config.loss.opd_beta),
                replay_tau=float(config.loss.replay_tau),
                kl_tau=float(config.loss.kl_tau),
            )
            result = trainer.trainer.update_with_config(
                _surogate.OptimizerConfig(
                    optimizer=config.optimizer,
                    learning_rate=0.0,
                    weight_decay=0.0,
                    grad_clip=config.max_grad_norm,
                    adamw_beta1=config.adamw_beta1,
                    adamw_beta2=config.adamw_beta2,
                    adamw_epsilon=config.adamw_epsilon,
                ),
                sample_index + 1,
            )
            metrics = dict(trainer.trainer.get_grpo_native_metrics())
            records.append(
                {
                    "sample_index": sample_index,
                    "row_id": row["row_id"],
                    "source": row["source"],
                    "branch_or_action": row.get("branch", row.get("action")),
                    "advantage": row["advantage"],
                    "masked_tokens": int(loss_mask.sum()),
                    "grad_norm": float(result["norm"]),
                    "policy_loss": float(metrics.get("policy_loss", 0.0)),
                    "mismatch_kl": float(metrics.get("mismatch_kl", 0.0)),
                    "ipo_masked_fraction": float(metrics.get("is_masked", 0.0)),
                }
            )
    finally:
        trainer.close()

    result = {
        "schema_version": SCHEMA_VERSION,
        "status": "native_zero_lr_per_sample_gradient_diagnostic",
        "step": step,
        "adapter_path": str(config.adapter_path),
        "samples": len(records),
        "batch_loss_scale": loss_scale,
        "max_grad_norm": max(record["grad_norm"] for record in records),
        "records": records,
        "optimizer_learning_rate": 0.0,
        "weights_changed": False,
        "external_calls": 0,
        "paid_calls": 0,
        "batch_sha256": sha256_file(batch_path),
    }
    diagnostic_path = stage_dir / f"gradient_diagnostic_step_{step}.json"
    diagnostic_path.write_text(
        json.dumps(result, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    return result


def prepare_gradient_source_diagnostic(
    *, stage_dir: Path, output_dir: Path, adapter_path: Path
) -> dict[str, Any]:
    """Prepare a production-path, zero-LR split of the anomalous step."""

    stage_dir = stage_dir.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    adapter_path = adapter_path.expanduser().resolve()
    if output_dir.exists():
        raise StructuralTrainingError(f"diagnostic output already exists: {output_dir}")
    if not (adapter_path / "adapter_model.safetensors").is_file():
        raise StructuralTrainingError(f"diagnostic adapter is incomplete: {adapter_path}")

    reference_report = _read_json(stage_dir / "reference_report.json")
    if reference_report.get("status") != "native_parent_rescored_isolated_batches_ready":
        raise StructuralTrainingError("isolated native batches are not ready")
    source_config_path = stage_dir / "train.yaml"
    source_config = yaml.safe_load(source_config_path.read_text(encoding="utf-8"))
    source_run_dir = Path(source_config["output_dir"]) / "run_default"

    def load_batch(step: int) -> TrainingBatch:
        path = source_run_dir / f"rollouts/step_{step}/rollouts.bin"
        if sha256_file(path) != reference_report["batch_sha256"][str(step)]:
            raise StructuralTrainingError(f"source batch {step} hash mismatch")
        return msgspec.msgpack.decode(path.read_bytes(), type=TrainingBatch)

    warm = load_batch(0)
    anomalous = load_batch(2)
    rows = [row for row in _read_jsonl(stage_dir / "rows.jsonl") if row["step"] == 2]
    if len(rows) != len(anomalous.examples):
        raise StructuralTrainingError("anomalous row/sample count mismatch")

    def masked_copy(source: str, step: int) -> TrainingBatch:
        batch = msgspec.msgpack.decode(
            msgspec.msgpack.encode(anomalous), type=TrainingBatch
        )
        batch.step = step
        for row, sample in zip(rows, batch.examples):
            if row["source"] == source:
                continue
            sample.completion_mask = [False] * len(sample.completion_mask)
            if sample.replay_mask is not None:
                sample.replay_mask = [False] * len(sample.replay_mask)
        return batch

    warm.step = 0
    preference = masked_copy("retained_current_pool_preference", 1)
    replay = masked_copy("accepted_unified_replay", 2)
    batches = (warm, preference, replay)
    selected_tokens = {
        "warm": sum(sum(sample.completion_mask) for sample in warm.examples),
        "preference": sum(
            sum(sample.completion_mask) for sample in preference.examples
        ),
        "replay": sum(sum(sample.completion_mask) for sample in replay.examples),
    }
    if selected_tokens != {"warm": 572, "preference": 552, "replay": 102}:
        raise StructuralTrainingError(
            f"unexpected diagnostic token split: {selected_tokens}"
        )

    diagnostic_config = dict(source_config)
    diagnostic_config.update(
        {
            "adapter_path": str(adapter_path),
            "output_dir": str(output_dir),
            "checkpoint_dir": str(output_dir),
            "surogate_metrics_path": str(output_dir / "surogate_metrics.jsonl"),
            "learning_rate": 0.0,
            "weight_decay": 0.0,
            "max_steps": 3,
            "save_steps": 0,
            "resume_from_checkpoint": False,
        }
    )
    output_dir.mkdir(parents=True)
    diagnostic_config_path = output_dir / "train.yaml"
    diagnostic_config_path.write_text(
        yaml.safe_dump(diagnostic_config, sort_keys=False), encoding="utf-8"
    )
    run_dir = output_dir / "run_default"
    control_path = run_dir / "control/orch.yaml"
    control_path.parent.mkdir(parents=True)
    control_path.write_text(
        yaml.safe_dump(
            _control_config(
                run_dir=run_dir,
                model=str(source_config["model"]),
                steps=3,
                max_batch=max(len(batch.examples) for batch in batches),
            ),
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    sender = FileSystemTrainingBatchSender(run_dir)
    batch_hashes: dict[str, str] = {}
    for batch in batches:
        sender.send(batch)
        batch_path = run_dir / f"rollouts/step_{batch.step}/rollouts.bin"
        batch_hashes[str(batch.step)] = sha256_file(batch_path)

    result = {
        "schema_version": SCHEMA_VERSION,
        "status": "zero_lr_gradient_source_diagnostic_ready",
        "adapter_path": str(adapter_path),
        "adapter_sha256": sha256_file(adapter_path / "adapter_model.safetensors"),
        "source_anomalous_step": 2,
        "diagnostic_steps": {
            "0": "warm_original_step_0",
            "1": "preference_only_original_step_2",
            "2": "replay_only_original_step_2",
        },
        "selected_tokens": selected_tokens,
        "learning_rate": 0.0,
        "weights_change_authorized": False,
        "external_calls": 0,
        "paid_calls": 0,
        "train_config_sha256": sha256_file(diagnostic_config_path),
        "control_sha256": sha256_file(control_path),
        "batch_sha256": batch_hashes,
    }
    report_path = output_dir / "diagnostic_report.json"
    report_path.write_text(
        json.dumps(result, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build")
    build.add_argument("--corpus", type=Path, required=True)
    build.add_argument("--replay-rows", type=Path, required=True)
    build.add_argument("--model", type=Path, required=True)
    build.add_argument("--parent-adapter", type=Path, required=True)
    build.add_argument("--output-dir", type=Path, required=True)
    build.add_argument("--training-output", type=Path, required=True)
    prepare = commands.add_parser("prepare-native")
    prepare.add_argument("--stage-dir", type=Path, required=True)
    pad = commands.add_parser("pad-native")
    pad.add_argument("--stage-dir", type=Path, required=True)
    rescore = commands.add_parser("rescore-native")
    rescore.add_argument("--stage-dir", type=Path, required=True)
    audit = commands.add_parser("audit-native")
    audit.add_argument("--stage-dir", type=Path, required=True)
    audit.add_argument("--step", type=int, default=0)
    diagnose = commands.add_parser("diagnose-native-gradients")
    diagnose.add_argument("--stage-dir", type=Path, required=True)
    diagnose.add_argument("--step", type=int, required=True)
    diagnose.add_argument("--adapter-path", type=Path)
    source_diagnostic = commands.add_parser("prepare-gradient-source-diagnostic")
    source_diagnostic.add_argument("--stage-dir", type=Path, required=True)
    source_diagnostic.add_argument("--output-dir", type=Path, required=True)
    source_diagnostic.add_argument("--adapter-path", type=Path, required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.command == "build":
        report = build_stage(
            corpus=args.corpus,
            replay_rows=args.replay_rows,
            model=args.model,
            parent_adapter=args.parent_adapter,
            output_dir=args.output_dir,
            training_output=args.training_output,
        )
    elif args.command == "prepare-native":
        report = prepare_native_batches(stage_dir=args.stage_dir)
    elif args.command == "pad-native":
        report = pad_existing_native_batches(stage_dir=args.stage_dir)
    elif args.command == "rescore-native":
        report = rescore_isolated_native_batches(stage_dir=args.stage_dir)
    elif args.command == "audit-native":
        report = audit_native_alignment(stage_dir=args.stage_dir, step=args.step)
    elif args.command == "diagnose-native-gradients":
        report = diagnose_native_gradients(
            stage_dir=args.stage_dir,
            step=args.step,
            adapter_path=args.adapter_path,
        )
    else:
        report = prepare_gradient_source_diagnostic(
            stage_dir=args.stage_dir,
            output_dir=args.output_dir,
            adapter_path=args.adapter_path,
        )
    print(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
