"""Build a replay-protected preference continuation for the accepted planner."""

from __future__ import annotations

import argparse
import ast
import json
import math
import re
import shutil
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import msgspec
import yaml
from transformers import AutoTokenizer

from director.agentic.fugu_contrastive_topology_training import (
    PAIRS_PER_STEP,
    _assert_identity_free,
    _control_config,
    _encode_row,
    _load_needed_candidates,
    _score_reference,
    _training_config,
    audit_native_alignment,
    prepare_native_batches,
)
from director.agentic.fugu_structural_outcome_training import (
    StructuralTrainingError,
    _read_json,
    _read_jsonl,
    _verify_distillation_corpus,
    _write_jsonl,
    sha256_file,
)
from surogate.grpo.transport import FileSystemTrainingBatchSender, TrainingBatch
from surogate.core.config.loader import load_config
from surogate.grpo.config import GRPOTrainConfig
from surogate.grpo.trainer import GRPOTrainer


SCHEMA_VERSION = "fugu_planner_contrastive_v2"
SOURCE_SCHEMA_VERSION = "fugu_planner_contrastive_v1"
PREFERENCE_TASK_CHARS = 1_200
ROLE_LINE = re.compile(r"^Model (\d+): roles=(.+)$", re.MULTILINE)
PLANNER_FIELDS = ("model_id", "subtasks", "access_list")


def _clip_middle(text: str, limit: int) -> str:
    text = text.strip()
    if len(text) <= limit:
        return text
    half = limit // 2
    omitted = len(text) - 2 * half
    return f"{text[:half]}\n\n[...{omitted} middle characters omitted...]\n\n{text[-half:]}"


def _rotation_map(rotation: int, worker_count: int = 4) -> dict[int, int]:
    return {worker_id: (worker_id + rotation) % worker_count for worker_id in range(worker_count)}


def _role_lines(profiles: Sequence[Mapping[str, Any]], rotation: int) -> str:
    mapping = _rotation_map(rotation, len(profiles))
    rotated: list[tuple[int, str]] = []
    for profile in profiles:
        worker_id = profile.get("worker_id")
        tags = profile.get("capability_tags")
        if (
            isinstance(worker_id, bool)
            or not isinstance(worker_id, int)
            or not isinstance(tags, list)
            or not all(isinstance(tag, str) and tag for tag in tags)
        ):
            raise StructuralTrainingError("invalid capability profile")
        rotated.append((mapping[worker_id], ", ".join(tags)))
    return "\n".join(
        f"Model {worker_id}: roles={tags}" for worker_id, tags in sorted(rotated)
    )


def _preference_messages(prompt: Mapping[str, Any], rotation: int) -> list[dict[str, str]]:
    source_messages = prompt.get("messages")
    profiles = prompt.get("capability_profiles")
    task_text = prompt.get("task_text")
    if (
        not isinstance(source_messages, list)
        or len(source_messages) != 2
        or not isinstance(profiles, list)
        or len(profiles) != 4
        or not isinstance(task_text, str)
    ):
        raise StructuralTrainingError("preference prompt has an invalid planner surface")
    system = source_messages[0].get("content")
    if not isinstance(system, str) or not system:
        raise StructuralTrainingError("preference prompt has no system message")
    user = (
        f"{_clip_middle(task_text, PREFERENCE_TASK_CHARS)}\n\n"
        "AVAILABLE LANGUAGE MODELS:\n"
        f"{_role_lines(profiles, rotation)}"
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _workflow_completion(candidate: Mapping[str, Any], rotation: int) -> str:
    workflow = candidate.get("workflow")
    steps = workflow.get("steps") if isinstance(workflow, Mapping) else None
    if not isinstance(steps, list) or not steps:
        raise StructuralTrainingError("preference candidate has no workflow")
    mapping = _rotation_map(rotation)
    model_ids: list[int] = []
    subtasks: list[str] = []
    access_list: list[list[int]] = []
    for position, step in enumerate(steps):
        if not isinstance(step, Mapping) or step.get("position_id") != position:
            raise StructuralTrainingError("preference workflow positions are invalid")
        worker_id = step.get("worker_id")
        subtask = step.get("subtask")
        access = step.get("access")
        if (
            isinstance(worker_id, bool)
            or worker_id not in mapping
            or not isinstance(subtask, str)
            or not subtask.strip()
            or not isinstance(access, list)
            or not all(isinstance(item, int) and not isinstance(item, bool) for item in access)
        ):
            raise StructuralTrainingError("preference workflow step is invalid")
        model_ids.append(mapping[int(worker_id)])
        subtasks.append(subtask.strip())
        access_list.append(list(access))
    return _planner_completion(model_ids, subtasks, access_list)


def _planner_completion(
    model_ids: Sequence[int], subtasks: Sequence[str], access_list: Sequence[Sequence[int]]
) -> str:
    return "\n".join(
        (
            f"model_id = {json.dumps(list(model_ids), ensure_ascii=True)}",
            f"subtasks = {json.dumps(list(subtasks), ensure_ascii=True)}",
            f"access_list = {json.dumps([list(items) for items in access_list], ensure_ascii=True)}",
        )
    )


def _parse_planner_completion(content: str) -> tuple[list[int], list[str], list[list[int]]]:
    parsed: dict[str, Any] = {}
    for name in PLANNER_FIELDS:
        match = re.search(rf"(?m)^{name}\s*=\s*(.+)$", content)
        if match is None:
            raise StructuralTrainingError(f"planner replay is missing {name}")
        try:
            parsed[name] = ast.literal_eval(match.group(1))
        except (SyntaxError, ValueError) as exc:
            raise StructuralTrainingError(f"planner replay has invalid {name}") from exc
    model_ids = parsed["model_id"]
    subtasks = parsed["subtasks"]
    access_list = parsed["access_list"]
    if (
        not isinstance(model_ids, list)
        or not model_ids
        or not all(isinstance(item, int) and not isinstance(item, bool) for item in model_ids)
        or not isinstance(subtasks, list)
        or not all(isinstance(item, str) and item for item in subtasks)
        or not isinstance(access_list, list)
        or not all(
            isinstance(items, list)
            and all(isinstance(item, int) and not isinstance(item, bool) for item in items)
            for items in access_list
        )
        or not len(model_ids) == len(subtasks) == len(access_list)
    ):
        raise StructuralTrainingError("planner replay lists are invalid")
    return model_ids, subtasks, access_list


def _rotate_replay_messages(
    messages: Sequence[Mapping[str, Any]], rotation: int
) -> list[dict[str, str]]:
    if len(messages) != 3:
        raise StructuralTrainingError("planner replay must contain three messages")
    system = messages[0].get("content")
    user = messages[1].get("content")
    completion = messages[2].get("content")
    if not all(isinstance(value, str) and value for value in (system, user, completion)):
        raise StructuralTrainingError("planner replay message content is invalid")
    matches = ROLE_LINE.findall(user)
    if len(matches) != 4:
        raise StructuralTrainingError("planner replay does not expose four role profiles")
    mapping = _rotation_map(rotation)
    roles = sorted((mapping[int(worker_id)], tags) for worker_id, tags in matches)
    marker = "AVAILABLE LANGUAGE MODELS:\n"
    if marker not in user:
        raise StructuralTrainingError("planner replay has no role-profile marker")
    user_prefix = user.split(marker, 1)[0]
    rotated_user = user_prefix + marker + "\n".join(
        f"Model {worker_id}: roles={tags}" for worker_id, tags in roles
    )
    model_ids, subtasks, access_list = _parse_planner_completion(completion)
    if any(worker_id not in mapping for worker_id in model_ids):
        raise StructuralTrainingError("planner replay selects an unknown worker")
    rotated_completion = _planner_completion(
        [mapping[worker_id] for worker_id in model_ids], subtasks, access_list
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": rotated_user},
        {"role": "assistant", "content": rotated_completion},
    ]


def _preference_rows(corpus: Path, *, split: str = "train") -> list[dict[str, Any]]:
    prompts = {row["prompt_id"]: row for row in _read_jsonl(corpus / "prompts.jsonl")}
    preferences = sorted(
        (
            row
            for row in _read_jsonl(corpus / "preferences.jsonl")
            if row.get("pool_epoch") == "current_pool" and row.get("split") == split
        ),
        key=lambda row: str(row["preference_id"]),
    )
    expected_preferences = {"train": 16, "holdout": 8}.get(split)
    if expected_preferences is None:
        raise StructuralTrainingError(f"unsupported preference split: {split}")
    if len(preferences) != expected_preferences:
        raise StructuralTrainingError(
            f"expected {expected_preferences} current-pool {split} preferences, "
            f"found {len(preferences)}"
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
        margin = float(preference["reward_margin"])
        for rotation in range(4):
            prompt_messages = _preference_messages(prompt, rotation)
            pair_id = f"{preference['preference_id']}::rotation-{rotation}"
            for branch, sign, candidate_key, reward_key in (
                ("chosen", 1.0, "chosen_candidate_id", "chosen_topology_mean_reward"),
                ("rejected", -1.0, "rejected_candidate_id", "rejected_topology_mean_reward"),
            ):
                content = _workflow_completion(candidates[str(preference[candidate_key])], rotation)
                row_id = f"{pair_id}::{branch}"
                messages = [*prompt_messages, {"role": "assistant", "content": content}]
                _assert_identity_free(messages, row_id)
                rows.append(
                    {
                        "row_id": row_id,
                        "source": "retained_current_pool_planner_preference",
                        "mask_mode": "planner_topology",
                        "step": pair_index // PAIRS_PER_STEP,
                        "pair_id": pair_id,
                        "preference_id": preference["preference_id"],
                        "prompt_id": preference["prompt_id"],
                        "rotation": rotation,
                        "branch": branch,
                        "advantage": sign * margin,
                        "reward": float(preference[reward_key]),
                        "reward_margin": margin,
                        "messages": messages,
                    }
                )
            pair_index += 1
    expected_pairs = expected_preferences * 4
    if pair_index != expected_pairs:
        raise StructuralTrainingError(
            f"planner preference pair count is not {expected_pairs}"
        )
    return rows


def _replay_rows(path: Path, *, steps: int) -> list[dict[str, Any]]:
    source_rows = _read_jsonl(path)
    if len(source_rows) != 40 or any(row.get("held_out_validation") for row in source_rows):
        raise StructuralTrainingError("accepted planner replay must contain 40 held-in rows")
    rows: list[dict[str, Any]] = []
    for source_index, source in enumerate(source_rows):
        messages = source.get("messages")
        if not isinstance(messages, list):
            raise StructuralTrainingError("planner replay has no messages")
        for rotation in range(4):
            row_id = f"{source['record_id']}::rotation-{rotation}"
            rotated = _rotate_replay_messages(messages, rotation)
            _assert_identity_free(rotated, row_id)
            rows.append(
                {
                    "row_id": row_id,
                    "source": "accepted_planner_v11_replay",
                    "mask_mode": "planner_replay",
                    "step": (source_index * 4 + rotation) % steps,
                    "family": source.get("family"),
                    "rotation": rotation,
                    "advantage": 0.0,
                    "reward": 0.0,
                    "messages": rotated,
                }
            )
    return rows


def build_stage(
    *,
    corpus: Path,
    planner_train: Path,
    planner_manifest: Path,
    model: Path,
    parent_adapter: Path,
    output_dir: Path,
    training_output: Path,
) -> dict[str, Any]:
    paths = [corpus, planner_train, planner_manifest, model, parent_adapter, output_dir, training_output]
    corpus, planner_train, planner_manifest, model, parent_adapter, output_dir, training_output = (
        path.expanduser().resolve() for path in paths
    )
    if output_dir.exists() or training_output.exists():
        raise StructuralTrainingError("stage or training output already exists")
    corpus_report = _verify_distillation_corpus(corpus)
    manifest = _read_json(planner_manifest)
    expected_train_hash = ((manifest.get("artifacts") or {}).get("train_sha256"))
    if expected_train_hash != sha256_file(planner_train):
        raise StructuralTrainingError("accepted planner replay hash does not match its manifest")
    for name in ("adapter_config.json", "adapter_model.safetensors"):
        if not (parent_adapter / name).is_file():
            raise StructuralTrainingError(f"parent adapter is missing {name}")

    tokenizer = AutoTokenizer.from_pretrained(str(model), local_files_only=True)
    preference = _preference_rows(corpus)
    steps = math.ceil((len(preference) // 2) / PAIRS_PER_STEP)
    replay = _replay_rows(planner_train, steps=steps)
    rows = [*preference, *replay]
    token_counts: list[int] = []
    target_counts: list[int] = []
    for row in rows:
        prompt_ids, completion_ids, mask = _encode_row(row, tokenizer)
        token_counts.append(len(prompt_ids) + len(completion_ids))
        target_counts.append(sum(mask))
    if Counter(row.get("branch") for row in preference) != {"chosen": 64, "rejected": 64}:
        raise StructuralTrainingError("planner preference branches are not symmetric")
    step_counts = Counter(int(row["step"]) for row in rows)
    if set(step_counts) != set(range(steps)):
        raise StructuralTrainingError("planner training steps are not contiguous")

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
            "inputs": {
                "distillation_preferences_sha256": corpus_report["artifacts"]["preferences.jsonl"]["sha256"],
                "planner_replay": str(planner_train),
                "planner_replay_sha256": expected_train_hash,
                "model": str(model),
                "parent_adapter": str(parent_adapter),
                "parent_adapter_sha256": sha256_file(parent_adapter / "adapter_model.safetensors"),
            },
            "counts": {
                "rows": row_count,
                "preference_pairs": len(preference) // 2,
                "preference_rows": len(preference),
                "replay_rows": len(replay),
                "optimizer_steps": steps,
                "rows_per_step": dict(sorted(step_counts.items())),
                "unpacked_tokens": sum(token_counts),
                "target_tokens": sum(target_counts),
                "min_row_tokens": min(token_counts),
                "max_row_tokens": max(token_counts),
            },
            "objective": {
                "preference_mask": ["model_id", "access_list"],
                "replay_mask": ["model_id", "access_list"],
                "chosen_advantage": "+reward_margin",
                "rejected_advantage": "-reward_margin",
                "subtask_tokens_directly_optimized": False,
                "adv_tau": 0.1,
                "replay_tau": 0.05,
                "kl_tau": 0.001,
            },
            "holdouts": {
                "current_pool_preference_groups": 8,
                "planner_v11_agentic_tasks": 8,
            },
            "evidence_scope": "capability_routing_only_not_agentic_coordination_lift",
            "artifacts": {
                "rows.jsonl": {"rows": row_count, "sha256": rows_hash},
                "train.yaml": {"sha256": sha256_file(config_path)},
            },
            "training_output": str(training_output),
            "promotion_requires_local_preference_and_agentic_generation_gate": True,
        }
        (output_dir / "build_report.json").write_text(
            json.dumps(report, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
            encoding="utf-8",
        )
    except BaseException:
        shutil.rmtree(output_dir, ignore_errors=True)
        raise
    return report


def reuse_parent_scores(*, source_stage: Path, stage_dir: Path) -> dict[str, Any]:
    """Reuse exact parent scores while correcting only planner replay masks."""

    source_stage = source_stage.expanduser().resolve()
    stage_dir = stage_dir.expanduser().resolve()
    source_build = _read_json(source_stage / "build_report.json")
    source_reference = _read_json(source_stage / "reference_report.json")
    target_build = _read_json(stage_dir / "build_report.json")
    if source_build.get("schema_version") != SOURCE_SCHEMA_VERSION:
        raise StructuralTrainingError("reference source has the wrong schema")
    if target_build.get("schema_version") != SCHEMA_VERSION:
        raise StructuralTrainingError("V2 stage has the wrong schema")
    if source_reference.get("status") != "native_parent_rescored_isolated_batches_ready":
        raise StructuralTrainingError("source parent scores are not ready")
    if (
        source_build["artifacts"]["rows.jsonl"]["sha256"]
        != target_build["artifacts"]["rows.jsonl"]["sha256"]
    ):
        raise StructuralTrainingError("V2 rows differ from the scored V1 rows")
    if source_reference["parent_adapter_sha256"] != target_build["inputs"]["parent_adapter_sha256"]:
        raise StructuralTrainingError("V2 parent differs from the scored V1 parent")

    source_config = yaml.safe_load((source_stage / "train.yaml").read_text(encoding="utf-8"))
    target_config = yaml.safe_load((stage_dir / "train.yaml").read_text(encoding="utf-8"))
    target_output = Path(target_config["output_dir"])
    if target_output.exists():
        raise StructuralTrainingError(f"V2 training output already exists: {target_output}")
    source_run = Path(source_config["output_dir"]) / "run_default"
    target_run = target_output / "run_default"
    rows_by_step: dict[int, list[dict[str, Any]]] = {
        step: [] for step in range(int(target_config["max_steps"]))
    }
    for row in _read_jsonl(stage_dir / "rows.jsonl"):
        rows_by_step[int(row["step"])].append(row)

    control_path = target_run / "control/orch.yaml"
    control_path.parent.mkdir(parents=True, exist_ok=True)
    control_path.write_text(
        yaml.safe_dump(
            _control_config(
                run_dir=target_run,
                model=str(target_config["model"]),
                steps=int(target_config["max_steps"]),
                max_batch=max(len(rows) for rows in rows_by_step.values()),
            ),
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    sender = FileSystemTrainingBatchSender(target_run)
    batch_hashes: dict[str, str] = {}
    replay_samples = 0
    replay_tokens = 0
    for step in range(int(target_config["max_steps"])):
        source_path = source_run / f"rollouts/step_{step}/rollouts.bin"
        if sha256_file(source_path) != source_reference["batch_sha256"][str(step)]:
            raise StructuralTrainingError(f"source reference batch {step} changed")
        batch = msgspec.msgpack.decode(source_path.read_bytes(), type=TrainingBatch)
        step_rows = rows_by_step[step]
        if len(batch.examples) != len(step_rows):
            raise StructuralTrainingError("reference batch row order is inconsistent")
        for row, sample in zip(step_rows, batch.examples):
            if row.get("mask_mode") == "planner_replay":
                sample.replay_mask = [False] * len(sample.prompt_ids) + list(sample.completion_mask)
                replay_samples += 1
                replay_tokens += sum(sample.completion_mask)
            else:
                sample.replay_mask = None
        sender.send(batch)
        target_path = target_run / f"rollouts/step_{step}/rollouts.bin"
        batch_hashes[str(step)] = sha256_file(target_path)

    source_scores = source_stage / "reference_scores.jsonl"
    if sha256_file(source_scores) != source_reference["reference_scores_sha256"]:
        raise StructuralTrainingError("source reference score file changed")
    shutil.copy2(source_scores, stage_dir / "reference_scores.jsonl")
    report = {
        "schema_version": SCHEMA_VERSION,
        "status": "native_parent_rescored_isolated_batches_ready",
        "external_calls": 0,
        "paid_calls": 0,
        "optimizer_steps_taken_during_scoring": 0,
        "optimizer_steps_authorized": int(target_config["max_steps"]),
        "parent_adapter_sha256": target_build["inputs"]["parent_adapter_sha256"],
        "rows_sha256": target_build["artifacts"]["rows.jsonl"]["sha256"],
        "train_config_sha256": target_build["artifacts"]["train.yaml"]["sha256"],
        "reference_rows": source_reference["reference_rows"],
        "reference_scores_sha256": source_reference["reference_scores_sha256"],
        "reference_reused_from": str(source_stage),
        "reference_reuse_reason": "identical_parent_tokens_and_masks_replay_bit_only",
        "replay_samples": replay_samples,
        "replay_masked_tokens": replay_tokens,
        "control_sha256": sha256_file(control_path),
        "batch_sha256": batch_hashes,
        "promotion_authorized": False,
    }
    (stage_dir / "reference_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    return report


def _native_adapter_scores(
    *,
    rows: Sequence[Mapping[str, Any]],
    tokenizer: Any,
    config_path: Path,
    adapter: Path,
    metrics_path: Path,
) -> dict[str, float]:
    config = load_config(GRPOTrainConfig, str(config_path))
    config.adapter_path = str(adapter.expanduser().resolve())
    config.surogate_metrics_path = str(metrics_path)
    trainer = GRPOTrainer(config)
    scores: dict[str, float] = {}
    try:
        trainer.trainer.set_grad_accumulation(len(rows))
        for row in rows:
            prompt_ids, completion_ids, completion_mask = _encode_row(row, tokenizer)
            reference = _score_reference(
                trainer, prompt_ids, completion_ids, completion_mask
            )
            selected = [
                score for score, keep in zip(reference, completion_mask) if keep
            ]
            if not selected:
                raise StructuralTrainingError("holdout row has no scored tokens")
            scores[str(row["row_id"])] = sum(selected) / len(selected)
    finally:
        trainer.close()
    return scores


def gate_holdout_likelihood(
    *, corpus: Path, stage_dir: Path, candidate: Path, report_path: Path
) -> dict[str, Any]:
    corpus = corpus.expanduser().resolve()
    stage_dir = stage_dir.expanduser().resolve()
    candidate = candidate.expanduser().resolve()
    report_path = report_path.expanduser().resolve()
    build = _read_json(stage_dir / "build_report.json")
    reference = _read_json(stage_dir / "reference_report.json")
    if build.get("schema_version") != SCHEMA_VERSION:
        raise StructuralTrainingError("holdout gate stage has the wrong schema")
    if reference.get("status") != "native_parent_rescored_isolated_batches_ready":
        raise StructuralTrainingError("holdout gate parent references are not ready")
    if not (candidate / "adapter_model.safetensors").is_file():
        raise StructuralTrainingError("holdout gate candidate is incomplete")
    raw_config = yaml.safe_load((stage_dir / "train.yaml").read_text(encoding="utf-8"))
    tokenizer = AutoTokenizer.from_pretrained(
        str(raw_config["model"]), local_files_only=True
    )
    rows = _preference_rows(corpus, split="holdout")
    parent = Path(raw_config["adapter_path"])
    parent_scores = _native_adapter_scores(
        rows=rows,
        tokenizer=tokenizer,
        config_path=stage_dir / "train.yaml",
        adapter=parent,
        metrics_path=stage_dir / "holdout_parent_metrics.jsonl",
    )
    candidate_scores = _native_adapter_scores(
        rows=rows,
        tokenizer=tokenizer,
        config_path=stage_dir / "train.yaml",
        adapter=candidate,
        metrics_path=stage_dir / "holdout_candidate_metrics.jsonl",
    )

    by_pair: dict[str, dict[str, dict[str, float]]] = {}
    for row in rows:
        pair = by_pair.setdefault(str(row["pair_id"]), {})
        pair[str(row["branch"])] = {
            "parent": parent_scores[str(row["row_id"])],
            "candidate": candidate_scores[str(row["row_id"])],
        }
    results: list[dict[str, Any]] = []
    for pair_id, branches in sorted(by_pair.items()):
        parent_delta = branches["chosen"]["parent"] - branches["rejected"]["parent"]
        candidate_delta = branches["chosen"]["candidate"] - branches["rejected"]["candidate"]
        results.append(
            {
                "pair_id": pair_id,
                "parent_delta": parent_delta,
                "candidate_delta": candidate_delta,
                "delta_change": candidate_delta - parent_delta,
            }
        )
    parent_wins = sum(row["parent_delta"] > 0 for row in results)
    candidate_wins = sum(row["candidate_delta"] > 0 for row in results)
    parent_mean = sum(row["parent_delta"] for row in results) / len(results)
    candidate_mean = sum(row["candidate_delta"] for row in results) / len(results)
    passed = candidate_wins >= parent_wins and candidate_mean > parent_mean
    report = {
        "schema_version": "fugu_planner_contrastive_holdout_gate_v1",
        "status": "pass" if passed else "fail",
        "external_calls": 0,
        "paid_calls": 0,
        "optimizer_steps": 0,
        "holdout_prompt_groups": 8,
        "rotated_pairs": len(results),
        "parent": {
            "adapter_sha256": sha256_file(parent / "adapter_model.safetensors"),
            "chosen_wins": parent_wins,
            "mean_chosen_minus_rejected_logprob": parent_mean,
        },
        "candidate": {
            "adapter_sha256": sha256_file(candidate / "adapter_model.safetensors"),
            "chosen_wins": candidate_wins,
            "mean_chosen_minus_rejected_logprob": candidate_mean,
        },
        "gate": {
            "no_chosen_win_regression": candidate_wins >= parent_wins,
            "strict_mean_pairwise_likelihood_improvement": candidate_mean > parent_mean,
            "passed": passed,
        },
        "pairs": results,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build")
    build.add_argument("--corpus", type=Path, required=True)
    build.add_argument("--planner-train", type=Path, required=True)
    build.add_argument("--planner-manifest", type=Path, required=True)
    build.add_argument("--model", type=Path, required=True)
    build.add_argument("--parent-adapter", type=Path, required=True)
    build.add_argument("--output-dir", type=Path, required=True)
    build.add_argument("--training-output", type=Path, required=True)
    prepare = commands.add_parser("prepare-native")
    prepare.add_argument("--stage-dir", type=Path, required=True)
    audit = commands.add_parser("audit-native")
    audit.add_argument("--stage-dir", type=Path, required=True)
    audit.add_argument("--step", type=int, default=0)
    reuse = commands.add_parser("reuse-native")
    reuse.add_argument("--source-stage", type=Path, required=True)
    reuse.add_argument("--stage-dir", type=Path, required=True)
    gate = commands.add_parser("gate-likelihood")
    gate.add_argument("--corpus", type=Path, required=True)
    gate.add_argument("--stage-dir", type=Path, required=True)
    gate.add_argument("--candidate", type=Path, required=True)
    gate.add_argument("--report", type=Path, required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.command == "build":
        report = build_stage(
            corpus=args.corpus,
            planner_train=args.planner_train,
            planner_manifest=args.planner_manifest,
            model=args.model,
            parent_adapter=args.parent_adapter,
            output_dir=args.output_dir,
            training_output=args.training_output,
        )
    elif args.command == "prepare-native":
        report = prepare_native_batches(
            stage_dir=args.stage_dir, schema_version=SCHEMA_VERSION
        )
    elif args.command == "audit-native":
        report = audit_native_alignment(
            stage_dir=args.stage_dir,
            step=args.step,
            schema_version=SCHEMA_VERSION,
        )
    elif args.command == "reuse-native":
        report = reuse_parent_scores(
            source_stage=args.source_stage,
            stage_dir=args.stage_dir,
        )
    else:
        report = gate_holdout_likelihood(
            corpus=args.corpus,
            stage_dir=args.stage_dir,
            candidate=args.candidate,
            report_path=args.report,
        )
    print(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
