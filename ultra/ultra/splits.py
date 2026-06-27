"""Split + dedup helpers (ultra-data2 §3; validation gate "Deduplication").

Prompt hashes alone are insufficient (ultra-data2 §9) — real dedup keys off the
``contamination_group`` (repo / source family / template / time window) when present,
falling back to a normalized prompt hash.
"""

from __future__ import annotations

import hashlib

from .schemas import TaskSpec


def normalized_prompt_hash(task: TaskSpec) -> str:
    text = " ".join(str(m.get("content", "")) for m in task.input.messages)
    text = " ".join(text.split()).lower()
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


def dedup_key(task: TaskSpec) -> str:
    return task.splitting.contamination_group or normalized_prompt_hash(task)
