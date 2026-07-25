#!/usr/bin/env python3
"""Collect direct one-call synthetic branchpoints from the accepted policy."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from urllib.request import urlopen


REPO_ROOT = Path(__file__).resolve().parents[1]
for _source_root in (
    REPO_ROOT,
    REPO_ROOT / "director",
    REPO_ROOT / "ultra",
):
    if str(_source_root) not in sys.path:
        sys.path.insert(0, str(_source_root))

from openai import AsyncOpenAI  # noqa: E402

from director.agentic.fugu_ultra_terminal import (  # noqa: E402
    LocalModelPromptTokenCounter,
    PRODUCT_RUNTIME_REVISION,
)
from ultra.live_control import OpenAILiveController  # noqa: E402
from ultra.pool_binding import load_pool_binding  # noqa: E402
from ultra.synthetic_branchpoint_collection import (  # noqa: E402
    collect_synthetic_branchpoints,
)


MAX_INPUT_TOKENS = 7_680
MAX_OUTPUT_TOKENS = 512


def _read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"{path} must contain one JSON object")
    return value


def _bound_base_matches(served_root: Path, bound_snapshot: str) -> bool:
    configured = Path(bound_snapshot).expanduser()
    if configured.exists():
        return served_root == configured.resolve()
    # Hugging Face snapshots are normally served from a cache directory such
    # as models--ORG--REPOSITORY/snapshots/REVISION.
    repository_marker = "models--" + bound_snapshot.strip("/").replace("/", "--")
    return repository_marker in served_root.parts


def _assert_served_checkpoint(
    *,
    models_url: str,
    model: str,
    base_model: str,
    adapter_path: Path,
    bound_base_snapshot: str,
) -> None:
    with urlopen(models_url, timeout=10) as response:  # noqa: S310 - local only
        payload = json.load(response)
    rows = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(rows, list):
        raise RuntimeError("local model catalog has no data list")
    by_id = {
        row["id"]: row
        for row in rows
        if isinstance(row, dict) and isinstance(row.get("id"), str)
    }
    adapter_row = by_id.get(model)
    base_row = by_id.get(base_model)
    if adapter_row is None:
        raise RuntimeError(f"served adapter model {model!r} is absent")
    if base_row is None:
        raise RuntimeError(f"served base model {base_model!r} is absent")

    served_adapter_root = Path(
        str(adapter_row.get("root", ""))
    ).expanduser().resolve()
    if served_adapter_root != adapter_path:
        raise RuntimeError(
            f"served model {model!r} is not rooted at accepted adapter "
            f"{adapter_path}"
        )
    if adapter_row.get("parent") != base_model:
        raise RuntimeError(
            f"served adapter {model!r} does not name {base_model!r} as parent"
        )
    if base_row.get("parent") is not None:
        raise RuntimeError(
            f"served base model {base_model!r} unexpectedly has a parent"
        )
    served_base_root = Path(
        str(base_row.get("root", ""))
    ).expanduser().resolve()
    if not _bound_base_matches(served_base_root, bound_base_snapshot):
        raise RuntimeError(
            f"served base root {served_base_root} does not match bound "
            f"checkpoint {bound_base_snapshot!r}"
        )


async def _run(args: argparse.Namespace) -> dict:
    binding_path = args.pool_binding.expanduser().resolve()
    binding = load_pool_binding(binding_path)
    adapter_path = (REPO_ROOT / binding.checkpoint.adapter_path).resolve()
    manifest = _read_json(adapter_path / "fugu_policy_revision.json")
    behavior_revision = manifest.get("policy_revision")
    if not isinstance(behavior_revision, str) or not behavior_revision.strip():
        raise RuntimeError("accepted adapter has no behavior-policy revision")

    base_url = args.base_url.rstrip("/")
    models_url = f"{base_url}/models"
    _assert_served_checkpoint(
        models_url=models_url,
        model=args.model,
        base_model=args.base_model,
        adapter_path=adapter_path,
        bound_base_snapshot=binding.checkpoint.base_model_snapshot,
    )
    prompt_counter = LocalModelPromptTokenCounter(
        model=args.model,
        models_url=models_url,
    )
    client = AsyncOpenAI(
        base_url=base_url,
        api_key="x",
        timeout=600.0,
        max_retries=0,
    )
    try:
        return await collect_synthetic_branchpoints(
            output_dir=args.output_dir.expanduser().resolve(),
            behavior_policy_revision=behavior_revision,
            runtime_revision=PRODUCT_RUNTIME_REVISION,
            pool_binding_path=binding_path,
            scenario_count=args.scenarios,
            samples_per_scenario=args.samples_per_scenario,
            seed=args.seed,
            concurrency=args.concurrency,
            controller_factory=lambda _scenario, policy: OpenAILiveController(
                model=args.model,
                base_url=base_url,
                client=client,
                max_tokens=MAX_OUTPUT_TOKENS,
                seed=policy.sampling_seed,
                temperature=1.0,
                record_token_data=True,
                prompt_token_counter=prompt_counter,
                max_input_tokens=MAX_INPUT_TOKENS,
                supplies_topology=True,
                capability_refs=True,
            ),
        )
    finally:
        await client.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--pool-binding",
        type=Path,
        default=Path(
            "director/manifests/fugu_clean_v1/grpo_pilot_train/"
            "current_pool_binding_v11.json"
        ),
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:8010/v1")
    parser.add_argument("--model", default="fugu-27b-conductor")
    parser.add_argument("--base-model", default="fugu-27b-base")
    parser.add_argument("--scenarios", type=int, default=32)
    parser.add_argument("--samples-per-scenario", type=int, default=8)
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20_260_724)
    args = parser.parse_args()
    report = asyncio.run(_run(args))
    print(
        json.dumps(
            {
                "version": report["version"],
                "verdict": report["verdict"],
                "runtime_revision": report["runtime_revision"],
                "sample_count": report["sample_count"],
                "eligible_count": report["eligible_count"],
                "disposition_counts": report["disposition_counts"],
                "reward_counts": report["reward_counts"],
                "paid_calls": report["paid_calls"],
                "output": str(args.output_dir.expanduser().resolve()),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
