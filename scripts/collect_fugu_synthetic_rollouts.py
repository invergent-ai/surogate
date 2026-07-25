#!/usr/bin/env python3
"""Collect exact-token synthetic rollouts from the accepted local conductor."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from urllib.request import urlopen

from openai import AsyncOpenAI

from director.agentic.fugu_ultra_terminal import (
    LocalModelPromptTokenCounter,
    PRODUCT_RUNTIME_REVISION,
)
from ultra.live_control import OpenAILiveController
from ultra.pool_binding import load_pool_binding
from ultra.synthetic_collection import collect_synthetic_rollouts


MAX_INPUT_TOKENS = 7_680
MAX_OUTPUT_TOKENS = 512


def _read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"{path} must contain one JSON object")
    return value


def _assert_served_adapter(
    *,
    models_url: str,
    model: str,
    adapter_path: Path,
) -> None:
    with urlopen(models_url, timeout=10) as response:  # noqa: S310 - local only
        payload = json.load(response)
    rows = payload.get("data") if isinstance(payload, dict) else None
    served = next(
        (
            row
            for row in rows or []
            if isinstance(row, dict) and row.get("id") == model
        ),
        None,
    )
    root = (
        Path(str(served.get("root", ""))).resolve()
        if isinstance(served, dict)
        else None
    )
    if root != adapter_path:
        raise RuntimeError(
            f"served model {model!r} is not rooted at accepted adapter {adapter_path}"
        )


async def _run(args: argparse.Namespace) -> dict:
    repo_root = Path(__file__).resolve().parents[1]
    binding_path = args.pool_binding.expanduser().resolve()
    binding = load_pool_binding(binding_path)
    adapter_path = (repo_root / binding.checkpoint.adapter_path).resolve()
    manifest = _read_json(adapter_path / "fugu_policy_revision.json")
    behavior_revision = manifest.get("policy_revision")
    if not isinstance(behavior_revision, str) or not behavior_revision.strip():
        raise RuntimeError("accepted adapter has no behavior-policy revision")

    base_url = args.base_url.rstrip("/")
    models_url = f"{base_url}/models"
    _assert_served_adapter(
        models_url=models_url,
        model=args.model,
        adapter_path=adapter_path,
    )
    prompt_counter = LocalModelPromptTokenCounter(
        model=args.model,
        models_url=models_url,
    )
    client = AsyncOpenAI(
        base_url=base_url,
        api_key="x",
        timeout=180.0,
        max_retries=0,
    )
    try:
        report = await collect_synthetic_rollouts(
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
    return report


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
    parser.add_argument("--scenarios", type=int, default=24)
    parser.add_argument("--samples-per-scenario", type=int, default=4)
    parser.add_argument("--concurrency", type=int, default=12)
    parser.add_argument("--seed", type=int, default=20_260_724)
    args = parser.parse_args()
    report = asyncio.run(_run(args))
    print(
        json.dumps(
            {
                "version": report["version"],
                "verdict": report["verdict"],
                "runtime_revision": report["runtime_revision"],
                "rollout_count": report["rollout_count"],
                "scenario_count": report["scenario_count"],
                "paid_calls": report["paid_calls"],
                "output": str(args.output_dir.expanduser().resolve()),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
