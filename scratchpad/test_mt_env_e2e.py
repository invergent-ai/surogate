"""Stage-2 MT env end-to-end smoke — ZERO worker spend (provider_mode=fake), live local vLLM.

Validates the three pending deploy items on the REAL stack (verifiers 0.1.11 + surogate vLLM):
  1. rubric wiring: ultra_mt_reward scores the final turn via runtime.score, flags surface;
  2. control flow: fake workers fail turn-0 -> revise turn generated -> max_turns=2 stop;
  3. multi-span tokens: OpenAIChatCompletionsTokenClient (the trainer's TITO client) returns
     per-turn token dicts for BOTH assistant spans.

Run: ULTRA_ALLOW_YUNWU=0 PYTHONPATH=ultra:. .venv/bin/python scratchpad/test_mt_env_e2e.py
"""
import asyncio, importlib.util, json, sys

sys.path.insert(0, "ultra")

spec = importlib.util.spec_from_file_location(
    "fpe", "environments/fugu-ultra-pilot/fugu_ultra_pilot.py")
mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)

from openai import AsyncOpenAI
from verifiers.clients.openai_chat_completions_token_client import OpenAIChatCompletionsTokenClient


async def main():
    D = "director/manifests/fugu_clean_v1/grpo_pilot_train"
    env = mod.load_environment(
        pilot_config_path=f"{D}/pilot_config_singleturn.json",
        task_manifest_path=f"{D}/hard_mix_all_taskspecs.jsonl",
        max_turns=2,
        provider_mode="fake",
        lane="single_turn",
        max_examples=2,
        artifact_dir=None,
        cache_dir=None,
        workflow_record_cache_dir=None,
        score_rollouts=True,
        max_seq_len=8192,
    )
    print(f"env: {type(env).__name__}, max_turns={env.max_turns}, dataset={len(env.dataset)}")
    assert type(env).__name__ == "FuguUltraMultiTurnEnv"

    row = env.dataset[0]
    inp = {
        "prompt": row["prompt"],
        "example_id": 0,
        "task": row.get("task", "fugu_ultra_pilot"),
        "info": row.get("info") or {},
    }
    client = OpenAIChatCompletionsTokenClient(
        AsyncOpenAI(base_url="http://localhost:8007/v1", api_key="x", timeout=180.0))
    # the exact run_rollout sequence (rollout -> rubric.score_rollout), on the raw state
    state = await env.rollout(
        inp, client, model="default",
        sampling_args={"temperature": 1.0, "max_tokens": 1024,
                       "extra_body": {"chat_template_kwargs": {"enable_thinking": False}}},
    )
    await env.rubric.score_rollout(state)
    d = state
    traj = d.get("trajectory") or []
    print(f"turns generated: {len(traj)}")
    assert len(traj) == 2, f"expected 2 turns (fake workers fail turn-0), got {len(traj)}"

    for i, step in enumerate(traj):
        toks = step.get("tokens") if isinstance(step, dict) else getattr(step, "tokens", None)
        n_prompt = len((toks or {}).get("prompt_ids") or (toks or {}).get("prompt_token_ids") or [])
        n_comp = len((toks or {}).get("completion_ids") or (toks or {}).get("completion_token_ids") or [])
        print(f"  turn {i}: tokens={'YES' if toks else 'NO'} prompt_ids={n_prompt} completion_ids={n_comp}")
        assert toks, f"turn {i} carries no token dict — TITO path broken"
        assert n_comp > 0, f"turn {i} has empty completion ids"

    reward = d.get("reward")
    metrics = d.get("metrics") or {}
    records = d.get("turn_records") or []
    print(f"reward={reward} metrics={dict(metrics)}")
    print(f"turn_records={len(records)} (rubric must have scored the final turn)")
    assert reward is not None, "rubric did not produce a reward"
    assert len(records) == 2, "final turn was not scored by ultra_mt_reward"
    assert metrics.get("ultra_mt_turns") == 2.0, metrics
    assert "_ultra_grade_success" in d, "terminal flags not surfaced to rollout state"
    print("\nMT ENV E2E SMOKE PASSED (2 turns, token spans on both, rubric scored the final turn)")


asyncio.run(main())
