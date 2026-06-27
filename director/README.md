# Director — a Fugu agentic-router replica

A faithful replica of Sakana's **Fugu** learned LLM orchestrator (Fugu technical report;
built on **Trinity** [arXiv:2512.04695] and **Conductor** [arXiv:2512.04388]).

Fugu is a tiny **learned router**: a *frozen* small-LM backbone (Qwen3-0.6B) is a feature
extractor, and a lightweight head reads its hidden state and picks **which frontier worker
model acts at each step**. The objective here is an **agentic (multi-turn) router** — best
worker *per turn* across long-horizon tool-using tasks — not a single-turn Q→A expert.
Everything is **judge-free**: rewards come from verifiable signals (tests, DB-state,
exact-match), never an LLM judge.

```
transcript ─► frozen Qwen3-0.6B (+ SVF scales) ─► h at token −2 ─► bias-free head ─► L logits ─► argmax ─► worker
```

## Architecture
- **Selection-only, decision-only.** No roles; the backbone never generates text.
- **Raw surface form, hidden@−2.** Transcript fed as raw `"role: content\n"` (NOT a chat
  template — that collapses routing accuracy); routing feature is the penultimate token.
- **Bias-free head** + **singular-value fine-tuning** (SVF) on the 7 layer-26 projections →
  **~13K trainable params** (`fugu/svf.py`, `fugu/model.py`).
- **Long-context routing.** Transcripts are windowed to the backbone's capacity (cap
  **32768**, auto-capped to the model); `head_tail` keeps the goal + most-recent state when
  over the cap (never first-N, which would drop recent turns). Configurable.

## Worker pool (OpenRouter)
All workers via OpenRouter's OpenAI-compatible endpoint (`OPENROUTER_API_KEY`). Default 6:
`anthropic/claude-opus-4.8`, `openai/gpt-5.5`, `google/gemini-3.5-flash`, `z-ai/glm-5.2`,
`qwen/qwen3.7-max`, `moonshotai/kimi-k2.7-code`. Features: disk cache (every
`(model,prompt,sampling)`), **prompt caching** (Anthropic `cache_control` + auto for
others), **function-calling** (`call_tools`), budget cap.

## Two training stages
1. **SFT warm-start (single-step, judge-free).** A curation engine probes a large candidate
   pool, keeps only **worker-disagreement** items, balances + splits, then labels soft
   targets; the router is trained by **KL** to them. Sources (13, GPQA-D held out for eval):
   math (MATH-500, AIME, Omni-MATH, MMLU-Pro-math), code (HumanEval, MBPP, code_contests,
   TACO), science (MMLU-Pro-STEM, SuperGPQA), general (MMLU-Pro, SuperGPQA), reasoning
   (ARC-AGI-2). Graders: exact-match, sympy, sandboxed code-exec (functional + stdin/stdout),
   MC, grid.
2. **sep-CMA-ES (agentic, the objective).** Per-step routed rollouts maximize terminal
   reward. **Checkpointed + resumable**, **replica-parallel** (K routers = K concurrent
   candidates), **flat-fitness-robust** (sparse rewards don't stop it early), with **fitness
   shaping** (diversity/turn/cost) to prevent single-worker collapse.

## Agentic harnesses (all judge-free, validated live)
| Harness | Modality | Reward |
|---|---|---|
| SWE-Bench Verified | shell/coding | official tests |
| SWE-Bench Pro | shell, multi-language | ScaleAI harness |
| Terminal-Bench | terminal | native tests |
| tau-bench | tool-use (function calls) | DB-state |

## Install (uv)
```bash
cd director
uv venv && uv pip install --python .venv/bin/python -e ".[all]"   # core + agentic + dev
export OPENROUTER_API_KEY=sk-or-...
```
Extras: `swebench`, `terminalbench`, `taubench` (git), `dev`, `all`.

## CLI
```bash
# single-step warm-start
director curate     --manifest-dir m/ --per-domain-target 750   # probe → keep disagreement → split
director label      --manifest-dir m/ --n-samples 4 --out labels.jsonl
director train-sft  --labels labels.jsonl --out router.pt
director eval       --ckpt router.pt --dataset gpqa             # vs each single worker (held-out)

# agentic (the objective)
director agentic-eval     --ckpt router.pt --harness swebench --limit 5
director agentic-train    --ckpt router.pt --harness swebench --checkpoint-dir ck/ --out router.pt
director terminal-bench-eval --ckpt router.pt --task-id hello-world
director taubench-eval    --ckpt router.pt --domain retail --limit 5
director taubench-train   --ckpt router.pt --checkpoint-dir ck/ --out router.pt
```
All commands take `--config cfg.yaml` (`DirectorConfig`: workers, pool, sampling,
`featurizer` context settings, backbone).

## Layout
```
director/shared/   worker pool (providers, cache, prompt_cache, budget, pool), tasks,
                   sources, verifiers (graders), curate, eval, transcript, types
director/fugu/     svf, model (SelectionRouter + windowing), labels, sft, cmaes
                   (checkpoint/parallel), inference, run (CLI + build_router)
director/agentic/  env + rollout (shell), toolenv + toolcall (tool-use), fitness (shaping),
                   swebench_env, swebench_pro_env, terminalbench_agent, taubench_env, run
tests/             45 offline tests (FakeProvider/ScriptedEnv, $0); -m network/gpu for live
```

## Tests
```bash
cd director && .venv/bin/python -m pytest      # 45 offline; network/gpu gated
```
