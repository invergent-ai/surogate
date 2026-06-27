# Fugu router — what we built, and the problem we found

## Goal
Replicate Sakana's **Fugu** learned orchestrator: a frozen Qwen3-0.6B backbone + a tiny
bias-free selection head + SVF (singular-value fine-tuning), trained in two stages
(SFT warm-start on soft reward targets → sep-CMA-ES on terminal reward) to route each
query/task to one of a pool of frontier **open-weight** workers. Success = the router's
accuracy **beats the best single worker**.

## What we built (and it all works)
- **Pool & infra:** 6 open workers via OpenRouter (deepseek-v4-pro, kimi-k2.7-code,
  glm-5.2, mimo-v2.5-pro, minimax-m3, deepseek-v4-flash); cached async pool; budget tracking.
- **Backbone + router:** frozen Qwen3-0.6B, hidden state at the penultimate token,
  bias-free head, SVF on layer-26 (matches Trinity exactly: 9 tensors / 9216 svals).
  Trainable params = 13,312.
- **Stage-1 data:** FULL clean build — 2,253 soft-label tasks (n=4 rewards/worker,
  τ=0.1) across math/code/science/general, plus 397 held-out val; eval-set contamination
  filtered by prompt-hash.
- **SFT trainer:** KL-to-soft-target on head+SVF, grad-accumulation, lr tuned to 1e-4.
- **Agentic stack:** tau-bench (retail/airline) + SWE-smith runners; agentic reward bank
  (480 cells); grader fixed (`sys.executable`), gold patches verified at 1.0.
- **OpenCode-in-container harness:** standalone opencode binary runs inside SWE-smith
  containers; glm drives real source edits at high reasoning with auto-approve.
- **Instant bank-based optimizer:** per-task routing optimization (agentic SFT + sep-CMA-ES)
  using the precomputed bank as fitness — no live rollouts.

All of the above is functional. The blocker is **not** any of these components.

## Three things we discovered along the way
1. **Live per-turn CMA-ES is infeasible for us.** Reading the Trinity source showed its ES
   was only feasible because it **self-hosted 60 small (32B-class) worker instances** across
   4 GPUs (60-way rollout concurrency, no API limits) on **fast 5-turn** tasks. Our pool is
   frontier-scale open models (API-only) on ~30-turn tau tasks → a single 8-generation run
   projected to ~10 hours; at full scale, years. (Production Trinity itself decouples this:
   traces → exported fitness dataset → external ES. Our bank is that fitness evidence.)
2. **The agentic-coding harness failures were setup, not capability.** SWE-smith "all-fail"
   was a chain of harness bugs (wrong grader interpreter → low reasoning → permission gate),
   each masking the next. glm-5.2 does real agentic coding in OpenCode once configured.
3. **The headroom that motivated Stage-2 was noise** (see below).

## The core problem
**The worker pool has no learnable accuracy complementarity.** A router can only beat
best-single if some worker reliably *succeeds where the best one fails*, predictably from
features. Noise-corrected tests on our two cleanest datasets show that signal does not exist:

**Agentic tau (80 complete bank items):**
```
observed headroom  +0.200
NULL (n=1 noise, zero signal)  +0.229   → observed is BELOW null  ⇒ NOISE
```
The "+0.203 tau headroom" we had logged as a win was a measurement artifact of n=1 rewards.
Only a coarse per-domain effect is real (retail→flash 0.725 / airline→mimo 0.45) and it's
worth just **+0.025**. A per-task router overfit train (cap 22%) and lost on held-out (cap −75%).

**Denoised diverse single-step (n=4, 2,253 tasks — our best data):**
```
per-worker: deepseek_flash 0.678 (DOMINANT) >> glm 0.402 > ... > mimo 0.24
best-single 0.678   oracle 0.802   observed headroom +0.124
NULL (n=4 noise, zero signal)  +0.216
SIGNAL above noise = −0.092   ⇒ NO complementarity
```
Even denoised, diverse, and large-n: one worker dominates, observed headroom is *below* the
noise floor. There is nothing for a router to capture.

### Why
Six strong, similar open **generalists** succeed and fail **together** (positively correlated
per task) and differ mainly in **overall skill** (one dominates) — not in *which individual
tasks they win*. Frontier models are converging; when they agree, the best one is unbeatable.
This is structural to the pool, not fixable by more SFT/CMA-ES/harness/compute.

## Implication / path forward
Accuracy routing needs a pool with **genuine, measured complementarity** — workers from
different families (idiosyncratic errors diverge) and/or task specialists, each clearly best
on some identifiable slice. The method itself is already validated (RouterBench: 84.5%
routable *with* a complementary pool) and our training machinery is ready.

**New step 0 for any pool decision:** run the instant null-corrected headroom test
(oracle-headroom vs the n≥4 noise floor). Only build the router if real signal clears the
floor. That ~free check would have prevented this entire detour and is now the gate.
