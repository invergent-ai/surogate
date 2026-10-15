# GRPO for Surogate3.7-35B-A3B: CISPO + Rollout Routing Replay (R3)

Status 2026-10-15, branch `claude/grpo-cispo-r3` (worktree `~/surogate-rl` on Spark 1). Step 0 of
`/mnt/data/work/rl/ultra_pilot/PLAN.md` (Ultramarine recipe: CISPO, group-relative advantages mean-centred without std
normalisation, zero-advantage groups dropped, Rollout Routing Replay).

## What the recipe needs and where it stands

| Piece | State |
|---|---|
| Mean-centred advantages, no std normalisation | already the default (`orchestrator/advantage.py: default_advantage_fn`) |
| Drop zero-advantage groups | already there: the flat-group filter (`grpo_orch_config.py`, buffer dispersion form) |
| CISPO objective | **done** (commit edee6259): `loss.variant: cispo`. The native seed already computes `A * min(ratio, ratio_clip)` per kept token; CISPO = masks wider than any probability difference, `kl_tau = 0`, `ratio_clip = 1 + eps_high`. Tests: `tests/grpo/test_cispo_loss.py` |
| R3, trainer side | **done in this change**: kernel `moe_topk_replay`, hook in `dispatch_moe_topk`, `DslRunState::set_routing_replay`, Python `trainer.set_routing_replay(rows)` / `clear_routing_replay()` / `routing_replay_diagnostics()`. Test: `csrc/src/testing/kernels/test-moe-topk-replay.cu` |
| R3, rollout capture | **to do** (below) |
| R3, rollout -> trainer plumbing | **to do** (below) |

## R3 trainer side (this change)

- Replay rows: int16 `[layer][token][k]` for the micro-batch tokens; a row starting with -1 is not replayed (padding,
  prompt tokens without captured routing). 256 experts fit int16; 40 layers x 8 x 2 B = 640 B per token (21 MB for a
  32k-token sample).
- `dispatch_moe_topk` runs the normal top-k, then `moe_topk_replay` overwrites the selection for replayed rows and
  recomputes the weights from THIS forward's router probabilities at the replayed experts (same rule as the forward:
  normalized over the selection, or softmax). The router therefore still gets gradient, and `moe_topk_backward`
  (indices treated as constants) is unchanged. Activation recomputation re-runs the op with the same rows.
- Not applied for routing with a correction bias / rounding / index sorting (DeepSeek-style); Qwen3.5-MoE uses none.
- Out-of-range ids leave the row untouched and are counted; forwards whose token count / top_k do not match the rows
  run without replay and are counted (`routing_replay_diagnostics()`), so a silent mis-alignment is visible.
- Chunked-sequence mode: `DslRunState::set_routing_replay_token_offset(chunk_start)` must be called per chunk; NOT yet
  wired into `step_grpo_native_chunked` (the shape check makes chunked forwards run without replay and count a
  mismatch until it is).

## To do 1: capture routing in the rollout engine

- Engine: `surogate/grpo/inference/surogate_engine.py` (colocated Surogate serve) - the serve MoE path picks top-k per
  token in its own kernels. Add an optional per-request capture buffer: for every token the engine processes (prefill
  and decode, including prompt tokens of later turns) write the selected expert ids per MoE layer; return them with the
  sampled tokens (`routed_experts`, int16 `[tokens, layers, top_k]`).
- MTP speculative decoding: only ACCEPTED tokens' routing is kept (rejected draft positions are discarded with them).
- Prefix cache: a cached prefix has no fresh routing; either store routing with cached KV blocks or mark those prompt
  tokens -1 (not replayed). Prompt tokens carry no loss, so -1 there only loosens the alignment of the KV the policy
  tokens attend to; start with -1 and measure the train/infer logprob mismatch.
- NVFP4 rollout vs bf16 training weights: replay fixes the expert choice; the weights still differ, which the CISPO
  importance weight absorbs (mismatch_kl metric shows it).

## To do 2: rollout -> trainer plumbing

- Transport (`surogate/grpo/transport/types.py`): optional `routed_experts` per sample (int16 bytes + shape).
- Orchestrator: copy the engine's routing into the sample; multi-turn trajectories concatenate per turn in token order.
- Packer (`surogate/grpo/packer.py`): pack routing rows exactly like `input_ids` (same offsets), pad with -1.
- Trainer (`surogate/grpo/trainer.py`): before each micro-step's native call, `trainer.set_routing_replay(rows)` with
  `[local_gpus, layers, B*T, top_k]`; `clear_routing_replay()` after the step; log `routing_replay_diagnostics()`.
- Gate: replay on/off A/B on a short run - train/infer logprob mismatch (mismatch_kl) must drop with replay; reward
  curve and BFCL/IFEval dev gates as in the pilot plan.

## Build and test on Spark 1

- `~/gb10-logs/rl/trainer-build-rl.sh` (worktree, 8 jobs, ccache; `TARGETS="_surogate integration-tests"`), logs in
  `~/gb10-logs/rl/`. The live service (`~/surogate-serve`) keeps running; nothing here touches it.
- Python tests: `PYTHONPATH=~/surogate-rl ~/surogate-gb10/.venv/bin/python -m pytest tests/grpo/test_cispo_loss.py`.
- C++: `csrc/build/integration-tests "[replay]"`.
