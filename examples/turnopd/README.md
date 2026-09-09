# Multi-turn on-policy distillation

This is one Qwen3.5-2B student / Qwen3.5-9B teacher recipe on the local `multihop-tools`
environment. It demonstrates teacher log-ratio rewards, turn-level supervision
metrics, and adaptive rollout-depth budgeting. The tiny step budget demonstrates
wiring; it is not a claim of reward improvement or a paper reproduction.

The teacher must score the **exact student token IDs** using the same tokenizer.
Native Surogate serving does not score prompt tokens. The included
[teacher_proxy.py](teacher_proxy.py) translates the orchestrator's token endpoint
into a vLLM `/v1/completions` prompt-scoring request. It selects each actual input
token's logprob and keeps the first-token placeholder so supervision stays aligned.

Run the external teacher in a separate vLLM environment (GPU 2), then the CPU proxy:

```bash
CUDA_VISIBLE_DEVICES=2 vllm serve Qwen/Qwen3.5-9B \
  --port 8010 --max-model-len 8192 --max-logprobs 1
# Another terminal, no GPU used by the proxy:
python examples/turnopd/teacher_proxy.py --upstream http://127.0.0.1:8010/v1 --port 8008
```

With Surogate's environment active, run from the repository root:

```bash
surogate grpo --train examples/turnopd/train.yaml --infer examples/turnopd/infer.yaml \
  --orch examples/turnopd/orch.yaml --infer-gpus 0 --trainer-gpus 1
python examples/turnopd/analyze.py outputs/turnopd/turn_stats.jsonl
```

The proxy only exposes health/model discovery and token scoring on localhost; it
is an example adapter for a local teacher, not a general inference server. Set
`TEACHER_API_KEY` if the upstream requires authentication. Errors and missing or
misaligned prompt scores stop the request instead of inventing training targets.
For other student/teacher pairs, verify tokenizer/vocabulary compatibility first.

[orch.yaml](orch.yaml) uses `use_token_client: false`: this chat template can rewrite
earlier thinking blocks, invalidating token-prefix extension. `rollout_depth`
adapts the turn cap using success-conditioned coverage, with periodic full-depth
probes. Turn depth and per-turn generation limits are independent. With no successful
trajectories the coverage controller has little information; start with an easier
environment or a stronger student.

[train.yaml](train.yaml) sets `teacher_tau: 1.0` and `turn_diagnostics: true`.
The diagnostics route the loss through Python and are slower; disable them for
normal throughput. Set `rollout_depth.enabled: false` to compare fixed-depth rollouts
without maintaining another almost-identical experiment directory.

To inspect task competence before training, start only the student and probe it:

```bash
CUDA_VISIBLE_DEVICES=0 surogate grpo-infer examples/turnopd/infer.yaml
python examples/turnopd/probe_env.py --port 8007 --model Qwen/Qwen3.5-2B -n 20
```

Stop that standalone student before using the split runner, which starts its own.
The probe reports malformed tools, repeated lookups, turn-cap hits and task success.
