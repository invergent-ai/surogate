# GRPO with a RULER judge

RULER scores groups of student rollouts with an independent LLM judge. This example
uses Qwen3-0.6B for training/rollouts and Qwen3-1.7B as the judge, on the local
`markdown-table-qa` environment.

**The judge must support `response_format: json_schema`. Native Surogate serving
does not implement constrained structured output.** Start an external compatible
judge, for example vLLM in its own environment on a third GPU:

```bash
CUDA_VISIBLE_DEVICES=2 vllm serve Qwen/Qwen3-1.7B \
  --port 8001 --max-model-len 16384
```

Once it is ready, run from the repository root with Surogate's environment active:

```bash
surogate grpo --train examples/ruler/train.yaml --infer examples/ruler/infer.yaml \
  --orch examples/ruler/orch.yaml --infer-gpus 0 --trainer-gpus 1
```

The runner manages the native student server and trainer. The external judge stays
running independently. For an existing judge service, change `ruler.judge.base_url`
and `ruler.judge_model` in [orch.yaml](orch.yaml), and export `RULER_JUDGE_API_KEY`
if it requires authentication. Do not pass `--judge-infer`: that launches a native
server which cannot satisfy the judge's schema request.

The config enables verification, four rollouts per example, and `ruler.mode: replace`.
Use `add` to combine environment reward and judge reward, or `metric` to log the judge
score without changing the reward. Optional `ruler.rubric` supplies judging criteria;
`ruler.cost.input_per_million` / `output_per_million` enable cost estimates.
Judge failures raise errors here instead of silently turning into zero rewards.

Training and evaluation both run for a reachable twenty-step schedule; evaluation
uses the environment's verifier every ten steps. Compare `ruler/score_mean` with
`eval/markdown-table-qa/avg@4` in `/tmp/grpo_metrics.jsonl` to check that the judge's
preferences agree with task correctness. Trainer metrics live in
`outputs/ruler/metrics.jsonl`.

A Qwen judge can spend its output budget on thinking before emitting JSON. The
example disables judge thinking and allows 4096 completion tokens. Increase context
and output budgets or reduce group size if judging truncates. Keep
`rollouts_per_example >= 2`, `verification.enabled: true`, and the orchestrator output
inside the trainer output as `run_default`.
