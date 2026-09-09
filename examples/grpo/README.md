# GRPO: split GPUs, one shared GPU, or separate processes

The same [train](train.yaml), [infer](infer.yaml), and [orch](orch.yaml) configs run
Qwen3-0.6B on the local `markdown-table-qa` reward environment. Run from the repository
root with the RL dependencies installed. The environment is loaded from
`environments/markdown-table-qa`; no separate environment installation is needed.

```bash
# Two GPUs: separate rollout server and trainer.
surogate grpo --train examples/grpo/train.yaml --infer examples/grpo/infer.yaml \
  --orch examples/grpo/orch.yaml --infer-gpus 0 --trainer-gpus 1
```

For **one GPU**, use a fresh `outputs/grpo` directory and the same files:

```bash
CUDA_VISIBLE_DEVICES=0 surogate grpo-colocate \
  --train examples/grpo/train.yaml --infer examples/grpo/infer.yaml --orch examples/grpo/orch.yaml
```

Colocation shares one BF16 base and alternates generation and training. It requires
unquantized BF16 safetensors, LoRA, one GPU, and a model supported by the shared
runner (Nemotron is excluded). It does not support checkpoint resume, QLoRA, CPU
weight offload, or QeRL noise. See [colocation limits](../../docs/guides/rl-colocate.md).

For real function-tool rollouts, use [tools-orch.yaml](tools-orch.yaml) with the
same training/inference files and a fresh output directory:

```bash
CUDA_VISIBLE_DEVICES=7 surogate grpo-colocate \
  --train examples/grpo/train.yaml --infer examples/grpo/infer.yaml \
  --orch examples/grpo/tools-orch.yaml
```

The local [tool environment](tool_env.py) exposes `add(a, b)`, executes the model's
structured tool call, returns a tool message, and rewards the final answer.
Schemas come from Verifiers' `ToolEnv`; custom environments can expose their own
Python functions the same way. Tool results are context tokens with no GRPO loss.
Every supported native-colocate model family can use tools. The shared training
server selects the checkpoint protocol or adapts templates without tool support;
base models still need training to learn reliable calls. See the
[protocols and limits](../../docs/guides/rl-colocate.md#agentic-tool-rollouts).

For independent processes, run these in three terminals:

```bash
CUDA_VISIBLE_DEVICES=0 surogate grpo-infer examples/grpo/infer.yaml
surogate grpo-orch examples/grpo/orch.yaml
CUDA_VISIBLE_DEVICES=1 surogate grpo-train examples/grpo/train.yaml
```

Processes must share the output filesystem and see the same paths to broadcasts.
For remote servers, update `client.base_url` and the bind address in the inference
config. Multi-node SFT uses Ray; this RL layout uses separate service processes.

The configs demonstrate grouped rewards, token-in/token-out rollouts, a temperature
schedule, online difficulty filtering, periodic evaluation, local metrics, and
orchestrator/trainer checkpoints. Evaluation runs before training and every ten
steps. Read metrics from `outputs/grpo/metrics.jsonl` (trainer) and
`/tmp/grpo_metrics.jsonl` (orchestrator).

Keep these values aligned when customizing:

- The model name in all three configs and `max_steps` in trainer/orchestrator.
- `sequence_len` in trainer/orchestrator, with sufficient `max_model_len` on the server.
- `orch.output_dir` must be `train.output_dir/run_<name>`.
- `rollouts_per_example` must divide `batch_size`; inference needs room for prompt
  plus completion, and adapters must fit `max_lora_rank`.

`use_token_client: true` preserves previously generated tokens and appends the
environment's response using a template bridge. Surogate supplies a bridge that
preserves real function names and handles Qwen's reasoning rules. Truncated
turns or incompatible custom templates can still fall back to message rendering;
the client logs this. The [multi-turn OPD example](../turnopd/README.md) uses
message rendering explicitly. Very short completion limits can truncate every answer and leave zero
reward; a small model may still need easier tasks or more steps to learn.

For split-mode resume, set trainer `resume_from_checkpoint: true` and orchestrator
`ckpt.resume_step` to the corresponding saved step (or `-1` for latest), preserving
the same output paths. Optional `ckpt.spool_inflight: true` persists in-flight
rollouts. Do not use the resume variant with colocation.

The active GRPO loss accepts `adv_tau`, `teacher_tau`, `kl_tau`, `ipo_mask_low`,
`ipo_mask_high`, and `ratio_clip`. The teacher term is demonstrated in
[TurnOPD](../turnopd/README.md); [RULER](../ruler/README.md) adds an external judge.
Native serving cannot apply QeRL base-weight noise, so `noise_scheduler.enabled`
is rejected by both native runners. Replay/OPD reference-loss knobs do not currently
form a complete replay-training CLI workflow and are not presented as runnable recipes.

## Token budgets, filtering and async collection

To collect a batch by generated-token budget, copy `orch.yaml`, remove `batch_size`
and add:

```yaml
token_batch_size: 8192
max_inflight_rollouts: 16
```

Keep `rollouts_per_example`; collection still processes complete groups. For
metrics on degenerate rollouts, add a filter (set `enforce: true` to mask them):

```yaml
filters:
  - type: repetition
    enforce: false
    window: 32
    prob_threshold: 0.99
```

For several reward tasks, append named entries to `env` and set
`buffer.env_ratios` to one positive weight per environment. Different environment
arguments can create easier/harder variants of the same task. For split or
separate-process runs, tune `max_async_level` in both trainer and orchestrator and
`max_off_policy_steps` in the orchestrator to bound policy staleness. Colocation
sets synchronous collection itself.

Separate processes can send batches over ZMQ by setting `transport_type: zmq`
in the trainer and the following in the orchestrator (adapter broadcasts still
need a shared filesystem):

```yaml
rollout_transport:
  type: zmq
  host: localhost
  port: 5555
  hwm: 10
```

Use filesystem transport with colocation. See the [RL guide](../../docs/guides/rl-training.md)
for custom environments and advantage functions.
