"""Decode on-disk GRPO rollouts (run_default/rollouts/step_N/rollouts.bin) to SEE the conductor's
generated workflows + their reward/advantage. Non-invasive: reads the .bin + tokenizer only (no GPU,
no vLLM contention). Usage: .venv/bin/python scratch_show_rollouts.py <step> [n_per_reward]"""
import sys, collections
import msgspec
from surogate.grpo.transport.types import TrainingBatch
from transformers import AutoTokenizer

STEP = sys.argv[1] if len(sys.argv) > 1 else "14"
N = int(sys.argv[2]) if len(sys.argv) > 2 else 1  # examples per reward level
path = f"output/fugu_ultra_lcb/run_default/rollouts/step_{STEP}/rollouts.bin"
batch = msgspec.msgpack.Decoder(type=TrainingBatch).decode(open(path, "rb").read())
tok = AutoTokenizer.from_pretrained("output/fugu_ultra_workflow_sft_qwen3_8b")

ex = batch.examples
print(f"=== step {STEP}: {len(ex)} rollouts ===")
rew = [s.reward for s in ex]
print("reward distribution:", dict(sorted(collections.Counter(rew).items())))
comp_lens = [len(s.completion_ids) for s in ex]
print(f"completion length tokens: min={min(comp_lens)} mean={sum(comp_lens)//len(comp_lens)} max={max(comp_lens)}")

def task_tail(s):
    p = tok.decode(s.prompt_ids, skip_special_tokens=True)
    return p[-600:]  # the actual task sits at the end of the few-shot Conductor prompt

# show N examples per distinct reward level (1.0 correct, 0.5 valid-wrong, 0.0 unparseable/fail)
by_r = collections.defaultdict(list)
for s in ex:
    by_r[s.reward].append(s)
for r in sorted(by_r, reverse=True):
    for s in by_r[r][:N]:
        wf = tok.decode(s.completion_ids, skip_special_tokens=True)
        print("\n" + "=" * 78)
        adv = s.advantage if s.advantage is not None else float("nan")
        print(f"reward={r}  advantage={adv:+.3f}  completion_tokens={len(s.completion_ids)}")
        print("--- TASK (tail of prompt) ---")
        print(task_tail(s).strip())
        print("--- CONDUCTOR WORKFLOW (generated) ---")
        print(wf.strip()[:2400])
