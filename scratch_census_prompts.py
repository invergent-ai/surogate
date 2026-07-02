"""Token census: every training (461) + eval (60) prompt under the PAPER prompt.
Gate: max prompt tokens + 1024 completion must fit 8192 (vLLM max_model_len AND trainer seq_len)."""
import importlib.util, json, sys
from collections import Counter

RAW = "/var/lib/mesh/flavius/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218"
D = "director/manifests/fugu_clean_v1/grpo_pilot_train"


def main():
    sys.path.insert(0, "ultra")
    spec = importlib.util.spec_from_file_location("fpe", "environments/fugu-ultra-pilot/fugu_ultra_pilot.py")
    env = importlib.util.module_from_spec(spec); spec.loader.exec_module(env)
    from ultra.schemas import TaskSpec
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(RAW)
    cfg = json.load(open(f"{D}/pilot_config_singleturn.json"))

    for name, path in (("train461", "hard_mix_all_taskspecs.jsonl"), ("eval60", "heldout_trend60_taskspecs.jsonl")):
        lens = []
        for line in open(f"{D}/{path}"):
            t = TaskSpec.model_validate(json.loads(line))
            msgs = env._prompt_for_task(t, cfg, "single_turn", max_task_chars=12000)
            txt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
            lens.append(len(tok(txt).input_ids))
        lens.sort()
        n = len(lens)
        over = sum(1 for x in lens if x > 8192 - 1024)
        print(f"{name}: n={n} p50={lens[n//2]} p90={lens[int(n*0.9)]} p99={lens[int(n*0.99)]} max={lens[-1]} | over-budget(>{8192-1024}): {over}", flush=True)


if __name__ == "__main__":
    main()
