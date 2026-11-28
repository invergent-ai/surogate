#!/usr/bin/env python3
"""A minimal end-to-end RL check for Inkling-2-Mini: can reward alone teach the model a rule it cannot know?

The game (rl_secret_rule_gen.py): a prompt shows an invented card and asks for its house, 1 to 4; the house follows a hidden rule (a secret
permutation of one attribute). An untrained model can only guess (25 %).

One box, one process: rollout engines on --engine-cards (the serving engine with its draft head on, the current adapter in the LoRA slot
eff-policy) and the trainer's workers on --trainer-cards (T8Trainer: expert LoRA r16 / alpha 32, B only, K 48, plain head, the PPO update of
T8Trainer.ppo_batch with GRPO advantages per prompt).
A step v: the engines hold adapter v; for the 32 prompts of schedule block v (prompts never seen before, so every step's on-policy pass rate is a
held-out number of adapter v) sample S replies each from the token prompt = the engine's own rendering of the chat request + the forced opener
<|message_model|><|content_text|> (a reply without a thinking segment), 2 tokens: the house digit and <|end_message|>; reward 1 / 0; rows = prompt
+ opener + sampled tokens, loss on the sampled tokens; ppo_batch (its replay gate demands the trainer's forward == the engine's decode-time
log-probs, bit for bit); export adapter v + 1; engines restarted on it.
Before step 0 and after the last step: the held-out read (the same prompts and seeds under the zero adapter and the final adapter), a same-norm
random-direction control adapter on the same read, and, when <pack>/split/damage.jsonl exists ({"prompt": ...} lines of unrelated problems), a
damage panel: teacher-forced nats per token on the start model's own replies to them.

<pack>: bin/surogate-engine + bin/libsinfer.so (the rollout engine), zero/ (a zero adapter directory), split/{train.jsonl, heldout.jsonl,
schedule.json} from rl_secret_rule_gen.py. The adapter this produces is a test object: it knows a made-up game and nothing else.

  rl_secret_rule.py --run-dir DIR --pack PACK --trainer-bin BIN --artifact MODEL.sinfer --trainer-cards 0,1,2,3,4,5 --engine-cards 6,7 --steps 60 --stop-pass 0.8

--sft is the supervised control (the gold reply in place of the sampled ones, the same trainer path); --shape attn64-ab adds rank-64 attention adapters and
trains expert A and B; --stop-pass ends the step loop early for the end reads. The trainer stack is T8Config's: C1, the CUDA backward, by default (library path in IFT_BWD_LIB); T8_STACK=A0 selects the torch reference.

Measured (8 x RTX 5090: 6 trainer cards, 2 rollout engines; expert LoRA B only; 32 prompts x 8 samples a step):
- RL at lr 1e-4, stack A0: LEARNED. 33 steps (8,448 rollouts, 62 minutes): held-out pass 23.8 % -> 95.3 % (+71.5 pp, paired 95 % interval 67.0 to 75.8), the
  same-norm random control 24.0 %, on-policy pass of the first / last 5 steps 23.8 % / 81.6 %, damage 0.0080 nats per token. Flat for about 8 steps, 0.5 by
  step 16, 0.93 at step 31.
- RL at lr 2e-5, 25 steps: not learned (pass 26-29 %); the supervised control at that rate does not learn either (20 steps), so the rate was the fault.
- The supervised control at lr 1e-4: flat for 18 steps, 0.65 by step 24, 0.94-1.00 over steps 31-39.
- RL at lr 1e-4 on stack C1, two seeds, about 1.7 times faster a step than A0. Second seed: LEARNED at A0's level, held-out 23.4 % -> 94.7 % (+71.3 pp,
  interval 67.2 to 75.2) in 59 steps, control 24.0 %, damage 0.0054, all four houses answered. First seed: about 0.75 by step 26, then no further update:
  the policy never answered one of the four houses, every prompt's samples agreed, so no prompt had a pass / fail spread (held-out 76.6 %). A real loop
  needs a guard against that collapse (exploration).
- Untruncated training rollouts (--roll-top-p 1.0 --roll-top-k 0, the default now; the reads keep the served pins): the first C1 seed again, with only
  that change, LEARNED without a hole: held-out 23.8 % -> 94.5 % (+70.7 pp, interval 66.6 to 74.8) in 31 steps, control 23.6 %. With top_p 0.95 /
  top_k 20 a class whose probability falls under the cut is never sampled again and cannot earn its reward.
- --thinking (whole replies with a thinking segment, about 800 tokens, the loss on every sampled token; a prompt wording that asks for a quick guess, or
  the model thinks to the cap): NOT learned in 18 steps x 256 and 23 steps x 512 rollouts (pass 26-29 %). In this game the thinking carries no
  information about the answer, so one or two tokens of a reply carry the signal: the worst case for credit assignment, not a statement about tasks
  whose reasoning decides the outcome.
The rate matters more than anything else here: check steps x rate against a working supervised recipe before an RL run, and expect a flat stretch first.
"""
import argparse
import concurrent.futures as cf
import hashlib
import http.client
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import time

import numpy as np

FLAGS = ("--max-model-len 132096 --max-num-seqs 8 --max-num-batched-tokens 4096 --kv-capacity auto --prefix-cache-host-gb 16 --enable-auto-tool-choice "
         "--default-max-tokens 32768 --max-pending-requests 64 --pending-timeout-ms 3600000 --enable-lora --max-loras 1 --max-lora-rank 16 --lora-swap in-place")
SPEC = "--spec dflash --draft-tokens 7 --spec-adaptive"
PINS = {"temperature": 1.0, "top_p": 0.95, "top_k": 20}
CTK = {"enable_thinking": True, "preserve_thinking": True, "reasoning_effort": "xhigh"}   # the template renders effort 0.99
OPENER = [200001, 200004]            # <|message_model|>, <|content_text|>: the forced opener of a reply without a thinking segment
PORT0 = 18960
LABEL = "secret-rule-test"


def log(m):
    print(f"[{time.strftime('%H:%M:%S', time.gmtime(time.time() + 7200))}] {m}", flush=True)


def jdump(path, obj):
    with open(path + ".tmp", "w") as f:
        json.dump(obj, f, indent=1, default=float)
    os.replace(path + ".tmp", path)


def post(port, route, body, timeout=600):
    last = None
    for attempt in range(3):
        try:
            c = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
            c.request("POST", route, json.dumps(body), {"Content-Type": "application/json"})
            r = c.getresponse()
            data = r.read()
            c.close()
            if r.status == 200:
                return json.loads(data)
            last = f"status {r.status}: {data[:300]!r}"
        except (OSError, http.client.HTTPException, ValueError) as e:
            last = repr(e)
        time.sleep(1 + attempt)
    raise RuntimeError(f"engine request failed 3 times on port {port}: {last}")


class Engines:
    def __init__(self, pack, artifact, cards, logdir):
        self.pack, self.artifact, self.cards, self.logdir = pack, artifact, cards, logdir
        self.procs = []
        self.ports = [PORT0 + c for c in cards]
        os.makedirs(logdir, exist_ok=True)

    def start(self, adapter, tag):
        exe = os.path.join(self.pack, "bin", "surogate-engine")
        os.chmod(exe, 0o755)
        for port in self.ports:
            with socket.socket() as s:
                s.settimeout(2)
                if s.connect_ex(("127.0.0.1", port)) == 0:
                    raise RuntimeError(f"port {port} already answers: not starting an engine")
        for c, port in zip(self.cards, self.ports):
            env = {k: v for k, v in os.environ.items() if k not in ("SUROGATE_INKLING_ATTN_EARLY_META", "PYTHONPATH", "PYTORCH_CUDA_ALLOC_CONF")}
            env.update(LD_LIBRARY_PATH=os.path.join(self.pack, "bin"), MALLOC_ARENA_MAX="2", CUDA_VISIBLE_DEVICES=str(c))
            cmd = ([exe, self.artifact, "--host", "127.0.0.1", "--port", str(port), "--served-model-name", "inkling"] + FLAGS.split() + SPEC.split()
                   + ["--lora-modules", f"eff-policy={adapter}"])
            lf = open(os.path.join(self.logdir, f"engine-{c}-{tag}.log"), "ab")
            self.procs.append(subprocess.Popen(cmd, env=env, stdout=lf, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL))   # same session as the job: the card session counts them as its own
        t0 = time.time()
        pending = set(self.ports)
        while pending:
            for p in self.procs:
                if p.poll() is not None:
                    raise RuntimeError(f"an engine ended at start with status {p.returncode} (see {self.logdir})")
            for port in list(pending):
                try:
                    c = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
                    c.request("GET", "/v1/models")
                    r = c.getresponse()
                    body = r.read()
                    c.close()
                    if r.status == 200:
                        if "eff-policy" not in [m.get("id") for m in json.loads(body).get("data", [])]:
                            raise RuntimeError("the engine does not list the LoRA slot eff-policy")
                        pending.discard(port)
                except (OSError, ValueError):
                    pass
            if time.time() - t0 > 600:
                raise RuntimeError(f"engines on ports {sorted(pending)} did not come up in 600 s")
            time.sleep(2)
        return time.time() - t0

    def stop(self):                                           # only this process's own children
        for p in self.procs:
            if p.poll() is None:
                p.send_signal(signal.SIGTERM)
        t0 = time.time()
        for p in self.procs:
            try:
                p.wait(max(1, 40 - (time.time() - t0)))
            except subprocess.TimeoutExpired:
                p.kill()
                p.wait(20)
        self.procs = []
        time.sleep(3)


def grade(text, gold):
    """the reply must be the house digit and nothing else (spaces around it allowed): reward 1 when it is the card's house"""
    t = (text or "").strip()
    ans = t if t in ("1", "2", "3", "4") else None
    return {"reward": 1.0 if ans is not None and ans == str(gold) else 0.0, "answer_norm": ans}


def attempt_seed(base, *parts):
    return int.from_bytes(hashlib.sha256(("secret-rl:" + ":".join(str(x) for x in (base,) + parts)).encode()).digest()[:4], "little") % 2 ** 31


def render_prompts(eng, prompts):
    """prompt id -> the engine's own token ids of the chat request (effort 0.99 line inside)"""
    def one(k):
        p = prompts[k]
        body = {"model": "eff-policy", "messages": [{"role": "user", "content": p["prompt"]}], "max_tokens": 1, **PINS, "chat_template_kwargs": dict(CTK),
                "logprobs": True, "return_token_ids": True, "stream": False, "seed": 1}
        r = post(eng.ports[k % len(eng.ports)], "/v1/chat/completions", body)
        ids = r.get("prompt_token_ids") or r["choices"][0].get("prompt_token_ids")
        if not isinstance(ids, list) or not ids:
            raise RuntimeError("the engine returned no prompt_token_ids")
        return p["id"], [int(x) for x in ids]
    with cf.ThreadPoolExecutor(16) as ex:
        return dict(ex.map(one, range(len(prompts))))


def sample(eng, jobs, max_tokens, concurrency=32, pins=None):
    """jobs: [{"key", "ids" (prompt + opener), "seed"}] -> per job {"C": completion ids, "lp": their log-probs, "text", "finish"}"""
    def one(k):
        j = jobs[k]
        body = {"tokens": j["ids"], "messages": [{"role": "user", "content": ""}], "max_tokens": max_tokens, **(pins or PINS), "model": "eff-policy", "logprobs": True,
                "return_token_ids": True, "stream": False, "skip_special_tokens": False, "seed": j["seed"]}
        r = post(eng.ports[k % len(eng.ports)], "/v1/chat/completions/tokens", body)
        ch = r["choices"][0]
        P = r.get("prompt_token_ids") or ch.get("prompt_token_ids")
        C = ch.get("token_ids")
        lp = [x["logprob"] for x in (ch.get("logprobs") or {}).get("content") or []]
        if P is not None and [int(x) for x in P] != j["ids"]:
            raise RuntimeError(f"the engine's prompt ids ({len(P)}) differ from the {len(j['ids'])} sent")
        if not isinstance(C, list) or not C or len(lp) != len(C) or not all(np.isfinite(lp)):
            raise RuntimeError(f"malformed reply: {len(C) if isinstance(C, list) else C} completion ids, {len(lp)} log-probs")
        msg = ch.get("message") or {}
        return {"C": [int(x) for x in C], "lp": [float(x) for x in lp], "text": msg.get("content") or "", "finish": ch.get("finish_reason")}
    with cf.ThreadPoolExecutor(concurrency) as ex:
        return list(ex.map(one, range(len(jobs))))


DIGIT = {"1": 16, "2": 17, "3": 18, "4": 19}                   # the house digits' token ids
END_MESSAGE = 200010


def sample_chat(eng, jobs, max_tokens, pins, concurrency=None):
    """a whole reply as served (thinking on) through the chat route. jobs: [{"prompt", "seed"}] -> {"P": prompt ids, "C", "lp", "text", "finish"}"""
    def one(k):
        j = jobs[k]
        body = {"model": "eff-policy", "messages": [{"role": "user", "content": j["prompt"]}], "max_tokens": max_tokens, **pins, "chat_template_kwargs": dict(CTK),
                "logprobs": True, "return_token_ids": True, "stream": False, "seed": j["seed"]}
        r = post(eng.ports[k % len(eng.ports)], "/v1/chat/completions", body, timeout=3600)
        ch = r["choices"][0]
        P = [int(x) for x in (r.get("prompt_token_ids") or ch.get("prompt_token_ids"))]
        C = [int(x) for x in ch["token_ids"]]
        lp = [float(x["logprob"]) for x in ch["logprobs"]["content"]]
        if not C or len(lp) != len(C) or not all(np.isfinite(lp)):
            raise RuntimeError(f"malformed reply: {len(C)} completion ids, {len(lp)} log-probs")
        return {"P": P, "C": C, "lp": lp, "text": (ch.get("message") or {}).get("content") or "", "finish": ch.get("finish_reason")}
    with cf.ThreadPoolExecutor(concurrency or 8 * len(eng.ports)) as ex:
        return list(ex.map(one, range(len(jobs))))


def make_row(Row, ids, C, lp, version, trial, group, adv):
    T = len(ids) + len(C)
    full = np.asarray(ids + C, dtype=np.int64)
    mask = np.zeros(T, dtype=bool)
    mask[len(ids):] = True
    elp = np.zeros(T, dtype=np.float64)
    elp[len(ids):] = np.asarray(lp, dtype=np.float64)
    ver = np.full(T, -1, dtype=np.int32)
    ver[len(ids):] = version
    return Row(ids=full, loss_mask=mask, engine_lp=elp, advantage=float(adv), versions=ver, teacher_lp=None, routed=None, trial_idx=int(trial), group_idx=int(group), notes={})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True); ap.add_argument("--pack", required=True); ap.add_argument("--artifact", required=True); ap.add_argument("--trainer-bin", required=True)
    ap.add_argument("--trainer-cards", required=True); ap.add_argument("--engine-cards", required=True)
    ap.add_argument("--steps", type=int, default=40); ap.add_argument("--samples", type=int, default=8); ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--shape", default="expert-b", choices=("expert-b", "attn64-ab"), help="expert-b: expert LoRA r16, B only; attn64-ab: rank-64 attention adapters (alpha 32) + expert A and B")
    ap.add_argument("--stop-pass", type=float, default=0.0, help="leave the step loop for the end reads once the mean on-policy pass of the last --stop-window steps reaches this")
    ap.add_argument("--stop-window", type=int, default=5)
    ap.add_argument("--roll-top-p", type=float, default=1.0, help="top_p of the TRAINING rollouts (the reads keep the served pins); 1.0 with --roll-top-k 0 = untruncated")
    ap.add_argument("--roll-top-k", type=int, default=0)
    ap.add_argument("--thinking", action="store_true", help="whole replies as served (thinking on) through the chat route; loss on every sampled token; cut replies are left out of the update")
    ap.add_argument("--cap", type=int, default=8192, help="reply cap in tokens (--thinking)"); ap.add_argument("--n-engines", type=int, default=0, help="> 0: the last N of all cards given serve rollouts")
    ap.add_argument("--prompts-per-step", type=int, default=32); ap.add_argument("--max-hours", type=float, default=0.0)
    ap.add_argument("--futility-step", type=int, default=0); ap.add_argument("--futility-pass", type=float, default=0.32)
    ap.add_argument("--sft", action="store_true", help="the supervised control: train on the gold reply of each prompt (advantage +1, ratio 1) instead of the sampled replies")
    ap.add_argument("--seed", type=int, default=20261127); ap.add_argument("--heldout-n", type=int, default=512); ap.add_argument("--max-tokens", type=int, default=2)
    ap.add_argument("--damage-n", type=int, default=16); ap.add_argument("--damage-max-tokens", type=int, default=768); ap.add_argument("--K", type=int, default=48)
    a = ap.parse_args()
    global FLAGS
    if a.shape == "attn64-ab":
        FLAGS = FLAGS.replace("--max-lora-rank 16", "--max-lora-rank 64")
    roll_pins = {"temperature": 1.0, "top_p": a.roll_top_p, "top_k": a.roll_top_k}
    if a.thinking and a.sft:
        raise SystemExit("--thinking and --sft do not go together")
    if a.n_engines:
        allc = a.trainer_cards.split(",") + a.engine_cards.split(",")
        a.trainer_cards, a.engine_cards = ",".join(allc[:-a.n_engines]), ",".join(allc[-a.n_engines:])
    t_start = time.time()
    sys.path.insert(0, a.pack)
    from surogate.inkling_rl.records import Row
    import t8_skyrl as SK
    import t8_step as TS

    S = a.run_dir
    os.makedirs(S, exist_ok=True)
    split = os.path.join(a.pack, "split")
    train = {json.loads(x)["id"]: json.loads(x) for x in open(os.path.join(split, "train.jsonl"))}
    blocks = json.load(open(os.path.join(split, "schedule.json")))["blocks"]
    held = [json.loads(x) for x in open(os.path.join(split, "heldout.jsonl"))][:a.heldout_n]
    dmg_file = os.path.join(split, "damage.jsonl")
    dmg_prompts = [json.loads(x) for x in open(dmg_file)][:a.damage_n] if os.path.exists(dmg_file) else []
    if a.steps > len(blocks):
        raise SystemExit(f"{a.steps} steps but the schedule has {len(blocks)} blocks")
    tcards = tuple(int(x) for x in a.trainer_cards.split(","))
    eng = Engines(a.pack, a.artifact, [int(x) for x in a.engine_cards.split(",")], os.path.join(S, "logs"))
    jdump(os.path.join(S, "run.json"), {"label": LABEL, "args": vars(a), "time": time.strftime("%F %T", time.gmtime(time.time() + 7200)), "opener": OPENER,
                                         "what": "secret-rule RL proof: reply without a thinking segment (forced opener), reward = exact house, GRPO + SkyRL PPO via T8Trainer.ppo_batch"})
    tr = None
    try:
        # ---------------------------------------------------------------- phase 0: the base (zero adapter)
        zero = os.path.join(a.pack, "zero")
        log(f"engines up with the zero adapter in {eng.start(zero, 'zero'):.0f} s")
        used = [train[i] for v in range(a.steps) for i in blocks[v]]
        ids = render_prompts(eng, used + held)
        log(f"rendered {len(ids)} prompts; lengths {min(map(len, ids.values()))}-{max(map(len, ids.values()))} tokens")

        def read_heldout(tag):
            if a.thinking:
                res = sample_chat(eng, [{"prompt": p["prompt"], "seed": attempt_seed(a.seed, "heldout", k)} for k, p in enumerate(held)], a.cap, PINS)
            else:
                jobs = [{"ids": ids[p["id"]] + OPENER, "seed": attempt_seed(a.seed, "heldout", k)} for k, p in enumerate(held)]
                res = sample(eng, jobs, a.max_tokens)
            ok = [int(grade(r["text"], p["gold"])["reward"]) for r, p in zip(res, held)]
            ans = [grade(r["text"], p["gold"]).get("answer_norm") for r, p in zip(res, held)]
            out = {"tag": tag, "n": len(held), "pass": float(np.mean(ok)), "per_prompt": ok, "answers": ans, "cut": float(np.mean([r["finish"] == "length" for r in res])),
                   "mean_completion_tokens": float(np.mean([len(r["C"]) for r in res]))}
            jdump(os.path.join(S, f"heldout-{tag}.json"), out)
            log(f"held-out read [{tag}]: pass {out['pass']:.3f} over {out['n']} prompts; cut {out['cut']:.3f}; {out['mean_completion_tokens']:.1f} tokens a reply")
            return out

        h0 = read_heldout("base")
        # the damage panel's rows: base replies (thinking on, as served) to maths problems, sampled once
        def one_dmg(k):
            p = dmg_prompts[k]
            body = {"model": "eff-policy", "messages": [{"role": "user", "content": p["prompt"]}], "max_tokens": a.damage_max_tokens, **PINS, "chat_template_kwargs": dict(CTK),
                    "logprobs": True, "return_token_ids": True, "stream": False, "seed": attempt_seed(a.seed, "damage", k)}
            r = post(eng.ports[k % len(eng.ports)], "/v1/chat/completions", body)
            ch = r["choices"][0]
            P = [int(x) for x in (r.get("prompt_token_ids") or ch.get("prompt_token_ids"))]
            C = [int(x) for x in ch["token_ids"]]
            lp = [float(x["logprob"]) for x in ch["logprobs"]["content"]]
            return make_row(Row, P, C, lp, 0, 10 ** 6 + k, 10 ** 6 + k, 0.0)
        with cf.ThreadPoolExecutor(16) as ex:
            dmg_rows = list(ex.map(one_dmg, range(len(dmg_prompts))))
        log(f"damage panel: {len(dmg_rows)} base replies, {sum(int(r.loss_mask.sum()) for r in dmg_rows)} completion tokens")

        # ---------------------------------------------------------------- the trainer
        shape = {"attn_rank": 64, "attn_alpha": 32.0, "train_keys": TS.param_keys(64)} if a.shape == "attn64-ab" else {"train_keys": ("B13", "B2")}
        cfg = TS.T8Config(bin_dir=a.trainer_bin, artifact=a.artifact, cards=tcards, K=a.K, head="plain", max_row_tokens=58010, segment_keys=(),
                          run_dir=os.path.join(S, "trainer"), **shape)
        log(f"trainer: stack {cfg.stack}, cards {tcards}, K {a.K}, shape {a.shape}, train keys {cfg.train_keys}")
        tr = TS.T8Trainer(cfg)
        theta0 = [{k: v.clone() for k, v in d.items()} for d in tr.masters()]
        lp_d0 = tr.logprobs(dmg_rows) if dmg_rows else []
        bad = sum(int((np.asarray(x, np.float32)[r.loss_mask[1:]] != r.engine_lp[1:].astype(np.float32)[r.loss_mask[1:]]).sum()) for x, r in zip(lp_d0, dmg_rows))
        log(f"trainer forward at the zero adapter vs the engine's decode-time log-probs on the damage panel: {bad} loss tokens differ")
        steps = []
        adapter = zero
        for v in range(a.steps):
            t0 = time.time()
            ps = [train[i] for i in blocks[v]][:a.prompts_per_step]
            if a.thinking:
                jobs = [{"prompt": p["prompt"], "seed": attempt_seed(a.seed, "train", v, g, s)} for g, p in enumerate(ps) for s in range(a.samples)]
                res = sample_chat(eng, jobs, a.cap, roll_pins)
                for j, r in zip(jobs, res):
                    j["ids"] = r["P"]
            else:
                jobs = [{"ids": ids[p["id"]] + OPENER, "seed": attempt_seed(a.seed, "train", v, g, s)} for g, p in enumerate(ps) for s in range(a.samples)]
                res = sample(eng, jobs, a.max_tokens, pins=roll_pins)
            t_roll = time.time() - t0
            with open(os.path.join(S, f"samples-v{v}.jsonl"), "w") as f:      # every reply as sampled (ids, text, finish reason)
                for n, r in enumerate(res):
                    f.write(json.dumps({"n": n, "gold": ps[n // a.samples]["gold"], "C": (len(r["C"]) if a.thinking else r["C"]), "text": r["text"][:200], "finish": r["finish"]}) + "\n")
            rew, grp, fin = [], [], []
            for n, r in enumerate(res):
                g = n // a.samples
                rew.append(float(grade(r["text"], ps[g]["gold"])["reward"]))
                grp.append(g)
                fin.append((r["finish"] != "length") if a.thinking else (len(r["C"]) == a.max_tokens or r["finish"] != "length"))   # the reply IS its first max_tokens tokens (the digit and <|end_message|>): the cap is the design, not a cut
            if a.thinking:                                     # a cut reply has no answer: it is left out of its prompt's group, not scored as a failure
                keep = [n for n in range(len(res)) if fin[n]]
                cnt = {}
                for n in keep:
                    cnt[grp[n]] = cnt.get(grp[n], 0) + 1
                keep = [n for n in keep if cnt[grp[n]] >= 2]   # one finished reply alone has no baseline (GRPO would give it its raw reward)
                adv = [0.0] * len(res)
                for n, x in zip(keep, SK.grpo_advantages([rew[n] for n in keep], [grp[n] for n in keep]) if keep else []):
                    adv[n] = float(x)
            else:
                adv = SK.grpo_advantages(rew, grp)
            rows = [make_row(Row, jobs[n]["ids"], r["C"], r["lp"], v, v * 100000 + n, n // a.samples, adv[n]) for n, r in enumerate(res) if fin[n] and adv[n] != 0.0]
            if a.sft:                                          # the supervised control: the same prompts, the gold reply, advantage +1; old log-probs = the trainer's own forward (ratio 1)
                gold = [make_row(Row, ids[p["id"]] + OPENER, [DIGIT[p["gold"]], END_MESSAGE], [0.0, 0.0], v, v * 100000 + g, g, 1.0) for g, p in enumerate(ps)]
                rows = []
                for r, x in zip(gold, tr.logprobs(gold)):
                    n0 = len(r.ids) - 2
                    x = np.asarray(x, np.float64)
                    rows.append(make_row(Row, [int(t) for t in r.ids[:n0]], [int(t) for t in r.ids[n0:]], [float(x[n0 - 1]), float(x[n0])], v, r.trial_idx, r.group_idx, 1.0))
                rec_gold_nll = -float(np.mean([r.engine_lp[len(r.ids) - 2] for r in rows]))
            mixed = len({r.group_idx for r in rows})
            rec = {"step": v, "pass": float(np.mean(rew)), "cut": 1.0 - float(np.mean(fin)), "rollouts": len(res), "rows": len(rows), "mixed_prompts": mixed,
                   "mean_completion_tokens": float(np.mean([len(r["C"]) for r in res])), "rollout_s": t_roll,
                   "answers": {str(k): int(sum(1 for r in res if grade(r["text"], "0").get("answer_norm") == str(k))) for k in (1, 2, 3, 4)},
                   "not_a_digit": int(sum(1 for r in res if grade(r["text"], "0").get("answer_norm") is None)),
                   "pass_by_house": {str(k): [int(sum(rew[n] for n in range(len(res)) if ps[n // a.samples]["gold"] == str(k))), int(sum(1 for n in range(len(res)) if ps[n // a.samples]["gold"] == str(k)))] for k in (1, 2, 3, 4)}}
            if a.sft:
                rec["gold_digit_nll"] = rec_gold_nll
            if v == 0 and rows:
                rec["zero_adapter_checks"] = len(tr.check_zero_adapter(rows[:2]))
            if rows:
                t1 = time.time()
                outs = tr.ppo_batch(rows, v, a.lr, 1)
                rec.update(train_s=time.time() - t1, grad_norm=float(outs[0][1]), replay_checked_rows=int(outs[0][0].metrics.get("replay_checked_rows", -1)),
                           clip_ratio=outs[0][0].metrics.get("ppo_clip_ratio"))
            else:
                rec.update(train_s=0.0, grad_norm=None, note="no prompt with a pass / fail spread: no update")
            if (v + 1) % 10 == 0 and dmg_rows:                              # the damage panel on the way (teacher-forced; the trainer holds adapter v + 1)
                lp_dv = tr.logprobs(dmg_rows)
                rec["damage_nats_per_token"] = sum(float((np.asarray(x0, np.float64)[r.loss_mask[1:]] - np.asarray(x1, np.float64)[r.loss_mask[1:]]).sum())
                                                   for x0, x1, r in zip(lp_d0, lp_dv, dmg_rows)) / max(sum(int(r.loss_mask[1:].sum()) for r in dmg_rows), 1)
            adapter = os.path.join(S, "adapters", f"v{v + 1}")
            rec["adapter_sha256"] = tr.export_adapter(adapter, dtype="BF16", meta={"label": LABEL, "step": v + 1})
            eng.stop()
            rec["engine_start_s"] = eng.start(adapter, f"v{v + 1}")
            old = os.path.join(S, "adapters", f"v{v}")            # keep only the adapter the engines hold
            if os.path.isdir(old):
                shutil.rmtree(old, ignore_errors=True)
            rec["seconds"] = time.time() - t0
            steps.append(rec)
            with open(os.path.join(S, "steps.jsonl"), "a") as f:
                f.write(json.dumps(rec, default=float) + "\n")
            log(f"step {v}: on-policy pass {rec['pass']:.3f} (fresh prompts), answers {rec['answers']} not a digit {rec['not_a_digit']}, rows {rec['rows']} of "
                f"{rec['rollouts']}, mixed prompts {mixed}, grad norm {rec['grad_norm']}, damage {rec.get('damage_nats_per_token')}, gold digit nll {rec.get('gold_digit_nll')}, {rec['seconds']:.0f} s")
            if a.max_hours and (time.time() - t_start) / 3600.0 >= a.max_hours:
                log(f"budget: {a.max_hours} hours reached; going to the end reads after step {v}")
                break
            if a.futility_step and len(steps) == a.futility_step and float(np.mean([x["pass"] for x in steps[-10:]])) < a.futility_pass:
                log(f"futility: mean on-policy pass of steps {v - 9}-{v} under {a.futility_pass}; going to the end reads")
                break
            if a.stop_pass and len(steps) >= a.stop_window and float(np.mean([x["pass"] for x in steps[-a.stop_window:]])) >= a.stop_pass:
                log(f"stop line: mean on-policy pass of the last {a.stop_window} steps >= {a.stop_pass}; going to the end reads after step {v}")
                break
        # ---------------------------------------------------------------- the end reads
        h1 = read_heldout("final")
        lp_d1 = tr.logprobs(dmg_rows) if dmg_rows else []
        num = sum(float((np.asarray(x0, np.float64)[r.loss_mask[1:]] - np.asarray(x1, np.float64)[r.loss_mask[1:]]).sum()) for x0, x1, r in zip(lp_d0, lp_d1, dmg_rows))
        den = sum(int(r.loss_mask[1:].sum()) for r in dmg_rows)
        damage = num / max(den, 1)
        log(f"damage panel: {damage:.6f} nats per token over {den} tokens (line 0.010)")
        final = [{k: v.clone() for k, v in d.items()} for d in tr.masters()]
        import torch
        g = torch.Generator().manual_seed(a.seed + 7)
        ctrl = []
        for d0, d1 in zip(theta0, final):                     # a random direction with each tensor's own update norm (zero where the update is zero)
            c = {}
            for k in d0:
                delta = (d1[k].double() - d0[k].double())
                n = float(delta.norm())
                if n == 0.0:
                    c[k] = d0[k].clone()
                else:
                    z = torch.randn(delta.shape, generator=g, dtype=torch.float64)
                    c[k] = (d0[k].double() + z * (n / float(z.norm()))).float()
            ctrl.append(c)
        tr.set_masters([{k: np.array(x, np.float32) for k, x in d.items()} for d in ctrl])
        cdir = os.path.join(S, "adapters", "control")
        tr.export_adapter(cdir, dtype="BF16", meta={"label": LABEL, "control": True})
        eng.stop()
        eng.start(cdir, "control")
        hc = read_heldout("control")
        both = [(x, y) for x, y in zip(h0["per_prompt"], h1["per_prompt"])]
        rng = np.random.default_rng(a.seed)
        d = np.array([y - x for x, y in both], dtype=np.float64)
        boot = [float(d[rng.integers(0, len(d), len(d))].mean()) for _ in range(4000)]
        verdict = {"label": LABEL, "heldout_base": h0["pass"], "heldout_final": h1["pass"], "heldout_control": hc["pass"], "gain": float(d.mean()),
                   "gain_ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))], "damage_nats_per_token": damage, "damage_line": 0.010,
                   "first5_pass": float(np.mean([s["pass"] for s in steps[:5]])), "last5_pass": float(np.mean([s["pass"] for s in steps[-5:]])),
                   "steps": len(steps), "rollouts": int(sum(s["rollouts"] for s in steps)), "zero_adapter_replay_differs": bad}
        jdump(os.path.join(S, "verdict.json"), verdict)
        log("VERDICT " + json.dumps(verdict))
        return 0
    finally:
        eng.stop()
        if tr is not None:
            tr.close()


if __name__ == "__main__":
    sys.exit(main())
