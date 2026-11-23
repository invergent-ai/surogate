"""CPU tests of the Inkling RL glue (surogate/inkling_rl). Plain asserts; run: python tests/inkling_rl/test_glue.py
(pytest also collects the test_* functions). Needs numpy, torch, msgspec, pyyaml; the kit worktree READ ONLY for its
loop module (eff_samples.to_training_sample) and its TRL reference fixture.

 1 pins          the vendored kit loss / config / types match their sha256 pins and load
 2 trl           the deployed loss through step's API == TRL 1.17.1 GRPO / DAPO gradients (the kit's own fixture)
 3 adapter       export -> import -> export byte-identical; bf16 RNE == torch's; fold at alpha / r = 2 exact
 4 records       rows built by the KIT's to_training_sample, encoded with the kit's schema, decoded by records.py
 5 e2e           fake engine (version 0 = the shim's zero adapter, engine fold) samples groups with routing capture ->
                 kit rows -> rollouts.bin -> Shim one step: engine_mismatch_kl == 0 EXACTLY, R3 counters, TIS capped 0,
                 broadcast step_1 + STABLE, the kit gate's regexes match the log; the engine loads step_1 (fold):
                 reward-weighted log-prob of in-sample AND held-out samples rises (gate 3's direction, toy reward)
 6 resume        save_steps 1: checkpoint step_00000001; a new Shim resumes at step 2, re-broadcasts step_2 with the
                 same bytes as an uninterrupted run's step_2
"""
from __future__ import annotations

import contextlib
import io
import json
import os
import re
import shutil
import sys
import tempfile

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
KIT = os.environ.get("KIT_RECIPE", "/mnt/data/work/repos/si-wt-resume/recipe")

from surogate.inkling_rl import adapter as AD  # noqa: E402
from surogate.inkling_rl import kitsrc, records  # noqa: E402
from surogate.inkling_rl.backend_fake import FakeInkling  # noqa: E402
from surogate.inkling_rl.shim import Shim, MasterAdamW, masters_layers  # noqa: E402
from surogate.inkling_rl.step import StepConfig  # noqa: E402


def kit_loop():
    sys.path.insert(0, os.path.join(KIT, "loop"))
    import eff_samples  # noqa: E402  (the kit's own row -> TrainingSample code)

    return eff_samples


def test_pins():
    cfg, fn = kitsrc.loss_api()
    TS, TB = kitsrc.types_api()
    assert callable(fn) and cfg(variant="gspo", seq_ratio_old="step_start", aggregation="token").aggregation == "token"
    assert "routed_experts" in TS.__struct_fields__ and TB.__struct_fields__ == ("examples", "step", "run_idx")


def test_trl():
    fx = np.load(os.path.join(KIT, "trainer-patches/gspo-token-aggregation/trl_reference.npz"))
    _, grads = kitsrc.loss_api()
    rows = [{k: fx[f"r{i}_{k}"] for k in ("new", "old", "beh", "mask", "adv")} for i in range(int(fx["n"]))]
    for agg, key in (("sequence", "trl_grpo"), ("token", "trl_dapo")):
        sc = StepConfig(aggregation=agg, behavior_is_cap=None)
        cfg = sc.loss_config()
        scale = len(rows) if agg == "sequence" else int(sum(r["mask"].sum() for r in rows))
        worst = 0.0
        for i, r in enumerate(rows):
            n = len(r["mask"])
            g, _ = grads(r["new"], r["old"], np.full(n, float(r["adv"])), r["mask"].astype(bool), cfg, [(0, n)])
            ref = fx[f"r{i}_{key}"]
            worst = max(worst, float(np.max(np.abs(-g / scale - ref))) / max(float(np.max(np.abs(ref))), 1e-30))
        assert worst < 1e-6, (agg, worst)


def test_adapter():
    x = np.random.default_rng(0).standard_normal(200000).astype(np.float32) * 3
    assert np.array_equal(AD.f32_to_bf16_bits(x), torch.from_numpy(x).to(torch.bfloat16).view(torch.int16).numpy().view(np.uint16))
    m = AD.init_masters(1, (0, 5), experts=4, hidden=8, inter=6, rank=2, a_std=0.05, b_std=0.01)
    d = tempfile.mkdtemp()
    s1 = AD.export_adapter(m, d + "/a", 2, 4)
    back, cfg = AD.import_adapter(d + "/a")
    assert AD.export_adapter(back, d + "/b", 2, 4) == s1
    for L in (0, 5):
        for k in AD.KEYS:
            assert np.array_equal(back[L][k].view(np.uint32), AD.round_bf16(m[L][k]).view(np.uint32))
        a, b = AD.engine_bank(back[L]["A13"], back[L]["B13"], 2.0)
        assert np.array_equal(a.view(np.uint32), (2 * back[L]["A13"]).astype(np.float32).view(np.uint32))
    shutil.rmtree(d)


def kit_sample(es, prompt, comp, comp_lp, adv, version, routed, dims):
    """One trial as the kit's loop turns it into a TrainingSample (eff_samples.to_training_sample)."""
    ids = list(prompt) + list(comp)
    mask = [0] * len(prompt) + [1] * len(comp)
    lps = [0.0] * len(prompt) + list(comp_lp)
    vers = [-1] * len(prompt) + [version] * len(comp)
    row = {"input_ids": ids, "loss_mask": mask, "behavior_logprobs": lps, "advantage": adv, "token_versions": vers,
           "routed": routed, "routed_dims": dims}
    return es.to_training_sample(row, versions=True)


def test_records():
    es = kit_loop()
    TS, TB = kitsrc.types_api()
    T, R, K = 7, 48, 8
    routed = np.arange(T * R * K, dtype=np.int16).reshape(T, R, K) % 128
    routed[-1] = -1
    s = kit_sample(es, [1, 2, 3], [4, 5, 6, 7], [-0.5, -0.25, -1.0, -2.0], 0.75, 3, routed.astype("<i2").tobytes(), [R, K])
    b = TB(examples=[TS(**s)], step=5)
    dec = records.decode_batch(records.encode_batch(b))
    r = records.row_from_sample(dec.examples[0])
    assert dec.step == 5 and r.T == 7 and r.loss_mask.tolist() == [False] * 3 + [True] * 4
    assert r.engine_lp[3:].tolist() == [-0.5, -0.25, -1.0, -2.0] and r.versions.tolist() == [-1] * 3 + [3] * 4
    assert np.array_equal(r.routed, routed)
    rp, c = records.replay_rows(r)
    assert c["r3_rows"] == 1 and c["r3_replayed_tokens"] == 6 and c["r3_uncovered_interior"] == 0
    assert records.replay_rows(r, 40, 8)[1]["r3_shape_mismatches"] == 1


def _cfg(tmp, max_steps, save_steps=3):
    return {"output_dir": os.path.join(tmp, "train"), "save_steps": save_steps, "max_steps": max_steps,
            "learning_rate": 5e-3, "max_grad_norm": 1.0, "weight_decay": 0.0, "optimizer": "adamw", "lora_rank": 4,
            "lora_alpha": 8, "max_policy_lag": 1, "resume_from_checkpoint": True,
            "loss": {"variant": "gspo", "seq_ratio_old": "step_start", "seq_clip_low": 3e-4, "seq_clip_high": 4e-4,
                     "teacher_tau": 0.3, "behavior_is_cap": 2.0, "aggregation": "token", "behavior_is_level": "token"},
            "surogate_metrics_path": os.path.join(tmp, "train", "metrics.jsonl"),
            "inkling_rl": {"backend": "fake", "run_dir": os.path.join(tmp, "run_eff"), "poll_s": 0.05,
                           "wait_timeout_s": 2, "engine_release_sha": "fake-engine-sha"}}


def reward(comp, V):
    return float(np.mean([t < V // 4 for t in comp]))    # toy verifiable reward: share of "low" tokens


def make_batch(es, eng, adapter_dir, version, step, prompts, n, new, seed):
    """Groups of n samples per prompt from the fake ENGINE with the adapter as the engine folds it. Returns the
    TrainingBatch bytes of the first half of every group (the gradient set) and the held-out half as (prompt, comp, A)."""
    TS, TB = kitsrc.types_api()
    lora, sc = eng.engine_lora_from_dir(adapter_dir)
    g = torch.Generator().manual_seed(seed)
    ex, held = [], []
    for pi, p in enumerate(prompts):
        draws = [eng.sample(p, new, lora, sc, g) for _ in range(2 * n)]
        for half, sl in ((0, draws[:n]), (1, draws[n:])):
            rw = np.array([reward(c, eng.V) for c, _, _ in sl])
            adv = rw - rw.mean()
            for (c, lps, cap), a in zip(sl, adv):
                if half == 0:
                    T = len(p) + len(c)
                    routed = np.full((T, eng.visits, eng.K), -1, dtype="<i2")
                    routed[: cap.shape[0]] = cap
                    s = kit_sample(es, p, c, lps, float(a), version, routed.tobytes(), [eng.visits, eng.K])
                    s["trial_idx"], s["group_idx"] = len(ex), pi
                    ex.append(TS(**s))
                else:
                    held.append((p, c, float(a)))
    return records.encode_batch(TB(examples=ex, step=step)), held


def write_batch(run_dir, step, data, sha="fake-engine-sha"):
    d = os.path.join(run_dir, "rollouts", f"step_{step}")
    os.makedirs(d, exist_ok=True)
    if sha is not None:
        open(os.path.join(d, "ENGINE_SHA"), "w").write(sha + "\n")    # as the kit branch inkling-rl's loop writes it
    open(os.path.join(d, "rollouts.bin"), "wb").write(data)


def dM(eng, adapter_old, adapter_new, items):
    """sum_i A_i (logp_new(y_i) - logp_old(y_i)) under the ENGINE's folded adapters (free routing)."""
    tot = 0.0
    for p, c, a in items:
        ids = torch.as_tensor(list(p) + list(c))
        vals = []
        for ad in (adapter_old, adapter_new):
            lora, sc = eng.engine_lora_from_dir(ad)
            saved = eng.scale
            eng.scale = sc
            with torch.no_grad():
                lp, _ = eng.forward(ids, lora=lora)
            eng.scale = saved
            vals.append(float(lp[len(p):].sum()))
        tot += a * (vals[1] - vals[0])
    return tot


GATE = {"STEP_RE": re.compile(r"step=(\d+) loss=\S+ grad_norm=\S+ lr=\S+ kl=(\S+) masked=\S+ tokens=(\d+) micro_batches=(\d+) expected=\S+ time=(\S+)s"),
        "DPP_RE": re.compile(r"dispatch-PP step (\d+): (\d+) rows, (\d+) tokens"),
        "FP32_RE": re.compile(r"Full-precision AdamW optimizer: \d+ tensors, (\d+) params, ([\d.]+) MB state \(FP32 m\+v\)")}


def test_e2e_and_resume():
    es = kit_loop()
    tmp = tempfile.mkdtemp()
    cfg = _cfg(tmp, max_steps=1)
    eng = FakeInkling(seed=0, rank=4, alpha=8)                 # the "engine": same base weights as the trainer's backend
    # version 0 = the trainer's zero adapter (its initial masters, B = 0)
    trainer0 = FakeInkling(seed=0, rank=4, alpha=8)
    opt0 = MasterAdamW(trainer0.parameters(), lr=0.0)
    v0 = os.path.join(tmp, "v0")
    AD.export_adapter(masters_layers(trainer0, opt0), v0, 4, 8, "BF16")
    prompts = [[1, 2, 3], [4, 5], [6, 7, 8, 9]]
    data, held = make_batch(es, eng, v0, 0, 0, prompts, n=6, new=6, seed=1)
    write_batch(cfg["inkling_rl"]["run_dir"], 0, data)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        Shim(cfg, backend=FakeInkling(seed=0, rank=4, alpha=8)).run()
    out = buf.getvalue()
    for k, rx in GATE.items():
        assert rx.search(out), (k, out[-2000:])
    met = [json.loads(x) for x in open(cfg["surogate_metrics_path"])]
    m = met[0]
    assert m["train/engine_mismatch_kl"] == 0.0 and m["train/engine_log_ratio"] == 0.0, m
    assert m["train/engine_behavior_w_capped"] == 0.0 and m["train/engine_behavior_w_mean"] == 1.0
    assert m["train/r3_rows"] == 2 * 18 and m["train/r3_uncovered_interior"] == 0 and m["train/r3_bad_rows"] == 0
    assert m["engine_release_sha"] == "fake-engine-sha"
    b1 = os.path.join(cfg["inkling_rl"]["run_dir"], "broadcasts", "step_1")
    assert os.path.exists(os.path.join(b1, "STABLE"))
    # gate 3's direction on the toy: the engine with step_1 vs version 0
    gtrain = []
    dec = records.decode_batch(data)
    for s in dec.examples:
        gtrain.append((s.prompt_ids, s.completion_ids, float(s.advantage)))
    d_in, d_ho = dM(eng, v0, b1, gtrain), dM(eng, v0, b1, held)
    assert d_in > 0, d_in
    print(f"test_e2e: mismatch 0 exactly; dM in-sample {d_in:+.4f}, held-out {d_ho:+.4f}")
    # resume: save every step, two steps; a new Shim on the same dirs resumes from step_00000001 and re-broadcasts
    # step_2 with the same bytes as the uninterrupted run
    t2 = os.path.join(tmp, "straight")
    c2 = _cfg(t2, max_steps=2, save_steps=1)
    write_batch(c2["inkling_rl"]["run_dir"], 0, data)
    write_batch(c2["inkling_rl"]["run_dir"], 1, data)          # the same rows again (version 0: inside lag 1 of step 1)
    with contextlib.redirect_stdout(io.StringIO()):
        Shim(c2, backend=FakeInkling(seed=0, rank=4, alpha=8)).run()
    st = os.path.join(t2, "run_eff", "broadcasts", "step_2", "STABLE")
    s_a = open(st).read()
    assert os.path.exists(os.path.join(t2, "train", "step_00000001", "checkpoint.json"))
    assert not os.path.exists(os.path.join(t2, "train", "step_00000000"))   # the kit saves only at step > 0
    os.remove(st)
    with contextlib.redirect_stdout(io.StringIO()):
        sh = Shim(c2, backend=FakeInkling(seed=0, rank=4, alpha=8))
        assert sh.start_step == 2, sh.start_step
        sh.run()
    assert open(st).read() == s_a
    met = [json.loads(x) for x in open(c2["surogate_metrics_path"])]
    assert met[1]["train/offpolicy_dropped_samples"] == 0 and met[1]["train/engine_mismatch_kl"] > 0   # step 1 trained on v0 rows
    # engine identity: a batch from another release, or without ENGINE_SHA, is refused
    for sha, tag in (("other-sha", "wrong"), (None, "missing")):
        t3 = os.path.join(tmp, tag)
        c3 = _cfg(t3, max_steps=1)
        write_batch(c3["inkling_rl"]["run_dir"], 0, data, sha=sha)
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                Shim(c3, backend=FakeInkling(seed=0, rank=4, alpha=8)).run()
            raise AssertionError(f"{tag} ENGINE_SHA not refused")
        except SystemExit as e:
            assert "REFUSED" in str(e), e
    shutil.rmtree(tmp)


def test_lag_rule():
    """R3 coverage rule: uncovered interior positions run the trainer's own routing, which equals the engine's only for
    rows of the step's own version; a lagged row with uncovered positions is refused, an on-policy one is trained."""
    from surogate.inkling_rl.step import run_step
    eng = FakeInkling(seed=0, rank=4, alpha=8)
    ids = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=np.int64)
    with torch.no_grad():
        lp, cap = eng.forward(torch.as_tensor(ids))
    routed = cap.numpy().astype(np.int16)
    routed[:3] = -1                                  # a cached prefix the engine returned no rows for
    routed[-1] = -1
    mask = np.array([0, 0, 0, 0, 1, 1, 1, 1], bool)

    def row(version):
        return records.Row(ids=ids, loss_mask=mask.copy(), engine_lp=np.where(mask, lp.double().numpy(), 0.0),
                           advantage=0.5, versions=np.where(mask, version, -1).astype(np.int32), teacher_lp=None,
                           routed=routed.copy(), trial_idx=0, group_idx=0)

    for version, refused in ((1, 0), (0, 1)):
        res = run_step(FakeInkling(seed=0, rank=4, alpha=8), [row(version)], 1, StepConfig(), log=lambda *_: None)
        assert res.metrics["r3_lagged_uncovered_refused_rows"] == refused, (version, res.metrics)
        if not refused:
            assert res.metrics["engine_mismatch_kl"] == 0.0     # own routing on the uncovered prefix == the engine's


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"{name}: PASS", flush=True)
