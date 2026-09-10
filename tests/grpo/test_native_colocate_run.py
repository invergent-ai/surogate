"""Opt-in real GRPO run: shared weights, nonzero updates, and alternating phases."""

import json
import os
import re
import subprocess
from pathlib import Path

import pytest
import yaml

MODEL = os.environ.get("SUROGATE_SHARED_MODEL", "")
ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.skipif(not MODEL, reason="set SUROGATE_SHARED_MODEL and CUDA_VISIBLE_DEVICES to one test GPU")
def test_native_colocate_trains_without_reloading_base(tmp_path):
    targets = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    out = tmp_path / "out"
    train = dict(model=MODEL, output_dir=str(out), gpus=1, per_device_train_batch_size=1,
                 sequence_len=2048, max_steps=2, logging_steps=1, learning_rate=1e-4,
                 lr_scheduler_type="constant", warmup_steps=0, recipe="bf16", optimizer="adamw",
                 lora=True, lora_rank=8, lora_alpha=16,
                 lora_target_modules=targets,
                 save_steps=0, merge_adapter=False)
    infer = dict(model=MODEL, backend="surogate", enable_lora=True, max_model_len=2048,
                 max_num_seqs=4, port=int(os.environ.get("SUROGATE_SHARED_PORT", "18652")))
    orch = dict(model=dict(name=MODEL, lora_adapter="default", lora_rank=8, lora_alpha=16),
                env=[dict(id="markdown-table-qa")], batch_size=4, rollouts_per_example=4,
                sequence_len=2048, max_steps=2, use_token_client=True,
                sampling=dict(max_tokens=512, min_tokens=8, temperature=1.0, top_p=1.0,
                              seed=99, extra_body=dict(top_k=-1, min_p=0.0,
                                                      chat_template_kwargs=dict(enable_thinking=True))),
                output_dir=str(out / "run_default"))
    argv = [str(ROOT / ".venv/bin/surogate"), "grpo-colocate"]
    for name, config in (("train", train), ("infer", infer), ("orch", orch)):
        path = tmp_path / f"{name}.yaml"
        path.write_text(yaml.safe_dump(config))
        argv.extend([f"--{name}", str(path)])
    log_path = tmp_path / "run.log"
    with log_path.open("w") as stream:
        completed = subprocess.run(argv, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, text=True, timeout=900)
    log = log_path.read_text()
    assert completed.returncode == 0, log[-18000:]
    rows = [json.loads(line) for line in (out / "shared_weights.jsonl").read_text().splitlines()]
    assert [(r["phase"], r["policy_version"]) for r in rows] == [
        ("rollouts", 0), ("training", 0), ("rollouts", 1), ("training", 1), ("rollouts", 2)]
    assert all(r["base_upload_bytes"] == 0 and r["serving_base_allocated_bytes"] == 0 for r in rows)
    assert len({r["shared_base_bytes"] for r in rows}) == 1
    assert rows[0]["shared_base_bytes"] > 0
    assert all(r["sleeping"] for r in rows if r["phase"] == "training")
    # Per-step publication is coordination only; final adapters/checkpoints remain normal files.
    assert not list((out / "run_default/broadcasts").rglob("*.safetensors"))
    adapters = list(out.rglob("adapter_model.safetensors"))
    assert adapters, log[-10000:]
    from safetensors.torch import load_file
    tensors = load_file(str(adapters[-1]))
    assert any(t.float().abs().max().item() > 0 for name, t in tensors.items() if "lora_B" in name)
    for module in targets:
        assert any(t.float().abs().max().item() > 0 for name, t in tensors.items()
                   if name.endswith(f".{module}.lora_B.weight")), module
    assert "Traceback" not in re.sub(r"\x1b\[[0-9;]*m", "", log)


@pytest.mark.skipif(not MODEL, reason="set SUROGATE_SHARED_MODEL and CUDA_VISIBLE_DEVICES to one test GPU")
def test_native_colocate_resumes_complete_updates(tmp_path):
    # A deterministic group reward keeps this integration fixture independent
    # of the tiny test model's task accuracy, while producing real gradients.
    (tmp_path / "native_resume_fixture.py").write_text("""
import verifiers as vf
from datasets import Dataset

def reward(completions):
    return [i / max(1, len(completions) - 1) for i in range(len(completions))]

def load_environment():
    data = Dataset.from_list([dict(question="Write a short sentence about item " + str(i), answer="") for i in range(32)])
    return vf.SingleTurnEnv(dataset=data, rubric=vf.Rubric(funcs=[reward]))
""")
    out = tmp_path / "out"
    train = dict(model=MODEL, output_dir=str(out), gpus=1, per_device_train_batch_size=1,
                 sequence_len=256, max_steps=2, logging_steps=1, learning_rate=1e-4,
                 lr_scheduler_type="constant", warmup_steps=0, recipe="bf16", optimizer="adamw",
                 lora=True, lora_rank=8, lora_alpha=16,
                 lora_target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                 save_steps=1, merge_adapter=False)
    infer = dict(model=MODEL, backend="surogate", enable_lora=True, max_model_len=256,
                 max_num_seqs=4, port=int(os.environ.get("SUROGATE_SHARED_PORT", "18653")))
    orch = dict(model=dict(name=MODEL, lora_adapter="default", lora_rank=8, lora_alpha=16),
                env=[dict(id="native-resume-fixture")], batch_size=4, rollouts_per_example=4,
                sequence_len=256, max_steps=2, use_token_client=True,
                sampling=dict(max_tokens=16, min_tokens=8, temperature=1.0, top_p=1.0),
                output_dir=str(out / "run_default"))
    env = os.environ | {"PYTHONPATH": str(tmp_path) + os.pathsep + str(ROOT),
                        "HF_HUB_OFFLINE": "1", "WANDB_MODE": "disabled"}
    for end_step in (2, 3):
        train["max_steps"] = orch["max_steps"] = end_step
        argv = [str(ROOT / ".venv/bin/surogate"), "grpo-colocate"]
        for name, config in (("train", train), ("infer", infer), ("orch", orch)):
            path = tmp_path / f"{name}.yaml"
            path.write_text(yaml.safe_dump(config))
            argv.extend([f"--{name}", str(path)])
        with (tmp_path / f"run_{end_step}.log").open("w") as stream:
            result = subprocess.run(argv, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT, timeout=900)
        log = (tmp_path / f"run_{end_step}.log").read_text()
        assert result.returncode == 0, log[-24000:]
        checkpoint = out / "native_colocate" / f"step_{end_step}"
        manifest = json.loads((checkpoint / "manifest.json").read_text())
        assert manifest["next_step"] == end_step and manifest["trainer_step"] == end_step - 1
        from safetensors.torch import load_file
        tensors = load_file(str(checkpoint / "trainer" / f"step_{end_step - 1:08d}" / "lora_optimizer.safetensors"))
        assert all(t.abs().max() > 0 for t in tensors.values())
        if end_step == 2:
            (out / "run_default/control/stale_fixture").write_text("preserve this later work")
        else:
            assert "Resuming native GRPO after 2 completed updates" in log
            assert not (out / "run_default/control/stale_fixture").exists()
            assert list((out / "run_default/recovery").rglob("stale_fixture"))
    rows = [json.loads(line) for line in (out / "shared_weights.jsonl").read_text().splitlines()]
    assert [(r["phase"], r["policy_version"]) for r in rows] == [
        ("rollouts", 0), ("training", 0), ("rollouts", 1), ("training", 1), ("rollouts", 2),
        ("rollouts", 2), ("training", 2), ("rollouts", 3)]
