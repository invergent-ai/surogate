"""Opt-in real GRPO run: shared weights, nonzero updates, and alternating phases."""

import json
import os
from pathlib import Path
import re
import subprocess

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
