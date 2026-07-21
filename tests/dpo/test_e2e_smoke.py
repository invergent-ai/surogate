"""End-to-end DPO smoke test (Tasks 6+7): `surogate dpo <yaml>` runs a real
lightweight LoRA offline DPO over a handful of minimal pairs and:
  - exits 0,
  - logs step-0 dpo_loss ~= log 2 = 0.693 (pi_theta == pi_ref at LoRA init),
  - writes a step checkpoint + a final adapter.

Needs a GPU and a model checkout/cache, so it is marked slow.
"""

import json
import os
import shutil
import subprocess
import sys

import pytest
import yaml

SUROGATE = shutil.which("surogate") or os.path.join(os.path.dirname(sys.executable), "surogate")

MODEL = os.environ.get("SUROGATE_DPO_TEST_MODEL", "Qwen/Qwen3.5-0.8B")


@pytest.mark.slow
@pytest.mark.parametrize("recipe", ["bf16", "fp8-hybrid"])
def test_dpo_end_to_end(tmp_path, recipe):
    data = tmp_path / "pairs.jsonl"
    with open(data, "w", encoding="utf-8") as f:
        for _ in range(8):
            f.write(
                json.dumps(
                    {
                        "prompt": "Scrie corect în limba română.",
                        "chosen": "Ei mergeau acasă obosiți.",
                        "rejected": "Ei mergerăm acasă obosiți.",
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    out = tmp_path / "out"
    cfg = tmp_path / "dpo.yaml"
    cfg.write_text(
        yaml.safe_dump(
            {
                "model": MODEL,
                "lora": True,
                "lora_rank": 8,
                "lora_alpha": 16,
                "lora_target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                # Both recipes must hit the step-0 identity below. fp8-hybrid is the
                # regression guard for the inline-reference fix: its current-batch
                # activation scaling only agrees with the policy forward because the
                # reference is computed in the SAME micro-batch (not precomputed offline).
                "recipe": recipe,
                "optimizer": "adamw_8bit",
                "learning_rate": 1.0e-4,  # aggressive: drive the margin off 0 within a few steps
                "lr_scheduler_type": "constant",
                "gpus": 1,
                "per_device_train_batch_size": 2,
                "gradient_accumulation_steps": 1,
                "sequence_len": 256,
                "max_steps": 24,
                "save_steps": 12,
                "logging_steps": 1,
                "output_dir": str(out),
                "surogate_metrics_path": str(out / "metrics.jsonl"),
                "loss": {"type": "dpo", "dpo_beta": 0.1, "length_norm": False},
                "datasets": [{"path": str(data), "type": "preference"}],
            }
        )
    )

    r = subprocess.run(
        [SUROGATE, "dpo", str(cfg)],
        capture_output=True,
        text=True,
        timeout=1800,
    )
    log = r.stdout + r.stderr
    assert r.returncode == 0, log[-4000:]

    metrics = [json.loads(line) for line in (out / "metrics.jsonl").read_text().splitlines() if line.strip()]

    # (1) step-0 identity: LoRA is zero-init so pi_theta == pi_ref => margin 0, dpo_loss == log 2.
    step0 = next(m for m in metrics if m["step"] == 0)
    assert abs(step0["dpo_loss"] - 0.6931) < 5e-3, step0

    # (2) the reference must be FROZEN, not tracking the policy. As the policy LoRA diverges
    # the margin must move off 0 and the loss drop below log 2. A ref-tracks-policy bug
    # (compute_ref_logprobs running the live LoRA-on forward) pins margin == 0 every step
    # forever — nonzero grad_norm but zero learning — which step-0-only checks cannot catch.
    margins = [abs(m["dpo_margin"]) for m in metrics]
    assert max(margins) > 0.05, f"DPO margin never separated from 0 (frozen ref broken?): {margins}"
    late_loss = min(m["dpo_loss"] for m in metrics if m["step"] >= len(metrics) // 2)
    assert late_loss < 0.6931, f"DPO loss never dropped below log 2 (no learning signal): {late_loss}"

    assert (out / "final_adapter").is_dir()
    # save_checkpoint writes output_dir/step_{:08} (the convention `surogate merge` consumes).
    ckpts = list(out.glob("step_*"))
    assert ckpts, "no step checkpoint written"
    assert (ckpts[0] / "adapter_model.safetensors").is_file()
