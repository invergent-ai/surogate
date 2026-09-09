"""A GRPO run, end to end, against this repository's own serving engine.

The unit suites cover the engine's surface; this covers the thing that actually has
to work -- rollouts generated, rewards scored, a policy step taken, the adapter
handed back to the sampler, and the whole pipeline exiting cleanly. Every failure
this test would have caught was silent in some way: a refused adapter reload killed
the run at its first policy update, a token budget too small for a thinking model
made every reward zero so the trainer never stepped, an orchestrator writing its
batches to a different directory than the trainer read them from left both GPUs
idle forever, and a finished run exited 134 because a daemon thread held the stderr
lock at interpreter shutdown.

Needs two GPUs and a checkpoint, so it skips unless SUROGATE_TEST_MODEL names one.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]
_MODEL = os.environ.get("SUROGATE_TEST_MODEL", "")
_CLI = _ROOT / ".venv" / "bin" / "surogate"
# One card serves, the other trains. Override for a busy machine.
_GPUS = os.environ.get("SUROGATE_TEST_GRPO_GPUS", "0,1").split(",")

needs_run = pytest.mark.skipif(
    not _MODEL or not Path(_MODEL).exists() or not _CLI.is_file() or len(_GPUS) != 2,
    reason="set SUROGATE_TEST_MODEL to a checkpoint; needs .venv/bin/surogate and two GPUs",
)


def _write_configs(root: Path) -> tuple[Path, Path, Path]:
    """A run small enough to be a test and complete enough to be one."""
    out = root / "out"
    infer = root / "infer.yaml"
    infer.write_text(
        "backend: surogate\n"
        f"model: {_MODEL}\n"
        "enable_lora: true\nmax_lora_rank: 32\nmax_loras: 4\n"
        "max_model_len: 2048\nmax_num_seqs: 16\nport: 8117\n"
    )
    orch = root / "orch.yaml"
    orch.write_text(
        f"model:\n  name: {_MODEL}\n  lora_adapter: default\n  lora_rank: 16\n  lora_alpha: 32\n"
        "env:\n  - id: markdown-table-qa\n"
        "batch_size: 4\nrollouts_per_example: 4\nseq_len: 2048\nmax_steps: 2\n"
        "use_token_client: true\n"
        # Every sampling field, so the run exercises the whole surface.
        "sampling:\n  max_tokens: 512\n  min_tokens: 8\n  temperature: 1.0\n  top_p: 1.0\n"
        "  repetition_penalty: 1.05\n  seed: 99\n"
        "  extra_body:\n    top_k: -1\n    min_p: 0.0\n"
        "    chat_template_kwargs:\n      enable_thinking: true\n"
        "client:\n  base_url:\n    - http://localhost:8117/v1\n"
        # Must be the trainer's output_dir plus its run name, or neither side errors
        # and the run hangs waiting for a policy update that never arrives.
        f"output_dir: {out}/run_default\n"
    )
    train = root / "train.yaml"
    train.write_text(
        f"model: {_MODEL}\noutput_dir: {out}\n"
        "per_device_train_batch_size: 1\nsequence_len: 2048\nmax_steps: 2\nlogging_steps: 1\n"
        "learning_rate: 1e-4\nlr_scheduler_type: constant\nwarmup_steps: 0\n"
        "max_grad_norm: 1.0\nweight_decay: 0.01\noptimizer: adamw\nrecipe: bf16\n"
        "lora: true\nlora_rank: 16\nlora_alpha: 32\n"
        "lora_target_modules:\n"
        + "".join(f"  - {m}\n" for m in
                 ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"))
        + f"save_steps: 100\ncheckpoint_dir: {out}/checkpoints\n"
    )
    return train, infer, orch


@needs_run
def test_grpo_run_completes_and_trains(tmp_path):
    train, infer, orch = _write_configs(tmp_path)
    completed = subprocess.run(
        [str(_CLI), "grpo", "--train", str(train), "--infer", str(infer), "--orch", str(orch),
         "--infer-gpus", _GPUS[0], "--trainer-gpus", _GPUS[1]],
        cwd=str(_ROOT), capture_output=True, text=True, timeout=2400,
    )
    log = completed.stdout + completed.stderr
    plain = re.sub(r"\x1b\[[0-9;]*m", "", log)

    assert completed.returncode == 0, (
        f"the run exited {completed.returncode}; a completed run must also exit cleanly\n"
        f"{plain[-3000:]}"
    )
    assert "Orchestrator finished" in plain, f"the orchestrator did not finish\n{plain[-3000:]}"

    # The trainer must actually have stepped. Rollouts that all truncate score zero,
    # every advantage is zero, and a run can "succeed" having learned nothing.
    steps = re.findall(r"step=(\d+) loss=", plain)
    assert steps, f"the trainer never took a step\n{plain[-3000:]}"

    # The adapter reload is the operation the loop performs every step; a refusal
    # there used to end the run at its first policy update.
    assert "400 Bad Request" not in plain, f"an admin call was refused\n{plain[-3000:]}"
    for symptom in ("incomplete UTF-8", "json.exception"):
        assert symptom not in plain, f"requests failed with {symptom}\n{plain[-3000:]}"

    adapter = tmp_path / "out" / "final_adapter" / "adapter_model.safetensors"
    assert adapter.is_file(), "no final adapter was written"
    config = json.loads((tmp_path / "out" / "final_adapter" / "adapter_config.json").read_text())
    assert config["r"] == 16
    # Including the fused pair, which the engine refused to serve until recently.
    assert {"gate_proj", "up_proj"} <= set(config["target_modules"])
