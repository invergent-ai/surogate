"""Shared-policy publication and cancellation must unblock the next owner."""

import asyncio
import json
import threading
from types import SimpleNamespace

import pytest

from surogate.grpo.native_colocate import SharedPolicy, _run


@pytest.mark.parametrize(
    "config,expected",
    [
        ({"model_type": "glm5_next", "text_config": {}}, True),
        ({"model_type": "glm5_next_text"}, True),
        ({"model_type": "llama"}, False),
        ({"model_type": "qwen3_5_moe_text", "num_experts": 4}, False),
        ({"model_type": "gpt_oss", "num_local_experts": 4}, False),
        ({"model_type": "laguna", "n_routed_experts": 4}, False),
    ],
)
def test_policy_parity_is_selected_before_allocating_the_trainer(tmp_path, monkeypatch, config, expected):
    from surogate.grpo import native_colocate, trainer

    (tmp_path / "config.json").write_text(json.dumps(config))
    train = SimpleNamespace(
        model_dir=tmp_path,
        sequence_len=512,
        lora_rank=8,
        lora_target_modules=["all"],
        runtime_config=SimpleNamespace(glm_rollout_parity=False),
        output_dir=tmp_path / "out", checkpoint_dir=tmp_path / "out", save_steps=1, max_steps=2,
        model="fixture", lora_alpha=16, lora_dtype="bf16", resume_from_checkpoint=True,
    )
    infer = SimpleNamespace(max_model_len=512, max_num_seqs=2, host="127.0.0.1", port=8000)
    orch = SimpleNamespace(output_dir=tmp_path / "out/run_default", model=SimpleNamespace(name="glm"), client=SimpleNamespace(), ckpt=None)

    def allocate(actual, **kwargs):
        assert actual.runtime_config.glm_rollout_parity is expected
        text = config.get("text_config", config)
        moe = bool(text.get("num_experts", text.get("num_local_experts", text.get("n_routed_experts", 0))))
        assert getattr(actual.runtime_config, "moe_rollout_parity", False) is moe
        raise RuntimeError("allocation reached")

    monkeypatch.setattr(native_colocate, "validate_configs", lambda *args: None)
    monkeypatch.setattr(native_colocate, "shared_execution", lambda *args: "training")
    monkeypatch.setattr(trainer, "GRPOTrainer", allocate)
    with pytest.raises(RuntimeError, match="allocation reached"):
        native_colocate.grpo_native_colocate(train, infer, orch)


def test_publication_releases_the_batch_receiver_and_rejects_stale_versions(tmp_path, monkeypatch):
    class Server:
        version = -1

        def publish(self, name, modules, version):
            self.version = version

        def begin_training(self):
            pass

        def summary(self):
            return dict(
                policy_version=self.version, shared_base_bytes=1234, serving_base_allocated_bytes=0, base_upload_bytes=0
            )

    manager = SimpleNamespace(used_idxs={0}, ready_to_update=[True])
    monkeypatch.setattr("surogate.grpo.runs.get_multi_run_manager", lambda: manager)
    monkeypatch.setattr("surogate.grpo.native_colocate.adapter_modules", lambda trainer, scale: [])
    train = SimpleNamespace(lora_alpha=16, lora_rank=8, output_dir=tmp_path)
    orch = SimpleNamespace(model=SimpleNamespace(lora_adapter="policy"), output_dir=tmp_path / "run_default")
    policy = SharedPolicy(Server(), train, orch)
    policy.broadcast(None, 0)
    policy.begin_training(0)
    with pytest.raises(RuntimeError, match="not the published"):
        asyncio.run(policy.acknowledge(None, "policy", 0))
    policy.broadcast(None, 1)
    assert manager.ready_to_update == [False]
    assert (tmp_path / "run_default/broadcasts/step_1/STABLE").is_file()
    asyncio.run(policy.acknowledge(None, "policy", 1))
    with pytest.raises(RuntimeError, match="not the published"):
        asyncio.run(policy.acknowledge(None, "policy", 0))
    with pytest.raises(RuntimeError, match="does not match"):
        policy.begin_training(0)
    assert not list(tmp_path.rglob("*.safetensors"))


@pytest.mark.parametrize("failing_side", ["trainer", "orchestrator"])
def test_component_failure_stops_the_other_before_releasing_shared_storage(monkeypatch, failing_side):
    from surogate.grpo.orchestrator import grpo_orch
    from surogate.grpo.utils import client

    started, exited, stop = threading.Event(), threading.Event(), threading.Event()
    pool_stopped = []

    async def stop_pool():
        pool_stopped.append(True)

    async def setup(*args, **kwargs):
        return SimpleNamespace(stop=stop_pool)

    def train():
        started.set()
        try:
            if failing_side == "trainer":
                raise RuntimeError("training failed")
            assert stop.wait(timeout=5), "orchestrator failure stranded the training thread"
        finally:
            exited.set()

    async def orchestrate(*args, **kwargs):
        while not started.is_set():
            await asyncio.sleep(0.001)
        if failing_side == "orchestrator":
            raise RuntimeError("orchestration failed")
        await asyncio.Event().wait()

    monkeypatch.setattr(client, "setup_inference_pool", setup)
    monkeypatch.setattr(grpo_orch, "orchestrate", orchestrate)
    config = SimpleNamespace(use_token_client=True, client=None, model=SimpleNamespace(name="base"))
    policy = SimpleNamespace(stop_event=stop, name="policy")
    with pytest.raises(RuntimeError, match="failed"):
        asyncio.run(_run(SimpleNamespace(train=train), policy, config))
    assert stop.is_set() and exited.is_set() and pool_stopped
