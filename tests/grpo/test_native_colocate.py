"""Shared-policy publication and cancellation must unblock the next owner."""

import asyncio
import threading
from types import SimpleNamespace

import pytest

from surogate.grpo.native_colocate import SharedPolicy, _run


def test_publication_releases_the_batch_receiver_and_rejects_stale_versions(tmp_path, monkeypatch):
    class Server:
        version = -1

        def publish(self, name, modules, version):
            self.version = version

        def begin_training(self):
            pass

        def summary(self):
            return dict(policy_version=self.version, shared_base_bytes=1234,
                        serving_base_allocated_bytes=0, base_upload_bytes=0)

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
            await asyncio.sleep(.001)
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
