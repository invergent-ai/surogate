"""Exercise real Python training startup without invoking GPU updates or exports."""

from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch  # noqa: F401 -- load torch/NCCL before native Surogate

from surogate.train import trainer as module


class StartupReachedLoop(Exception):
    pass


def startup(monkeypatch, allocator_error):
    logged = Mock()
    native = Mock()
    native.get_allocator_info.side_effect = [allocator_error, {"memory": {"device": 123}}]
    warning = Mock()
    monkeypatch.setattr(module.logger, "warning", warning)

    @contextmanager
    def context(*_args, **_kwargs):
        yield logged

    monkeypatch.setattr(module, "training_logger_context", context)
    instance = SimpleNamespace(
        total_batch_size=32, steps_per_epoch=160, max_steps=160,
        config=SimpleNamespace(gpus=2, recipe="fp8_hybrid", optimizer="adamw",
                               lr_scheduler_type="cosine", from_scratch=False, lora=False, cooldown_steps=0),
        _train_vision=False, eval_loader=None, trainer=native, start_step=0,
        warmup_steps=8, cooldown_steps=0,
        run_training_loop=Mock(side_effect=StartupReachedLoop),
    )
    return instance, logged, native, warning


def test_allocator_bad_name_does_not_abort_real_python_startup(monkeypatch):
    bad_name = UnicodeDecodeError("utf-8", b"\x90", 0, 1, "invalid start byte")
    instance, logged, native, warning = startup(monkeypatch, bad_name)
    with pytest.raises(StartupReachedLoop):
        module.SurogateTrainerWrapper.train(instance)
    assert native.get_allocator_info.call_count == 2
    logged.log_allocator.assert_called_once_with({"memory": {"device": 123}})
    warning.assert_called_once()
    assert "GPU 0" in warning.call_args.args[0]
    assert "invalid UTF-8" in warning.call_args.args[0]
    instance.run_training_loop.assert_called_once_with(logged)


def test_numerical_native_errors_still_abort_startup(monkeypatch):
    instance, logged, _native, warning = startup(monkeypatch, RuntimeError("CUDA numerical failure"))
    with pytest.raises(RuntimeError, match="CUDA numerical failure"):
        module.SurogateTrainerWrapper.train(instance)
    instance.run_training_loop.assert_not_called()
    logged.log_allocator.assert_not_called()
    warning.assert_not_called()
