"""CPU tests for DistillationConfig parsing, defaults, and validation.

SFTConfig.__post_init__ resolves the model over the network, so these tests
exercise SFTConfig.__init__ (pure field parsing) plus the validation helper
`_validate_distillation_config` and the `create_runtime_config` CUDA-graph
disable block directly, with a stubbed model_info.
"""

from types import SimpleNamespace

import pytest

sft_config = pytest.importorskip(
    "surogate.core.config.sft_config", reason="requires the built _surogate native module"
)

from surogate.core.config.sft_config import DistillationConfig, DistributedConfig, SFTConfig  # noqa: E402
from surogate.utils.dict import DictDefault  # noqa: E402


def _make_config(**extra) -> SFTConfig:
    return SFTConfig(DictDefault({"model": "dummy/model", **extra}))


def test_defaults():
    d = DistillationConfig()
    assert d.teacher_model is None
    assert d.top_k == 32
    assert d.temperature == 1.0
    assert d.kd_weight == 0.5
    assert d.ce_weight is None
    assert d.teacher_batch_size == 4
    assert d.kd_dir is None
    assert d.teacher_api_base is None
    assert d.teacher_api_key_var == "VLLM_API_KEY"
    assert d.teacher_api_concurrency == 8
    assert d.teacher_api_timeout == 1200


def test_api_fields_dict_parsing():
    c = _make_config(
        distillation={
            "teacher_model": "served-name",
            "teacher_api_base": "http://localhost:8000/v1",
            "teacher_api_key_var": "MY_KEY",
            "teacher_api_concurrency": 16,
            "teacher_api_timeout": 60,
        }
    )
    d = c.distillation
    assert d.teacher_api_base == "http://localhost:8000/v1"
    assert d.teacher_api_key_var == "MY_KEY"
    assert d.teacher_api_concurrency == 16
    assert d.teacher_api_timeout == 60
    c._validate_distillation_config()


def test_absent_block_parses_to_none():
    assert _make_config().distillation is None


def test_dict_block_parsing():
    c = _make_config(
        distillation={
            "teacher_model": "org/teacher",
            "top_k": 64,
            "temperature": 2.0,
            "kd_weight": 0.7,
            "ce_weight": 0.4,
            "teacher_batch_size": 8,
            "kd_dir": "/tmp/kd",
        }
    )
    d = c.distillation
    assert isinstance(d, DistillationConfig)
    assert d.teacher_model == "org/teacher"
    assert d.top_k == 64
    assert d.temperature == 2.0
    assert d.kd_weight == 0.7
    assert d.ce_weight == 0.4
    assert d.teacher_batch_size == 8
    assert d.kd_dir == "/tmp/kd"


def test_partial_dict_block_uses_defaults():
    c = _make_config(distillation={"teacher_model": "org/teacher"})
    d = c.distillation
    assert d.top_k == 32
    assert d.temperature == 1.0
    assert d.kd_weight == 0.5
    assert d.ce_weight is None


def test_instance_passthrough():
    inst = DistillationConfig(teacher_model="org/teacher", top_k=16)
    c = _make_config(distillation=inst)
    assert c.distillation is inst


def test_ce_weight_defaults_to_one_minus_kd_weight():
    c = _make_config(distillation={"kd_weight": 0.3})
    c._validate_distillation_config()
    assert c.distillation.ce_weight == pytest.approx(0.7)


def test_ce_weight_explicit_is_kept():
    c = _make_config(distillation={"kd_weight": 0.3, "ce_weight": 1.0})
    c._validate_distillation_config()
    assert c.distillation.ce_weight == 1.0


def test_none_distillation_validation_is_noop():
    c = _make_config()
    c._validate_distillation_config()
    assert c.distillation is None


@pytest.mark.parametrize(
    "block, match",
    [
        ({"top_k": 0}, "top_k"),
        ({"top_k": 2048}, "top_k"),
        ({"temperature": 0.0}, "temperature"),
        ({"temperature": -1.0}, "temperature"),
        ({"kd_weight": -0.1}, "kd_weight"),
        ({"kd_weight": 0.5, "ce_weight": -0.5}, "ce_weight"),
        ({"kd_weight": 1.5}, "ce_weight"),  # implied ce_weight = 1 - 1.5 < 0
        ({"teacher_batch_size": 0}, "teacher_batch_size"),
        ({"teacher_api_concurrency": 0}, "teacher_api_concurrency"),
        ({"teacher_api_timeout": 0}, "teacher_api_timeout"),
    ],
)
def test_validation_errors(block, match):
    c = _make_config(distillation=block)
    with pytest.raises(ValueError, match=match):
        c._validate_distillation_config()


def test_distributed_rejected_with_distillation():
    c = _make_config(distillation={"kd_weight": 0.5})
    c.distributed = DistributedConfig(num_nodes=2)
    with pytest.raises(ValueError, match="distributed"):
        c._validate_distillation_config()


def test_dispatch_pp_rejected_with_distillation():
    c = _make_config(distillation={"kd_weight": 0.5})
    c.parallelism = "dispatch_pp"
    with pytest.raises(ValueError, match="dispatch_pp"):
        c._validate_distillation_config()


def test_lmhead_drop_ignored_rows_forced_off():
    c = _make_config(distillation={"kd_weight": 0.5}, lmhead_drop_ignored_rows=True)
    assert c.lmhead_drop_ignored_rows is True
    c._validate_distillation_config()
    assert c.lmhead_drop_ignored_rows is False


def test_cuda_graphs_auto_disabled_in_runtime_config():
    c = _make_config(distillation={"kd_weight": 0.5}, use_cuda_graphs=True)
    c.model_info = SimpleNamespace(quant_info=None)
    assert c.use_cuda_graphs is True
    c.create_runtime_config()
    assert c.use_cuda_graphs is False


def test_cuda_graphs_untouched_without_distillation():
    c = _make_config(use_cuda_graphs=True)
    c.model_info = SimpleNamespace(quant_info=None)
    c.create_runtime_config()
    assert c.use_cuda_graphs is True
