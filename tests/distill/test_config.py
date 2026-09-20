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
    assert d.candidate_only is False
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


def test_candidate_mode_has_explicit_hard_label_contract():
    c = _make_config(eval_steps=0, distillation={"candidate_only": True, "top_k": 8,
                                                "kd_weight": 1., "ce_weight": 0.})
    c._validate_distillation_config()
    assert c.distillation.candidate_only
    c.distillation.temperature = 2.
    with pytest.raises(ValueError, match="candidate_only"):
        c._validate_distillation_config()


def test_candidate_mode_rejects_teacher_capture_before_io():
    from surogate.distill.capture import run_capture

    c = _make_config(eval_steps=0, distillation={"candidate_only": True, "top_k": 8,
                                                "kd_weight": 1., "ce_weight": 0.})
    with pytest.raises(ValueError, match="supplied answer-token IDs"):
        run_capture(c, ["must-not-be-opened.bin"], "unused")


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


@pytest.mark.parametrize("objective", ["cross_entropy", "brier", "rps"])
def test_candidate_objective_parses_explicit_hard_gold_contract(objective):
    config = _make_config(lora=True, recipe="fp8_hybrid", eval_steps=0,
                         distillation={"candidate_only": True, "candidate_objective": objective,
                                       "top_k": 512, "kd_weight": 1., "ce_weight": 0.})
    config._validate_distillation_config()
    assert config.distillation.candidate_objective == objective
    # Slots may include interspersed padding. Native preflight enforces <=255
    # actual non-padding candidates for the two new proper objectives.
    assert config.distillation.top_k == 512


@pytest.mark.parametrize("objective", ["unknown", "forward_kl", "reverse_kl", "", None, 1])
def test_candidate_objective_rejects_unsupported_semantics(objective):
    config = _make_config(distillation={"candidate_objective": objective})
    with pytest.raises(ValueError, match="candidate_objective"):
        config._validate_distillation_config()


@pytest.mark.parametrize("override", [{"candidate_only": False}, {"lora": False}, {"recipe": "bf16"}, {"recipe": "nvfp4"}])
def test_new_candidate_objective_requires_explicit_lora_fp8(override):
    kwargs = dict(lora=True, recipe="fp8_hybrid", eval_steps=0)
    distillation = dict(candidate_only=True, candidate_objective="brier", kd_weight=1., ce_weight=0.)
    if "candidate_only" in override:
        distillation.update(override)
    else:
        kwargs.update(override)
    config = _make_config(**kwargs, distillation=distillation)
    with pytest.raises(ValueError, match="candidate_objective"):
        config._validate_distillation_config()


def test_legacy_candidate_ce_default_preserves_old_recipe_contract():
    config = _make_config(recipe="bf16", eval_steps=0,
                         distillation={"candidate_only": True, "kd_weight": 1., "ce_weight": 0.})
    config._validate_distillation_config()
    assert config.distillation.candidate_objective == "cross_entropy"


@pytest.mark.parametrize("objective", ["brier", "rps"])
def test_proper_candidates_reject_expert_parallelism_before_dispatch(objective):
    config = _make_config(lora=True, recipe="fp8_hybrid", eval_steps=0, ep_size=2,
                         distillation={"candidate_only": True, "candidate_objective": objective,
                                       "kd_weight": 1., "ce_weight": 0.})
    with pytest.raises(ValueError, match="expert parallelism"):
        config._validate_distillation_config()
