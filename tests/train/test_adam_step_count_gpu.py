"""Adam's step count: the first update of a run is t = 1.

The native update takes a zero-based index and adds 1. The trainers used to pass `step + 1`, so the one-GPU paths
(`train_step_graphed`, `update_with_config`) ran the first update at t = 2 and every later one a step ahead.
At t = 1, from empty moments, Adam moves every coordinate by lr, whatever its gradient: m / c1 = g and sqrt(v / c2) = |g|.
At t = 2 the same update is 0.744 x lr, which is what these tests would see on the old call sites. dispatch-PP passes the
zero-based step as it always did, and is checked here to stay at t = 1.
"""

import ast
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from safetensors.numpy import load_file, save_file

needs_gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
ROOT = Path(__file__).resolve().parents[2]
LR, RANK, ALPHA = 1e-3, 16, 32
CALLERS = ("surogate/train/trainer.py", "surogate/train/distributed.py", "surogate/grpo/trainer.py", "surogate/dpo/trainer.py",
           "surogate/debug/gradients.py")


@pytest.mark.parametrize("path", CALLERS)
def test_callers_pass_the_zero_based_step(path):
    """No caller adds 1 to the index it passes (the native side does)."""
    tree = ast.parse((ROOT / path).read_text())
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
             and n.func.attr in ("update_with_config", "train_step_graphed")]
    assert calls
    for call in calls:
        last = call.args[-1]
        assert not isinstance(last, ast.BinOp), f"{path}:{call.lineno} passes {ast.unparse(last)}"


def config_json():
    from tests.grpo.shared_model_configs import configurations

    config = dict(configurations()["qwen3_5_moe"])
    config.update(tie_word_embeddings=False, mtp_num_hidden_layers=0)
    return config


def make_model(root):
    """A miniature checkpoint with fixed random BF16 weights, a tokenizer, and a LoRA adapter with nonzero A and B."""
    from examples.sft.glm.create_dummy import dummy_tokenizer
    from safetensors.torch import save_file as save_torch
    from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import Qwen3_5MoeTextConfig
    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeForCausalLM

    torch.manual_seed(5)
    rng = np.random.default_rng(5)
    model_dir = root / "model"
    model_dir.mkdir()
    raw = config_json()
    (model_dir / "config.json").write_text(json.dumps(raw))
    config = Qwen3_5MoeTextConfig(**{k: v for k, v in raw.items() if k not in ("architectures", "model_type")})
    model = Qwen3_5MoeForCausalLM(config)
    for name, p in model.named_parameters():
        if p.dim() >= 2:
            torch.nn.init.normal_(p, std=0.1)
    save_torch({k: v.detach().clone().to(torch.bfloat16) for k, v in model.state_dict().items()},
               str(model_dir / "model.safetensors"), metadata={"format": "pt"})
    dummy_tokenizer().save_pretrained(model_dir)
    C, M, E = config.hidden_size, config.moe_intermediate_size, config.num_experts
    tensors = {}
    for l in range(config.num_hidden_layers):
        for name, ain, bout in (("gate_proj", C, M), ("up_proj", C, M), ("down_proj", M, C)):
            for e in range(E):
                prefix = f"base_model.model.model.layers.{l}.mlp.experts.{e}.{name}"
                tensors[f"{prefix}.lora_A.weight"] = rng.normal(0, 0.02, (RANK, ain)).astype(np.float32)
                tensors[f"{prefix}.lora_B.weight"] = rng.normal(0, 0.002, (bout, RANK)).astype(np.float32)
        for name, ain, bout in (("up_proj", C, M), ("down_proj", M, C)):
            prefix = f"base_model.model.model.layers.{l}.mlp.shared_experts.{name}"
            tensors[f"{prefix}.lora_A.weight"] = rng.normal(0, 0.02, (RANK, ain)).astype(np.float32)
            tensors[f"{prefix}.lora_B.weight"] = rng.normal(0, 0.002, (bout, RANK)).astype(np.float32)
    adapter = root / "adapter0"
    adapter.mkdir()
    save_file(tensors, str(adapter / "adapter_model.safetensors"), metadata={"format": "pt"})
    (adapter / "adapter_config.json").write_text(json.dumps({
        "base_model_name_or_path": "", "bias": "none", "fan_in_fan_out": False, "lora_alpha": float(ALPHA), "lora_dropout": 0.0,
        "peft_type": "LORA", "r": RANK, "target_modules": ["gate_proj", "up_proj", "down_proj"], "task_type": "CAUSAL_LM",
        "use_rslora": False}))
    return model_dir, adapter


def assert_first_update_is_lr(adapter0, adapter1):
    """Every coordinate with a gradient moved by lr: the median |change| / lr is 1, not 0.744."""
    before, after = load_file(str(adapter0)), load_file(str(adapter1))
    assert before.keys() == after.keys()
    moved = np.concatenate([np.abs(after[k].astype(np.float64) - before[k].astype(np.float64)).ravel() for k in before])
    moved = moved[moved > 0]
    assert moved.size
    ratio = float(np.median(moved)) / LR
    assert abs(ratio - 1.0) < 1e-3, f"median update {ratio:.4f} x lr: not Adam's t = 1"


def common_yaml(root, model_dir, adapter, dispatch):
    lines = [
        f"model: {model_dir}", f"output_dir: {root / 'out'}", "per_device_train_batch_size: 1", "gpus: 1", "sequence_len: 512",
        "sequence_chunks: 1", f"learning_rate: {LR}", "lr_scheduler_type: constant", "max_grad_norm: 1000.0", "weight_decay: 0.0",
        "optimizer: adamw_8bit", "recipe: bf16", "lora: true", f"lora_rank: {RANK}", f"lora_alpha: {ALPHA}", "lora_dtype: fp32",
        "lora_dropout: 0.0", "lora_target_modules: [gate_proj, up_proj, down_proj]", f"adapter_path: {adapter}",
        "adapter_init_mode: trainable", "train_router: false", "train_vision: false", "router_aux_loss_coef: 0.0", "router_z_loss_coef: 0.0",
    ]
    if dispatch:
        lines += ["parallelism: dispatch_pp", "offload_master: true", "fp8_weight_cache: off", "recompute: false", "qlora_recompute: false"]
    return lines


@pytest.mark.gpu
@pytest.mark.slow
@needs_gpu
@pytest.mark.parametrize("dispatch", [False, True], ids=["one-gpu-graph", "dispatch-pp"])
def test_sft_first_update_is_t_1(tmp_path, dispatch, monkeypatch):
    from surogate.core.config.loader import load_config
    from surogate.core.config.sft_config import SFTConfig
    from surogate.train.tokenize import TokenizedDataFileWriter
    from surogate.train.trainer import SurogateTrainerWrapper

    model_dir, adapter = make_model(tmp_path)
    if dispatch:
        monkeypatch.setenv("SUROGATE_DISPATCH_STAGE_BLOCKS", "3")
        monkeypatch.setenv("SUROGATE_DISPATCH_LORA_LOCAL", "1")
        monkeypatch.setenv("SUROGATE_DISPATCH_STAGE_PLAN", "3")
        monkeypatch.setenv("SUROGATE_DISPATCH_PREFETCH_BLOCKS", "3")
    (tmp_path / "out").mkdir()
    yaml = common_yaml(tmp_path, model_dir, adapter, dispatch) + [
        "gradient_accumulation_steps: 2", "sample_packing: false", "max_steps: 2", "save_steps: 1", "warmup_steps: 0", "eval_steps: 1000",
        "logging_steps: 1", "loss_scale: default", "datasets:", f"  - path: {tmp_path / 'unused.jsonl'}", "    type: conversation"]
    path = tmp_path / "train.yaml"
    path.write_text("\n".join(yaml) + "\n")
    config = load_config(SFTConfig, str(path))
    config.__post_init__()
    rng = np.random.default_rng(7)
    shard = tmp_path / "out/train-000.bin"
    with TokenizedDataFileWriter(str(shard), 512, masking=True, non_overlapping=True) as writer:
        for _ in range(8):
            tokens = rng.integers(3, 500, 512, dtype=np.int32)
            mask = np.ones(512, dtype=np.int32)
            mask[:40] = 0
            writer.add_document(tokens=tokens, position_ids=np.arange(512, dtype=np.int32), mask=mask)
    wrapper = SurogateTrainerWrapper(config=config, train_files=[str(shard)], eval_files=None)
    wrapper.train()
    assert_first_update_is_lr(adapter / "adapter_model.safetensors", tmp_path / "out/step_00000001/adapter_model.safetensors")


@pytest.mark.gpu
@pytest.mark.slow
@needs_gpu
@pytest.mark.parametrize("dispatch", [False, True], ids=["one-gpu-native", "dispatch-pp"])
def test_grpo_first_update_is_t_1(tmp_path, dispatch, monkeypatch):
    from surogate.core.config.loader import load_config
    from surogate.grpo.batch import prepare_batch
    from surogate.grpo.config import GRPOTrainConfig
    from surogate.grpo.data import microbatch_to_numpy
    from surogate.grpo.packer import init_multi_run_manager
    from surogate.grpo.trainer import GRPOTrainer
    from surogate.grpo.transport.types import TrainingSample

    model_dir, adapter = make_model(tmp_path)
    if dispatch:
        monkeypatch.setenv("SUROGATE_DISPATCH_STAGE_BLOCKS", "3")
        monkeypatch.setenv("SUROGATE_DISPATCH_LORA_LOCAL", "1")
        monkeypatch.setenv("SUROGATE_DISPATCH_STAGE_PLAN", "3")
        monkeypatch.setenv("SUROGATE_DISPATCH_PREFETCH_BLOCKS", "3")
    yaml = common_yaml(tmp_path, model_dir, adapter, dispatch) + [
        "max_steps: 1", "save_steps: 0", "broadcast_dtype: master", "transport_type: filesystem", "max_async_level: 0", "pad_to_multiple_of: 512",
        "single_sample_bins: true", "minibatches: 1", "loss:", "  kl_tau: 0.0", "  ipo_mask_low: 2.0", "  ipo_mask_high: 2.0"]
    path = tmp_path / "train.yaml"
    path.write_text("\n".join(yaml) + "\n")
    config = load_config(GRPOTrainConfig, str(path))
    trainer = GRPOTrainer(config)
    rng = np.random.default_rng(7)
    rows = [TrainingSample(prompt_ids=rng.integers(3, 500, 40).tolist(), prompt_mask=[False] * 40, completion_ids=rng.integers(3, 500, 400).tolist(),
                           completion_mask=[True] * 400, completion_logprobs=[-1.0] * 400, completion_temperatures=[1.0] * 400,
                           advantage=float(a), reward=0.0) for a in (100.0, -50.0, 70.0, -120.0)]  # gradients far above Adam's epsilon
    grid = prepare_batch(rollouts=rows, seq_len=512, pad_to_multiple_of=512, num_train_workers=1, idxs=[0] * len(rows), num_loras=1, single_sample_bins=True)[0]

    def setup_data(start_step=0):
        init_multi_run_manager(output_dir=config.output_dir)
        trainer.packer = SimpleNamespace(pack=lambda: None)
        trainer.data_loader = SimpleNamespace(wait_for_batch=lambda: None, get_batch=lambda: [microbatch_to_numpy(m) for m in grid])

    trainer._setup_data = setup_data
    trainer.train()
    trainer.close()
    assert_first_update_is_lr(adapter / "adapter_model.safetensors", tmp_path / "out/final_adapter/adapter_model.safetensors")
