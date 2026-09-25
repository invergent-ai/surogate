"""Shared-model GRPO publish on Qwen3.5 with the default `lora_target_modules: ["all"]`.

`all` adapts the linear-attention (GatedDeltaNet) mixer too. Serving's LoRA directory names those modules
`linear_attn.<proj>` (bind_lora_gdn); publishing them bare (`in_proj_qkv`) was refused ("adapter module
'in_proj_qkv' on layer 0 is unsupported by this serving model"). Here a nonzero adapter is published from the
trainer and the served greedy completion is scored in the trainer: serving must agree with the trainer's
log-probs WITH the linear-attention adapters, and measurably better than with them zeroed.
"""

import os
import tempfile
from pathlib import Path

import numpy as np
import pytest
import requests

pytestmark = [pytest.mark.gpu, pytest.mark.slow]


def _qwen35_snapshot():
    from tests import test_onboarding_qwen3_5 as onboarding

    return onboarding.resolve_model_path()


def test_publish_all_targets_with_linear_attention():
    torch = pytest.importorskip("torch")
    try:
        from surogate import _surogate, _surogate_serve
    except ImportError:
        pytest.skip("surogate extensions not built")
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    model = _qwen35_snapshot()
    if model is None:
        pytest.skip("Qwen3.5-0.8B not available (set QWEN3_5_MODEL_PATH)")
    model = str(model)
    from safetensors.torch import save_file

    from surogate.dsl.ir_builder import build_dsl_ir_for_model
    from surogate.grpo.shared_model import shared_execution
    from surogate.grpo.shared_weights import adapter_modules, borrow_weights, write_shared_artifact
    from surogate.kernels.jit_compile import compile_jit_kernels
    from surogate.utils.hf import get_model_weights_path

    import json

    assert shared_execution(json.loads((Path(model) / "config.json").read_text()), ["all"]) == "serve"
    options = _surogate.RuntimeOptions(use_cuda_graphs=False, master_dtype="bf16", offload_master=False,
                                       offload_grads=False, offload_optimizer=False)
    options.dsl_ir_json = build_dsl_ir_for_model(model)
    manifests = compile_jit_kernels(options.dsl_ir_json)
    if manifests:
        options.jit_kernel_manifests = manifests
    trainer = _surogate.SurogateTrainer(
        ngpu=1, config=_surogate.PretrainedConfig.from_pretrained(model, "bf16"), options=options, batch_size=1,
        seq_len=128, grad_accum=1,
        lora_config=_surogate.LoRAAdapterConfig(rank=8, alpha=8, dropout=0., dtype="fp32", target_modules=["all"]))
    trainer.import_weights(get_model_weights_path(model))
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        shapes = {k: tuple(torch.from_dlpack(v).shape) for k, v in trainer.get_lora_weights(0).items()}
        assert any(".linear_attn." in k for k in shapes)
        g = torch.Generator().manual_seed(7)
        live = {k: torch.randn(s, generator=g) * (0.05 if "lora_A" in k else 0.03) for k, s in sorted(shapes.items())}
        # Only the linear-attention adapters nonzero (B = 0 elsewhere), and a zero adapter: a strong random
        # adapter on every module moves the model by ~3 nats, which amplifies bf16 serving/training differences
        # past any useful bound; each projection alone serves at the zero-adapter floor (measured 0.02-0.04).
        gdn_only = {k: (v if ("lora_A" in k or ".linear_attn." in k) else torch.zeros_like(v)) for k, v in live.items()}
        zero = {k: (v if "lora_A" in k else torch.zeros_like(v)) for k, v in live.items()}
        save_file(live, str(tmp / "live.safetensors"))
        save_file(gdn_only, str(tmp / "gdn_only.safetensors"))
        save_file(zero, str(tmp / "zero.safetensors"))
        trainer.import_adapter(str(tmp / "live.safetensors"))
        modules = adapter_modules(trainer, 1.)
        assert {m["module"] for m in modules} >= {f"linear_attn.{p}" for p in
                                                   ("in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b", "out_proj")}
        assert not {m["module"] for m in modules} & {"in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b"}
        weights = borrow_weights(trainer, write_shared_artifact(model, tmp / "shared.sinfer"))
        settings = dict(host="127.0.0.1", port=int(os.environ.get("SUROGATE_SHARED_PORT", "18979")), device=0,
                        model="qwen3", max_context=256, prefill_chunk=128, max_concurrency=4, kv_capacity=1024,
                        use_cuda_graph=False, rank=8)
        server = _surogate_serve.SharedServer(str(tmp / "shared.sinfer"), weights, settings)
        url = f"http://127.0.0.1:{settings['port']}/v1/chat/completions"
        body = dict(model="default",
                    messages=[dict(role="user", content="Write one sentence about rivers and one about mountains.")],
                    max_tokens=48, temperature=0, logprobs=True, return_token_ids=True,
                    chat_template_kwargs=dict(enable_thinking=False))

        def served_and_scored(adapter_file, version, score_with):
            """Publish `adapter_file`, generate greedily, score the completion in the trainer with each adapter
            of `score_with`; returns (served log-probs, {name: trainer log-probs})."""
            trainer.import_adapter(str(adapter_file))
            server.publish("default", adapter_modules(trainer, 1.), version)
            response = requests.post(url, json=body, timeout=120)
            response.raise_for_status()
            result = response.json()
            server.begin_training()
            prompt, completion = result["prompt_token_ids"], result["choices"][0]["token_ids"]
            tokens = prompt + completion
            ids = np.zeros((1, 128), np.int32)
            targets = np.full_like(ids, -100)
            ids[0, :len(tokens)] = tokens
            targets[0, len(prompt) - 1:len(tokens) - 1] = completion
            window = slice(len(prompt) - 1, len(tokens) - 1)
            scores = {}
            for name, path in score_with.items():
                trainer.import_adapter(str(path))
                scores[name] = trainer.compute_logprobs(ids, targets)[0, window].copy()
            served = np.array([t["logprob"] for t in result["choices"][0]["logprobs"]["content"]])
            return served, scores

        try:
            # Floor: a zero adapter, served vs trainer (bf16 hybrid recurrence, different prefill widths).
            served0, s0 = served_and_scored(tmp / "zero.safetensors", 0, {"zero": tmp / "zero.safetensors"})
            served1, s1 = served_and_scored(tmp / "gdn_only.safetensors", 1, {"gdn": tmp / "gdn_only.safetensors",
                                                                               "zero": tmp / "zero.safetensors"})
            # The full random adapter publishes too (every module name is accepted).
            trainer.import_adapter(str(tmp / "live.safetensors"))
            server.publish("default", adapter_modules(trainer, 1.), 2)
        finally:
            server.close()
    floor = float(np.abs(served0 - s0["zero"]).mean())
    err_with = float(np.abs(served1 - s1["gdn"]).mean())
    err_without = float(np.abs(served1 - s1["zero"]).mean())
    print({"floor_serve_vs_trainer_zero_adapter": floor, "serve_vs_trainer_gdn_adapters": err_with,
           "serve_vs_trainer_without_gdn_adapters": err_without})
    # Served with the linear-attention adapters as closely as a zero adapter is served, and far from the
    # trainer without them: they are published under the names serving binds, and applied the same way.
    assert err_with <= 1.5 * floor + 0.03, (err_with, floor)
    assert err_with < 0.25 * err_without, (err_with, err_without)
