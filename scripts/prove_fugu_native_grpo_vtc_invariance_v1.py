#!/usr/bin/env python3
"""Zero-paid dynamic proof for native GRPO final-call VTC invariance.

This is deliberately a small-model runtime proof, not a campaign optimizer run.
It creates two exact clones of one deterministic BF16 LoRA adapter, accumulates
the same gradient-bearing native GRPO micro-step, and follows it with a zero-
dloss micro-step whose valid-token count is either 1 or 8.  The repaired native
path passes only if the accumulated gradients, optimizer norm, and post-update
LoRA weights are bit-identical across the two variants.

The script never downloads a model.  The model path must already be present.
It also refuses a GPU with more than --max-used-memory-mib already allocated.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = Path(
    "/home/densemax2/.cache/huggingface/hub/"
    "models--Qwen--Qwen3.5-0.8B/snapshots/2fc06364715b967f1860aea9cf38778875588b17"
)
DEFAULT_OUTPUT = REPO_ROOT / "tests/grpo/artifacts/fugu_native_grpo_vtc_invariance_v1.json"
REBUILT_EXTENSION = REPO_ROOT / "csrc/build-fugu-repair/_surogate.abi3.so"
REBUILT_COMMON = REPO_ROOT / "csrc/build-fugu-repair/libsurogate-common.so"
NATIVE_SOURCE = REPO_ROOT / "csrc/src/runtime/dsl/dsl_model_execution.cpp"
SEQUENCE_LENGTH = 16
GRAD_ACCUMULATION = 2
FINAL_VTC_VARIANTS = (1, 8)


class ProofFailure(RuntimeError):
    """Raised when a proof precondition or invariant fails."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_attestation(path: Path) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    return {
        "path": str(path),
        "resolved_path": str(resolved),
        "size_bytes": resolved.stat().st_size,
        "sha256": _sha256_file(resolved),
    }


def _gpu_preflight(physical_gpu: str, max_used_memory_mib: int) -> dict[str, Any]:
    if "," in physical_gpu:
        raise ProofFailure("--gpu must name exactly one physical GPU")
    command = [
        "nvidia-smi",
        f"--id={physical_gpu}",
        "--query-gpu=index,name,uuid,memory.used,memory.total,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    fields = [field.strip() for field in result.stdout.strip().split(",")]
    if len(fields) != 6:
        raise ProofFailure(f"unexpected nvidia-smi response: {result.stdout!r}")
    index, name, uuid, memory_used, memory_total, utilization = fields
    used_mib = int(memory_used)
    if used_mib > max_used_memory_mib:
        raise ProofFailure(
            f"GPU {physical_gpu} already uses {used_mib} MiB, above the "
            f"{max_used_memory_mib} MiB proof limit"
        )
    return {
        "physical_index": index,
        "name": name,
        "uuid": uuid,
        "memory_used_mib_before": used_mib,
        "memory_total_mib": int(memory_total),
        "utilization_percent_before": int(utilization),
        "maximum_allowed_used_memory_mib": max_used_memory_mib,
    }


def _loaded_common_path(extension: Path) -> Path:
    result = subprocess.run(["ldd", str(extension)], check=True, capture_output=True, text=True)
    for line in result.stdout.splitlines():
        match = re.search(r"libsurogate-common\.so\s+=>\s+(\S+)", line)
        if match:
            return Path(match.group(1)).resolve(strict=True)
    raise ProofFailure("the imported extension did not resolve libsurogate-common.so")


def _tensor_map(raw: Any, torch: Any) -> dict[str, Any]:
    return {
        name: torch.utils.dlpack.from_dlpack(array)
        .detach()
        .float()
        .cpu()
        .contiguous()
        .numpy()
        .copy()
        for name, array in raw.items()
    }


def _tensor_map_digest(tensors: dict[str, Any]) -> str:
    """Preserve the digest convention used by the original console proof."""
    digest = hashlib.sha256()
    for name in sorted(tensors):
        digest.update(name.encode())
        digest.update(tensors[name].tobytes())
    return digest.hexdigest()


def _tensor_map_summary(tensors: dict[str, Any]) -> dict[str, Any]:
    return {
        "digest_sha256": _tensor_map_digest(tensors),
        "tensor_count": len(tensors),
        "element_count": sum(int(tensor.size) for tensor in tensors.values()),
        "tensors": {
            name: {"shape": list(tensor.shape), "dtype": str(tensor.dtype)}
            for name, tensor in sorted(tensors.items())
        },
    }


def _max_abs_diff(left: dict[str, Any], right: dict[str, Any], np: Any) -> float:
    if left.keys() != right.keys():
        raise ProofFailure("tensor maps have different keys")
    maximum = 0.0
    for name in left:
        if left[name].shape != right[name].shape:
            raise ProofFailure(f"tensor shape mismatch for {name}")
        maximum = max(maximum, float(np.max(np.abs(left[name] - right[name]))))
    return maximum


def _build_report(args: argparse.Namespace, gpu_attestation: dict[str, Any]) -> dict[str, Any]:
    # Import CUDA-facing modules only after CUDA_VISIBLE_DEVICES is narrowed.
    import numpy as np
    import torch

    from surogate import _surogate
    from surogate.dsl.ir_builder import build_dsl_ir_for_model
    from surogate.kernels.jit_compile import compile_jit_kernels
    from surogate.utils.hf import get_model_weights_path

    model = args.model.resolve(strict=True)
    for required in (model / "config.json", model / "model.safetensors.index.json"):
        if not required.is_file():
            raise ProofFailure(f"cached model is incomplete: missing {required}")
    # Keep the snapshot-relative index path for import: resolving its HF-cache
    # symlink would move the apparent parent into blobs/, breaking relative
    # shard lookup inside the index.
    weight_import_path = Path(get_model_weights_path(str(model)))
    weight_import_path.resolve(strict=True)
    weights_index = json.loads((model / "model.safetensors.index.json").read_text())
    weight_files = [
        (model / relative_path).resolve(strict=True)
        for relative_path in sorted(set(weights_index["weight_map"].values()))
    ]
    imported_extension = Path(_surogate.__file__).resolve(strict=True)
    loaded_common = _loaded_common_path(imported_extension)

    imported_extension_hash = _sha256_file(imported_extension)
    rebuilt_extension_hash = _sha256_file(REBUILT_EXTENSION.resolve(strict=True))
    loaded_common_hash = _sha256_file(loaded_common)
    rebuilt_common_hash = _sha256_file(REBUILT_COMMON.resolve(strict=True))
    if imported_extension_hash != rebuilt_extension_hash:
        raise ProofFailure("imported _surogate extension is not the rebuilt repair artifact")
    if loaded_common != REBUILT_COMMON.resolve(strict=True):
        raise ProofFailure(
            f"imported extension loaded {loaded_common}, not rebuilt common {REBUILT_COMMON.resolve()}"
        )
    if loaded_common_hash != rebuilt_common_hash:
        raise ProofFailure("loaded common library hash differs from rebuilt repair artifact")

    native_source_text = NATIVE_SOURCE.read_text()
    if "void DslModel::step_grpo_native(" not in native_source_text:
        raise ProofFailure("native GRPO implementation is missing")
    native_body = native_source_text.split("void DslModel::step_grpo_native(", 1)[1]
    if "mUseTokenScale = false;" not in native_body[:4096]:
        raise ProofFailure("native GRPO does not disable legacy optimizer token scaling")

    ir = build_dsl_ir_for_model(str(model))
    jit_manifests = compile_jit_kernels(ir)

    def make_trainer() -> Any:
        config = _surogate.PretrainedConfig.from_pretrained(str(model), "bf16")
        options = _surogate.RuntimeOptions(
            recompute="true",
            offload_residual=False,
            use_cuda_graphs=False,
            offload_master=False,
            offload_grads=False,
            offload_optimizer=False,
            shard_gradients=False,
            use_zero_copy=False,
            doc_masking=False,
            recipe="bf16",
        )
        options.dsl_ir_json = ir
        if jit_manifests:
            options.jit_kernel_manifests = jit_manifests
        lora = _surogate.LoRAAdapterConfig(
            rank=2,
            alpha=2.0,
            dropout=0.0,
            target_modules=["q_proj"],
            dtype="bf16",
            use_rslora=False,
            train_router=False,
        )
        trainer = _surogate.SurogateTrainer(
            ngpu=1,
            config=config,
            options=options,
            batch_size=1,
            seq_len=SEQUENCE_LENGTH,
            grad_accum=GRAD_ACCUMULATION,
            memcpy_all_gather=True,
            memcpy_send_recv=True,
            lora_config=lora,
            qlora_config=None,
        )
        trainer.import_weights(str(weight_import_path))
        return trainer

    def run_variant(final_vtc: int, adapter_file: Path) -> dict[str, Any]:
        trainer = make_trainer()
        trainer.import_adapter(str(adapter_file))
        trainer.set_grad_accumulation(GRAD_ACCUMULATION)
        initial_weights = _tensor_map(trainer.get_lora_weights(0), torch)

        inputs = np.arange(100, 100 + SEQUENCE_LENGTH, dtype=np.int32).reshape(1, SEQUENCE_LENGTH)
        position_ids = np.arange(SEQUENCE_LENGTH, dtype=np.int32).reshape(1, SEQUENCE_LENGTH)
        temperatures = np.ones((1, SEQUENCE_LENGTH), dtype=np.float32)
        sample_starts = np.array([0], dtype=np.int32)
        sample_ends = np.array([SEQUENCE_LENGTH], dtype=np.int32)
        inference_logprobs = np.full(SEQUENCE_LENGTH, -10.0, dtype=np.float32)

        # The common first micro-step is the only gradient-bearing call.
        targets = np.full((1, SEQUENCE_LENGTH), -100, dtype=np.int32)
        targets[0, 0] = inputs[0, 1]
        advantages = np.zeros(SEQUENCE_LENGTH, dtype=np.float32)
        advantages[1] = 1.0
        loss_mask = np.zeros(SEQUENCE_LENGTH, dtype=np.uint8)
        loss_mask[1] = 1
        replay_mask = np.zeros(SEQUENCE_LENGTH, dtype=np.uint8)
        trainer.step_grpo_native(
            inputs,
            targets,
            inference_logprobs,
            advantages,
            loss_mask,
            sample_starts,
            sample_ends,
            position_ids=position_ids,
            temperatures=temperatures,
            replay_mask=replay_mask,
            loss_scale=1.0,
            adv_tau=1.0,
            replay_tau=0.0,
            kl_tau=0.0,
        )

        # This final micro-step has exactly zero custom dloss: every selected
        # token enters the replay branch, where replay_tau is zero.
        final_targets = np.full((1, SEQUENCE_LENGTH), -100, dtype=np.int32)
        final_advantages = np.zeros(SEQUENCE_LENGTH, dtype=np.float32)
        final_loss_mask = np.zeros(SEQUENCE_LENGTH, dtype=np.uint8)
        final_replay_mask = np.zeros(SEQUENCE_LENGTH, dtype=np.uint8)
        for logical_index in range(1, final_vtc + 1):
            final_loss_mask[logical_index] = 1
            final_replay_mask[logical_index] = 1
            final_targets[0, logical_index - 1] = inputs[0, logical_index]
        trainer.step_grpo_native(
            inputs,
            final_targets,
            inference_logprobs,
            final_advantages,
            final_loss_mask,
            sample_starts,
            sample_ends,
            position_ids=position_ids,
            temperatures=temperatures,
            replay_mask=final_replay_mask,
            loss_scale=1.0,
            adv_tau=1.0,
            replay_tau=0.0,
            kl_tau=0.0,
        )

        observed_final_vtc = int(trainer.get_valid_token_count(0))
        gradients = _tensor_map(trainer.get_lora_gradients(0), torch)
        optimizer = _surogate.OptimizerConfig(
            optimizer="adamw",
            learning_rate=1e-3,
            weight_decay=0.0,
            grad_clip=0.0,
            adamw_beta1=0.9,
            adamw_beta2=0.999,
            adamw_epsilon=1e-8,
        )
        optimizer_result = dict(trainer.update_with_config(optimizer, 0))
        updated_weights = _tensor_map(trainer.get_lora_weights(0), torch)

        result = {
            "requested_final_vtc": final_vtc,
            "observed_final_vtc": observed_final_vtc,
            "initial_weights": _tensor_map_summary(initial_weights),
            "accumulated_gradients": _tensor_map_summary(gradients),
            "optimizer_norm": float(optimizer_result["norm"]),
            "post_update_weights": _tensor_map_summary(updated_weights),
            "_initial_weights": initial_weights,
            "_gradients": gradients,
            "_updated_weights": updated_weights,
        }
        del trainer
        gc.collect()
        torch.cuda.empty_cache()
        return result

    with tempfile.TemporaryDirectory(prefix="fugu_grpo_scale_audit_") as temp_dir:
        seed_trainer = make_trainer()
        adapter_dir = Path(temp_dir) / "initial"
        seed_trainer.export_adapter(str(adapter_dir), str(model))
        adapter_file = adapter_dir / "adapter_model.safetensors"
        seed_trainer.import_adapter(str(adapter_file))
        adapter_file_attestation = {
            "size_bytes": adapter_file.stat().st_size,
            "sha256": _sha256_file(adapter_file),
        }
        del seed_trainer
        gc.collect()
        torch.cuda.empty_cache()

        variants = {
            final_vtc: run_variant(final_vtc, adapter_file) for final_vtc in FINAL_VTC_VARIANTS
        }

    variant_1 = variants[1]
    variant_8 = variants[8]
    initial_weight_max_abs_diff = _max_abs_diff(
        variant_1["_initial_weights"], variant_8["_initial_weights"], np
    )
    gradient_max_abs_diff = _max_abs_diff(variant_1["_gradients"], variant_8["_gradients"], np)
    post_update_weight_max_abs_diff = _max_abs_diff(
        variant_1["_updated_weights"], variant_8["_updated_weights"], np
    )
    norm_abs_diff = abs(variant_1["optimizer_norm"] - variant_8["optimizer_norm"])

    assertions = {
        "initial_adapters_are_identical": initial_weight_max_abs_diff == 0.0,
        "requested_final_vtcs_were_observed": (
            variant_1["observed_final_vtc"] == 1 and variant_8["observed_final_vtc"] == 8
        ),
        "final_vtcs_differ": variant_1["observed_final_vtc"] != variant_8["observed_final_vtc"],
        "accumulated_gradients_are_identical": gradient_max_abs_diff == 0.0,
        "optimizer_norms_are_identical": norm_abs_diff == 0.0,
        "post_update_lora_weights_are_identical": post_update_weight_max_abs_diff == 0.0,
    }

    for variant in variants.values():
        variant.pop("_initial_weights")
        variant.pop("_gradients")
        variant.pop("_updated_weights")

    report = {
        "schema_version": "fugu.native_grpo.final_vtc_invariance.v1",
        "verdict": "PASS" if all(assertions.values()) else "FAIL",
        "claim": (
            "With one identical gradient-bearing native GRPO micro-step and an exactly zero-dloss "
            "final micro-step, changing only the final valid-token count from 1 to 8 does not change "
            "the accumulated LoRA gradients, full-precision AdamW norm, or post-update LoRA weights."
        ),
        "scope": {
            "paid_calls": 0,
            "campaign_optimizer_steps": 0,
            "local_small_model_optimizer_updates": 2,
            "campaign_27b_checkpoint_touched": False,
            "paid_collection_artifacts_touched": False,
            "proof_model_only": True,
        },
        "proof_source": _file_attestation(Path(__file__)),
        "runtime": {
            "native_source": _file_attestation(NATIVE_SOURCE),
            "imported_extension": _file_attestation(imported_extension),
            "rebuilt_extension": _file_attestation(REBUILT_EXTENSION),
            "loaded_common": _file_attestation(loaded_common),
            "rebuilt_common": _file_attestation(REBUILT_COMMON),
            "imported_extension_matches_rebuilt": imported_extension_hash == rebuilt_extension_hash,
            "loaded_common_is_rebuilt": loaded_common == REBUILT_COMMON.resolve(strict=True),
            "native_token_scaling_fix_present": True,
        },
        "model": {
            "name": "Qwen/Qwen3.5-0.8B",
            "snapshot_revision": model.name,
            "snapshot_path": str(model),
            "config": _file_attestation(model / "config.json"),
            "weights_index": _file_attestation(model / "model.safetensors.index.json"),
            "import_path": _file_attestation(weight_import_path),
            "weight_files": [_file_attestation(path) for path in weight_files],
            "weight_bytes": sum(path.stat().st_size for path in weight_files),
            "network_downloads": 0,
        },
        "gpu_preflight": gpu_attestation,
        "experiment": {
            "recipe": "bf16",
            "gpu_count": 1,
            "batch_size": 1,
            "sequence_length": SEQUENCE_LENGTH,
            "gradient_accumulation": GRAD_ACCUMULATION,
            "lora": {
                "rank": 2,
                "alpha": 2.0,
                "dropout": 0.0,
                "target_modules": ["q_proj"],
                "dtype": "bf16",
            },
            "optimizer": {
                "type": "adamw_full_precision",
                "learning_rate": 1e-3,
                "weight_decay": 0.0,
                "grad_clip": 0.0,
                "beta1": 0.9,
                "beta2": 0.999,
                "epsilon": 1e-8,
            },
            "initial_adapter_file": adapter_file_attestation,
            "final_micro_step": "all selected tokens are replay with replay_tau=0 and kl_tau=0",
        },
        "variants": {
            "final_vtc_1": variants[1],
            "final_vtc_8": variants[8],
        },
        "comparisons": {
            "initial_weight_max_abs_diff": initial_weight_max_abs_diff,
            "gradient_max_abs_diff": gradient_max_abs_diff,
            "optimizer_norm_abs_diff": norm_abs_diff,
            "post_update_weight_max_abs_diff": post_update_weight_max_abs_diff,
        },
        "assertions": assertions,
    }
    if report["verdict"] != "PASS":
        raise ProofFailure(json.dumps(report, sort_keys=True))
    return report


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", default="2", help="one physical GPU index; default: 2")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--max-used-memory-mib",
        type=int,
        default=1024,
        help="refuse to run if the selected GPU already exceeds this allocation",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    gpu_attestation = _gpu_preflight(args.gpu, args.max_used_memory_mib)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    report = _build_report(args, gpu_attestation)
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(report, indent=2, sort_keys=True) + "\n").encode()
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_bytes(encoded)
    temporary.replace(output)
    print(json.dumps({"output": str(output), "sha256": _sha256_file(output), "verdict": "PASS"}))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ProofFailure as exc:
        print(f"PROOF FAILURE: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
