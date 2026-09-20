"""Bounded synthetic CE geometry profile; no checkpoints or accuracy claims."""

import argparse
import datetime
import hashlib
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path


def dump(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence-length", type=int, choices=[4352], required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0,1":
        raise RuntimeError("This bounded profile owns only physical GPUs 0,1")
    args.out.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root))
    sys.path.insert(0, str(root.parent / "jev/scripts"))
    from native_surogate import load_native
    native = root / "csrc/build-candidate-proper-v2/_surogate.abi3.so"
    expected = "eee051742ae055b8233d51e42e6aa1b78c18627b74ed681335c21c317a52431a"
    if hashlib.sha256(native.read_bytes()).hexdigest() != expected:
        raise RuntimeError("Corrected native extension hash mismatch")
    load_native(str(native))
    import numpy as np

    from surogate import _surogate as sg
    from surogate.dsl.ir_builder import build_dsl_ir_for_model
    from surogate.kernels.jit_compile import compile_jit_kernels
    from surogate.utils.hf import get_model_weights_path
    from tests.distill.test_kd_gradient import opt_config, torch
    from tests.distill.test_qwen35_9b_candidate_smoke import MODEL

    def sync():
        for device in [0, 1]:
            torch.cuda.synchronize(device)

    result = {
        "status": "running", "model": str(MODEL), "native_module": str(native),
        "native_sha256": expected, "purpose": "synthetic memory/throughput only;not accuracy",
        "sequence_length": args.sequence_length, "ngpu": 2, "batch_per_gpu": 1,
        "gradient_accumulation": 2, "warmup_updates": 2, "measured_updates": 3,
        "recipe": "fp8_hybrid", "lora_rank": 32, "lora_alpha": 32,
        "lora_dtype": "fp32", "candidate_objective": "cross_entropy",
        "runtime_options": {
            "offload_master": True, "offload_grads": False, "offload_optimizer": False,
            "offload_residual": False, "cpu_training": False, "use_cuda_graphs": False,
            "shard_gradients": True, "lmhead_chunks": 1, "skip_ignored_lmhead_rows": False,
        },
        "optimizer": {"type": "adamw", "learning_rate": 1e-4, "weight_decay": 0,
                      "grad_clip": 1, "beta1": 0.9, "beta2": 0.999, "epsilon": 1e-8},
        "input": {"tokens": "arange(sequence_length)%1000+100,identical on both ranks",
                  "supervised_position": args.sequence_length - 2,
                  "gold": 101, "candidates": [102, -1, 100, -1, 101]},
        "updates": [], "saved_adapters": False,
    }
    path = args.out / "results.json"
    dump(path, result)
    sampler_log = (args.out / "memory.csv").open("w")
    sampler = subprocess.Popen([
        "nvidia-smi", "--query-gpu=timestamp,index,memory.used",
        "--format=csv,noheader,nounits", "-i", "0,1", "-lms", "20",
    ], stdout=sampler_log, stderr=subprocess.STDOUT)
    trainer = None
    try:
        config = sg.PretrainedConfig.from_pretrained(str(MODEL), "bf16")
        options = sg.RuntimeOptions(recipe="fp8_hybrid", use_cuda_graphs=False,
                                    offload_master=True, offload_grads=False,
                                    offload_optimizer=False, offload_residual=False,
                                    shard_gradients=True, cpu_training=False)
        options.dsl_ir_json = build_dsl_ir_for_model(str(MODEL))
        options.jit_kernel_manifests = compile_jit_kernels(options.dsl_ir_json)
        started = time.perf_counter()
        trainer = sg.SurogateTrainer(
            ngpu=2, config=config, options=options, batch_size=1,
            seq_len=args.sequence_length, grad_accum=2,
            memcpy_all_gather=True, memcpy_send_recv=True,
            lora_config=sg.LoRAAdapterConfig(rank=32, alpha=32, dropout=0, dtype="fp32", target_modules=["all"]),
            qlora_config=None,
        )
        trainer.import_weights(get_model_weights_path(str(MODEL)))
        sync()
        result["initialization_seconds"] = time.perf_counter() - started
        inputs = np.tile((np.arange(args.sequence_length, dtype=np.int32) % 1000 + 100)[None, :], (2, 1))
        targets = np.full_like(inputs, -100)
        ids = np.full((2, args.sequence_length, 5), -1, dtype=np.int32)
        targets[:, -2] = 101
        ids[:, -2] = [102, -1, 100, -1, 101]
        unused = np.zeros_like(ids, dtype=np.float32)
        for step in range(5):
            sync()
            start_utc = datetime.datetime.now(datetime.UTC).isoformat()
            started = time.perf_counter()
            for _ in range(2):
                trainer.step_with_kd(inputs, targets, ids, unused, top_k=5,
                                     temperature=1., kd_weight=1., ce_weight=0., candidate_only=True)
            update = trainer.update_with_config(opt_config(1e-4), step)
            sync()
            elapsed = time.perf_counter() - started
            row = {"step": step + 1, "phase": "warmup" if step < 2 else "measured",
                   "start_utc": start_utc, "seconds": elapsed, "loss": update["loss"],
                   "norm": update["norm"], "global_input_tokens": 4 * args.sequence_length,
                   "global_valid_decisions": 4}
            result["updates"].append(row)
            dump(path, result)
            print(json.dumps(row), flush=True)
            if not np.isfinite(update["loss"]) or not np.isfinite(update["norm"]):
                raise RuntimeError("Non-finite training loss/norm")
        # Optional segment names are not used for the external VRAM profile.
        # Some native builds return a non-UTF8 name here after valid updates.
        try:
            result["allocator_info"] = {str(device): trainer.get_allocator_info(device) for device in [0, 1]}
        except UnicodeDecodeError as exc:
            result["allocator_diagnostic_error"] = {"type": type(exc).__name__, "message": str(exc)}
        measured = [row["seconds"] for row in result["updates"] if row["phase"] == "measured"]
        result["measured_summary"] = {
            "mean_seconds_per_update": statistics.mean(measured),
            "median_seconds_per_update": statistics.median(measured),
            "min_seconds_per_update": min(measured), "max_seconds_per_update": max(measured),
            "global_input_tokens_per_second": 4 * args.sequence_length / statistics.mean(measured),
            "global_decisions_per_second": 4 / statistics.mean(measured),
        }
        result["status"] = "complete"
    except BaseException as exc:
        result["status"] = "failed"
        result["error"] = {"type": type(exc).__name__, "message": str(exc)}
        raise
    finally:
        sampler.terminate()
        try:
            sampler.wait(timeout=5)
        except subprocess.TimeoutExpired:
            sampler.kill()
            sampler.wait()
        sampler_log.close()
        peak = {"0": 0, "1": 0}
        count = 0
        for line in (args.out / "memory.csv").read_text().splitlines():
            fields = line.split(",")
            if len(fields) == 3 and fields[1].strip() in peak:
                try:
                    memory = int(fields[2].strip())
                except ValueError:
                    continue
                key = fields[1].strip()
                peak[key] = max(peak[key], memory)
                count += 1
        result["vram"] = {"sampled_peak_mib": peak, "sample_rows": count,
                          "requested_sample_period_ms": 20,
                          "scope": "whole-GPU used memory including initialization;sampled lower bound,not exact allocation watermark"}
        dump(path, result)
        del trainer


if __name__ == "__main__":
    main()
