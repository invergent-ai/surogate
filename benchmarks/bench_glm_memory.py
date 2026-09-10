"""Measure native GLM training allocation after two updates on a small checkpoint.

Run each mode in a fresh process on an otherwise idle GPU:
CUDA_VISIBLE_DEVICES=7 python benchmarks/bench_glm_memory.py
CUDA_VISIBLE_DEVICES=7 python benchmarks/bench_glm_memory.py --tiled

Reports device allocation after warmup, not a sampled transient peak. CUDA and
library caches are included; these numbers do not predict full-model memory.
"""

import argparse
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    from examples.sft.glm.create_dummy import create_dummy
    from surogate import _surogate as ext
    from surogate.dsl.ir_builder import build_dsl_ir_for_model
    from surogate.kernels.jit_compile import compile_jit_kernels

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--length", type=int, default=4096)
    parser.add_argument("--tiled", action="store_true")
    parser.add_argument("--graphs", action="store_true")
    args = parser.parse_args()
    if args.length <= 0 or args.length % 16:
        parser.error("--length must be a positive multiple of 16")

    with tempfile.TemporaryDirectory(prefix="surogate-glm-memory-") as directory:
        root = Path(directory)
        create_dummy(root, index_topk=32, max_sequence_length=args.length)
        options = ext.RuntimeOptions(
            recompute="true",
            long_context=args.tiled,
            use_cuda_graphs=args.graphs,
            master_dtype="bf16",
            offload_master=False,
            offload_grads=False,
            offload_optimizer=False,
        )
        options.dsl_ir_json = build_dsl_ir_for_model(str(root))
        options.jit_kernel_manifests = compile_jit_kernels(options.dsl_ir_json)
        torch.cuda.synchronize()
        free_before, _ = torch.cuda.mem_get_info()
        trainer = ext.SurogateTrainer(
            ngpu=1,
            config=ext.PretrainedConfig.from_pretrained(str(root), "bf16"),
            options=options,
            batch_size=1,
            seq_len=args.length,
            grad_accum=1,
            lora_config=ext.LoRAAdapterConfig(rank=8, alpha=13, dropout=0, dtype="bf16", target_modules=["all"]),
        )
        trainer.import_weights(str(root / "model.safetensors"))
        ids = np.random.default_rng(83).integers(3, 259, (1, args.length), dtype=np.int32)
        targets = np.roll(ids, -1, axis=1).copy()
        targets[:, -1] = -100
        positions = np.arange(args.length, dtype=np.int32)[None]
        scales = (targets != -100).astype(np.float32)
        scales /= scales.sum()
        for step in range(2):
            trainer.step_with_custom_loss(ids, targets, scales, position_ids=positions)
            result = trainer.update_with_config(ext.OptimizerConfig(learning_rate=1e-4), step + 1)
            if not np.isfinite(result["norm"]) or result["norm"] <= 0:
                raise RuntimeError(f"Invalid gradient norm: {result}")
        torch.cuda.synchronize()
        free_after, _ = torch.cuda.mem_get_info()
        arenas = trainer.get_debug_arena_summary()
        print(
            json.dumps(
                {
                    "length": args.length,
                    "tiled": args.tiled,
                    "graphs": args.graphs,
                    "device": torch.cuda.get_device_name(),
                    "allocated_delta_mib": (free_before - free_after) / 2**20,
                    "arenas": {name: value for name, value in arenas.items() if name.endswith("_bytes")},
                    "gradient_norm": result["norm"],
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
