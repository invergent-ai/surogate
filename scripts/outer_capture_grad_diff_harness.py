#!/usr/bin/env python3
"""Compare eager and outer-captured LoRA gradients on identical data.

This harness runs one logical optimizer step up to backward only:

* eager path: ``grad_accum`` calls to ``trainer.step(...)``
* captured path: one ``trainer.train_step_graphed(...)`` with
  ``SUROGATE_FULLSTEP_GRAPH_MODE=fwd_bwd``

It then dumps and compares the post-backward LoRA A/B gradient tensors.
The default model setup reuses the Qwen3.5 mini-model preparation from the
onboarding test, which keeps the repro small while exercising the same DSL
and LoRA paths as the full checkpoint.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import re
import sys
from pathlib import Path

import numpy as np


@contextlib.contextmanager
def patched_env(updates: dict[str, str | None]):
    old: dict[str, str | None] = {}
    for key, value in updates.items():
        old[key] = os.environ.get(key)
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value
    try:
        yield
    finally:
        for key, value in old.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def import_runtime():
    root = repo_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    import torch
    import surogate._surogate as _surogate
    from surogate.dsl.ir_builder import build_dsl_ir_for_model
    from surogate.utils.hf import get_model_weights_path

    return torch, _surogate, build_dsl_ir_for_model, get_model_weights_path


def resolve_qwen35_model(model_dir_arg: str | None, use_mini: bool) -> Path:
    if model_dir_arg:
        model_dir = Path(model_dir_arg).expanduser().resolve()
        if not model_dir.exists():
            raise FileNotFoundError(model_dir)
        return model_dir

    root = repo_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    # Reuse the existing mini-model helper so this harness stays aligned with
    # the Qwen3.5 onboarding test's checkpoint slicing and config fixes.
    from tests import test_onboarding_qwen3_5 as q35

    snapshot = q35.resolve_model_path()
    if snapshot is None:
        raise FileNotFoundError("Qwen3.5 weights not found. Set QWEN3_5_MODEL_PATH or pass --model-dir.")
    return q35.prepare_mini_model(snapshot) if use_mini else Path(snapshot)


def read_vocab_size(model_dir: Path) -> int:
    cfg = json.loads((model_dir / "config.json").read_text())
    text_cfg = cfg.get("text_config", {})
    return int(text_cfg.get("vocab_size", cfg.get("vocab_size")))


def make_batch(
    vocab_size: int,
    batch: int,
    seq_len: int,
    grad_accum: int,
    seed: int,
    packed: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    rows = batch * grad_accum
    inputs = rng.integers(0, vocab_size, size=(rows, seq_len), dtype=np.int32)
    targets = np.empty_like(inputs)
    targets[:, :-1] = inputs[:, 1:]
    targets[:, -1] = -100

    position_ids = np.empty((rows, seq_len), dtype=np.int32)
    for row in range(rows):
        if not packed:
            position_ids[row] = np.arange(seq_len, dtype=np.int32)
            continue

        # Vary the split point by row so multi-ms capture sees more than one
        # doc-boundary topology while keeping the batch deterministic.
        lo = max(1, seq_len // 3)
        hi = max(lo + 1, (2 * seq_len) // 3)
        cut = lo + (row % (hi - lo))
        position_ids[row, :cut] = np.arange(cut, dtype=np.int32)
        position_ids[row, cut:] = np.arange(seq_len - cut, dtype=np.int32)
        targets[row, cut - 1] = -100
    return inputs, targets, position_ids


def make_lora_config(_surogate, rank: int, alpha: int, dtype: str, target_modules: list[str]):
    return _surogate.LoRAAdapterConfig(
        rank=rank,
        alpha=alpha,
        dropout=0.0,
        dtype=dtype,
        target_modules=target_modules,
        use_rslora=False,
    )


def make_trainer(
    model_dir: Path,
    batch: int,
    seq_len: int,
    grad_accum: int,
    use_cuda_graphs: bool,
    lora_rank: int,
    lora_alpha: int,
    lora_dtype: str,
    target_modules: list[str],
):
    _torch, _surogate, build_dsl_ir_for_model, get_model_weights_path = import_runtime()
    from surogate.kernels.jit_compile import compile_jit_kernels

    cfg = _surogate.PretrainedConfig.from_pretrained(str(model_dir), "bf16")
    ir_json = build_dsl_ir_for_model(str(model_dir))
    opts = _surogate.RuntimeOptions(
        offload_residual=False,
        use_cuda_graphs=use_cuda_graphs,
        offload_master=False,
        offload_grads=False,
        offload_optimizer=False,
        shard_gradients=True,
        use_zero_copy=False,
        doc_masking=True,
    )
    opts.dsl_ir_json = ir_json
    manifests = compile_jit_kernels(ir_json)
    if manifests:
        opts.jit_kernel_manifests = manifests

    trainer = _surogate.SurogateTrainer(
        ngpu=1,
        config=cfg,
        options=opts,
        batch_size=batch,
        seq_len=seq_len,
        grad_accum=grad_accum,
        memcpy_all_gather=True,
        memcpy_send_recv=True,
        lora_config=make_lora_config(_surogate, lora_rank, lora_alpha, lora_dtype, target_modules),
        qlora_config=None,
    )
    trainer.import_weights(get_model_weights_path(str(model_dir)))
    return trainer


def clone_lora_grads(trainer) -> dict[str, np.ndarray]:
    torch, _surogate, _build, _weights = import_runtime()
    del _surogate, _build, _weights
    out: dict[str, np.ndarray] = {}
    raw = trainer.get_lora_gradients(0)
    for name, arr in raw.items():
        tensor = torch.utils.dlpack.from_dlpack(arr).detach().float().cpu().contiguous()
        out[name] = tensor.numpy().copy()
    return out


def clone_lora_weights(trainer) -> dict[str, np.ndarray]:
    torch, _surogate, _build, _weights = import_runtime()
    del _surogate, _build, _weights
    out: dict[str, np.ndarray] = {}
    raw = trainer.get_lora_weights(0)
    for name, arr in raw.items():
        tensor = torch.utils.dlpack.from_dlpack(arr).detach().float().cpu().contiguous()
        out[name] = tensor.numpy().copy()
    return out


def run_eager(
    args,
    model_dir: Path,
    data: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    inputs, targets, position_ids = data
    trainer = make_trainer(
        model_dir,
        args.batch,
        args.seq_len,
        args.grad_accum,
        use_cuda_graphs=False,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dtype=args.lora_dtype,
        target_modules=args.target_modules,
    )
    initial_weights = clone_lora_weights(trainer)
    for ms in range(args.grad_accum):
        start = ms * args.batch
        end = start + args.batch
        trainer.step(inputs[start:end], targets[start:end], position_ids[start:end])
    grads = clone_lora_grads(trainer)
    del trainer
    return grads, initial_weights


def run_captured(
    args,
    model_dir: Path,
    data: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    inputs, targets, position_ids = data
    _torch, _surogate, _build, _weights = import_runtime()
    trainer = make_trainer(
        model_dir,
        args.batch,
        args.seq_len,
        args.grad_accum,
        use_cuda_graphs=True,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dtype=args.lora_dtype,
        target_modules=args.target_modules,
    )
    initial_weights = clone_lora_weights(trainer)
    opt = _surogate.OptimizerConfig(
        optimizer=args.optimizer,
        learning_rate=args.learning_rate,
        weight_decay=0.0,
        grad_clip=args.grad_clip,
        adamw_beta1=0.9,
        adamw_beta2=0.95,
    )
    env = {
        "SUROGATE_FULLSTEP_GRAPH_MODE": "fwd_bwd",
        "SUROGATE_DEBUG_FORCE_OUTER_CAPTURE": "1",
    }
    if args.disable_pad_to_max:
        env["SUROGATE_DISABLE_PAD_TO_MAX"] = "1"
    with patched_env(env):
        trainer.train_step_graphed(inputs, targets, position_ids, opt, 0)
    grads = clone_lora_grads(trainer)
    del trainer
    return grads, initial_weights


_LAYER_RE = re.compile(r"layers\.(\d+)")


def sort_key(name: str):
    match = _LAYER_RE.search(name)
    layer = int(match.group(1)) if match else 10**9
    suffix_rank = 0 if "lora_A" in name else 1 if "lora_B" in name else 2
    return (layer, name, suffix_rank)


def compare(eager: dict[str, np.ndarray], captured: dict[str, np.ndarray], rtol: float, atol: float):
    names = sorted(set(eager) | set(captured), key=sort_key)
    rows = []
    for name in names:
        if name not in eager or name not in captured:
            rows.append(
                {
                    "name": name,
                    "status": "missing",
                    "in_eager": name in eager,
                    "in_captured": name in captured,
                }
            )
            continue
        a = eager[name].astype(np.float32, copy=False)
        b = captured[name].astype(np.float32, copy=False)
        diff = b - a
        rms = float(np.sqrt(np.mean(diff * diff))) if diff.size else 0.0
        max_abs = float(np.max(np.abs(diff))) if diff.size else 0.0
        eager_norm = float(np.linalg.norm(a.reshape(-1)))
        captured_norm = float(np.linalg.norm(b.reshape(-1)))
        rel_rms = rms / max(float(np.sqrt(np.mean(a * a))) if a.size else 0.0, 1e-12)
        allclose = bool(np.allclose(a, b, rtol=rtol, atol=atol))
        rows.append(
            {
                "name": name,
                "status": "ok" if allclose else "diff",
                "shape": list(a.shape),
                "eager_norm": eager_norm,
                "captured_norm": captured_norm,
                "norm_ratio": captured_norm / max(eager_norm, 1e-12),
                "rel_rms": rel_rms,
                "rms": rms,
                "max_abs": max_abs,
            }
        )
    return rows


def dump_results(
    out_dir: Path, eager: dict[str, np.ndarray], captured: dict[str, np.ndarray], rows: list[dict]
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_dir / "eager_lora_grads.npz", **eager)
    np.savez_compressed(out_dir / "captured_lora_grads.npz", **captured)
    (out_dir / "summary.json").write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n")


def assert_initial_weights_match(eager: dict[str, np.ndarray], captured: dict[str, np.ndarray]) -> None:
    rows = compare(eager, captured, rtol=0.0, atol=0.0)
    failures = [r for r in rows if r["status"] != "ok"]
    if failures:
        first = failures[0]
        raise RuntimeError(
            "Initial LoRA weights differ before training; gradient comparison would be invalid.\n"
            + json.dumps(first, indent=2, sort_keys=True)
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", default=None, help="HF model snapshot/mini-model directory.")
    parser.add_argument("--no-mini", action="store_true", help="Use the resolved Qwen3.5 snapshot directly.")
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--grad-accum", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--unpacked", action="store_true", help="Use monotonic position_ids with no doc boundaries.")
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dtype", default="bf16")
    parser.add_argument(
        "--target-modules",
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
        type=lambda s: [x.strip() for x in s.split(",") if x.strip()],
    )
    parser.add_argument("--optimizer", default="adamw_8bit")
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--rtol", type=float, default=1e-2)
    parser.add_argument("--atol", type=float, default=1e-4)
    parser.add_argument("--top", type=int, default=20)
    parser.add_argument("--dump-dir", default="tmp/outer_capture_grad_diff")
    parser.add_argument("--disable-pad-to-max", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    model_dir = resolve_qwen35_model(args.model_dir, use_mini=not args.no_mini)
    vocab_size = read_vocab_size(model_dir)
    data = make_batch(
        vocab_size=vocab_size,
        batch=args.batch,
        seq_len=args.seq_len,
        grad_accum=args.grad_accum,
        seed=args.seed,
        packed=not args.unpacked,
    )

    print(
        f"model={model_dir} batch={args.batch} seq_len={args.seq_len} "
        f"grad_accum={args.grad_accum} packed={not args.unpacked}"
    )
    eager, eager_initial = run_eager(args, model_dir, data)
    captured, captured_initial = run_captured(args, model_dir, data)
    assert_initial_weights_match(eager_initial, captured_initial)
    rows = compare(eager, captured, rtol=args.rtol, atol=args.atol)
    dump_results(Path(args.dump_dir), eager, captured, rows)

    failures = [r for r in rows if r["status"] != "ok"]
    diffs = [r for r in rows if r["status"] == "diff"]
    print(f"compared={len(rows)} failures={len(failures)} dump_dir={args.dump_dir}")
    if failures:
        first = failures[0]
        print("first_failure:")
        print(json.dumps(first, indent=2, sort_keys=True))

    if diffs:
        print("largest_diffs:")
        for row in sorted(diffs, key=lambda r: r["rel_rms"], reverse=True)[: args.top]:
            print(
                f"  {row['name']} rel_rms={row['rel_rms']:.4e} "
                f"rms={row['rms']:.4e} max={row['max_abs']:.4e} "
                f"norm_ratio={row['norm_ratio']:.4e}"
            )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
