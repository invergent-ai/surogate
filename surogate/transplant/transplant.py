"""Vocabulary transplantation for cross-tokenizer knowledge distillation.

Wraps `mergekit-tokensurgeon` (Arcee's tokenizer-transplant tool) as a
subprocess: the student model's embedding and LM-head rows are reconstructed
for the teacher's vocabulary (shared tokens copied exactly, new tokens
approximated — default method OMP with k=64, per arXiv:2506.06607). After the
transplant the student uses the teacher's tokenizer, so the existing
same-tokenizer KD pipeline (tokenize → distill-capture → sft) runs unchanged.

mergekit is an OPTIONAL dependency, invoked strictly as an external command:
it is LGPL-3.0-licensed and must not be vendored or imported into surogate.
Install it with `pip install mergekit` (or run via `uvx --from mergekit ...`
in a separate environment to avoid its pydantic/click pins).

A `transplant_manifest.json` is written next to the output model so the
reverse transplant (back to the student's native tokenizer after KD training)
can be driven by `surogate transplant-tokenizer --restore`.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import time

from surogate.utils.logger import get_logger

logger = get_logger()

TRANSPLANT_MANIFEST = "transplant_manifest.json"

_APPROXIMATION_METHODS = (
    "omp",
    "common_interpolation",
    "subword",
    "mean",
    "zero",
    "randn",
    "john_hewitt",
    "landmark_pca",
    "sparse_token_basis",
    "mp_rope",
)

_INSTALL_GUIDANCE = (
    "mergekit-tokensurgeon was not found on PATH. Vocabulary transplantation uses "
    "mergekit (LGPL-3.0) as an external tool; install it with `pip install mergekit` "
    "or run surogate transplant-tokenizer inside an environment where "
    "`uvx --from mergekit mergekit-tokensurgeon` works. mergekit is intentionally NOT "
    "a surogate dependency (license + pinned-dependency isolation)."
)


def _find_tokensurgeon() -> list[str]:
    exe = shutil.which("mergekit-tokensurgeon")
    if exe:
        return [exe]
    raise RuntimeError(_INSTALL_GUIDANCE)


def _resolve_model_dir(model: str) -> str:
    """Local directory as-is; otherwise download the HF snapshot (merge.py idiom)."""
    if os.path.isdir(model):
        return model
    from huggingface_hub import snapshot_download

    logger.info(f"Downloading model from HuggingFace: {model}")
    return snapshot_download(model)


def _validate_output_model(out_dir: str) -> dict:
    config_path = os.path.join(out_dir, "config.json")
    if not os.path.exists(config_path):
        raise RuntimeError(
            f"tokensurgeon produced no config.json in {out_dir}; the transplant did not complete."
        )
    with open(config_path) as f:
        config = json.load(f)

    has_tokenizer = any(
        os.path.exists(os.path.join(out_dir, name)) for name in ("tokenizer.json", "tokenizer.model")
    )
    if not has_tokenizer:
        raise RuntimeError(
            f"tokensurgeon output {out_dir} has no tokenizer.json/tokenizer.model — the donor "
            "tokenizer was not written; re-run with --trust-remote-code if the donor needs it."
        )
    has_weights = any(name.endswith(".safetensors") for name in os.listdir(out_dir))
    if not has_weights:
        raise RuntimeError(f"tokensurgeon output {out_dir} contains no safetensors weight files.")

    if config.get("tie_word_embeddings"):
        logger.warning(
            "Transplanted model has tie_word_embeddings=true: input embeddings and LM head share "
            "one matrix, so the transplant solves a compromise between the two approximations. "
            "Small tied-embedding students heal, but expect a few hundred million KD tokens "
            "before quality recovers."
        )
    return config


def _run_tokensurgeon(
    base_model: str,
    donor_model: str,
    out_dir: str,
    method: str,
    k: int,
    device: str | None,
    trust_remote_code: bool,
    extra_args: list[str] | None,
) -> None:
    if method not in _APPROXIMATION_METHODS:
        raise ValueError(
            f"Unknown approximation method '{method}'. Supported: {', '.join(_APPROXIMATION_METHODS)}."
        )
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}.")

    cmd = _find_tokensurgeon() + [
        base_model,
        donor_model,
        out_dir,
        "--approximation-method",
        method,
        "--k",
        str(k),
    ]
    if device:
        cmd += ["--device", device]
    if trust_remote_code:
        cmd += ["--trust-remote-code"]
    if extra_args:
        cmd += list(extra_args)

    logger.info(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        tail = (result.stderr or result.stdout or "").strip().splitlines()[-15:]
        raise RuntimeError(
            f"mergekit-tokensurgeon failed (exit {result.returncode}):\n" + "\n".join(tail)
        )


def run_transplant(
    student_model: str,
    teacher_model: str,
    out_dir: str,
    method: str = "omp",
    k: int = 64,
    device: str | None = None,
    trust_remote_code: bool = False,
    extra_args: list[str] | None = None,
) -> str:
    """Transplant the teacher's tokenizer onto the student model.

    Returns the output model directory. The output carries the student's
    weights with embedding/LM-head rows rebuilt for the teacher vocabulary,
    plus the teacher's tokenizer files — ready for the same-tokenizer KD
    pipeline (`surogate tokenize` / `distill-capture` / `sft` against it).
    """
    student_dir = _resolve_model_dir(student_model)
    teacher_dir = _resolve_model_dir(teacher_model)

    _run_tokensurgeon(
        student_dir, teacher_dir, out_dir, method, k, device, trust_remote_code, extra_args
    )
    _validate_output_model(out_dir)

    manifest = {
        "version": 1,
        "student_model": student_model,
        "student_dir": os.path.abspath(student_dir),
        "teacher_model": teacher_model,
        "teacher_dir": os.path.abspath(teacher_dir),
        "method": method,
        "k": k,
        "created_unix": int(time.time()),
    }
    with open(os.path.join(out_dir, TRANSPLANT_MANIFEST), "w") as f:
        json.dump(manifest, f, indent=2)
    logger.info(
        f"Transplant complete: {out_dir} now uses the tokenizer of {teacher_model}. "
        f"Next: point your KD config's `model:` at it, then run `surogate distill-capture` "
        f"and `surogate sft`. Restore the native tokenizer afterwards with "
        f"`surogate transplant-tokenizer --restore {out_dir}/{TRANSPLANT_MANIFEST} ...`."
    )
    return out_dir


def run_transplant_back(
    distilled_model: str,
    manifest_path: str,
    out_dir: str,
    device: str | None = None,
    trust_remote_code: bool = False,
    extra_args: list[str] | None = None,
) -> str:
    """Transplant a distilled model back to the original student tokenizer.

    `distilled_model` is the KD-trained model (teacher vocabulary);
    `manifest_path` is the transplant_manifest.json written by run_transplant.
    The original student model acts as the tokenizer donor. A short SFT on
    native-template data afterwards is recommended (chat-template semantics
    shift back with the tokenizer).
    """
    with open(manifest_path) as f:
        manifest = json.load(f)
    donor = manifest.get("student_dir") or manifest.get("student_model")
    if not donor:
        raise ValueError(f"Manifest {manifest_path} has no student_model/student_dir entry.")
    if not os.path.isdir(donor):
        donor = _resolve_model_dir(manifest["student_model"])

    distilled_dir = _resolve_model_dir(distilled_model)
    _run_tokensurgeon(
        distilled_dir,
        donor,
        out_dir,
        manifest.get("method", "omp"),
        int(manifest.get("k", 64)),
        device,
        trust_remote_code,
        extra_args,
    )
    _validate_output_model(out_dir)
    logger.info(
        f"Restored native tokenizer: {out_dir}. A short healing SFT on native-template data "
        "is recommended before release."
    )
    return out_dir
