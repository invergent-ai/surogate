"""Serve GRPO rollouts from this repository's own engine instead of vLLM.

The orchestrator talks to an inference server over HTTP and nothing else: the
OpenAI chat surface for the rollouts, `/tokenize` and `/v1/chat/completions/tokens`
for stitching a multi-turn trajectory, and `/load_lora_adapter` for the adapter the
trainer writes after every step. Our engine speaks all of them, so swapping the
backend is a matter of translating the config into its flags and handing the
process over.

`execv` rather than a subprocess: `split.py` puts this child in its own process
group and reaps that group on shutdown, so the engine must *be* this process
rather than hide behind it -- otherwise the wrapper dies and the engine is what
gets left holding the GPU.
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

from surogate.core.config.grpo_inference_config import GRPOInferenceConfig


def _cli() -> str:
    """The `surogate` entry point, which converts a checkpoint before serving it.

    The engine binary itself takes a built `.sinfer` artifact; the CLI is what turns
    a HuggingFace directory or a GGUF into one first, and caches the result.
    """
    candidate = Path(sys.executable).with_name("surogate")
    if candidate.is_file():
        return str(candidate)
    found = shutil.which("surogate")
    if found is None:
        raise RuntimeError(
            "the surogate CLI is not on PATH and is not beside this interpreter; "
            "the surogate inference backend needs it to build and serve the artifact"
        )
    return found


def build_argv(config: GRPOInferenceConfig) -> list[str]:
    """The engine command line this config asks for. Separated so a test can read it
    without starting a server."""
    if not config.model:
        raise ValueError("the surogate inference backend needs `model` set")
    argv: list[str] = [_cli(), "serve", str(config.model)]

    # The orchestrator asserts that /v1/models lists the name it was configured
    # with, and that name is the checkpoint's -- not the identity our artifact
    # carries. Serve under the name the caller already uses.
    argv += ["--served-model-name", str(config.model)]
    if config.port is not None:
        argv += ["--port", str(config.port)]
    if config.host:
        argv += ["--host", str(config.host)]
    if config.max_model_len is not None:
        argv += ["--max-model-len", str(config.max_model_len)]
    if config.max_num_seqs is not None:
        argv += ["--max-num-seqs", str(config.max_num_seqs)]
    if config.kv_cache_dtype:
        argv += ["--kv-dtype", str(config.kv_cache_dtype)]
    if config.enable_lora:
        argv += ["--enable-lora"]
        if config.max_loras is not None:
            argv += ["--max-loras", str(config.max_loras)]
        if config.max_lora_rank is not None:
            argv += ["--max-lora-rank", str(config.max_lora_rank)]
    if config.seed is not None:
        argv += ["--seed", str(config.seed)]

    # One engine per visible device set: split.py already gave this child its own
    # CUDA_VISIBLE_DEVICES, so device 0 here is the first card it was granted.
    argv += ["--device", "0"]
    return argv


def server(config: GRPOInferenceConfig) -> None:
    argv = build_argv(config)
    print(f"surogate-engine backend: {' '.join(argv)}", flush=True)
    os.execv(argv[0], argv)
