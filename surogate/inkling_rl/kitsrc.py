"""The kit's own trainer-side sources, byte copies pinned by sha256, loaded WITHOUT importing their modules.

kit/loss_041f2c0c.py.txt      = the deployed GSPO loss (pkg-async-r3's loss.py, the kit env's sha_loss 041f2c0cc9652d8b;
                                source rl/trainer-check-2026-11-06/trainer-patch/new/loss.py)
kit/config_ea32d191.py.txt    = the GRPOLossConfig of the same patch set (trainer-check new/config.py ea32d1918ca33dbc:
                                gspo, seq_ratio_old, seq_clip_*, behavior_is_cap, aggregation; the deployed r3 config
                                2b766bd6 adds only behavior_is_level, which step.py implements)
kit/sg_types_ow_a102d7f.py.txt = the loop's copy of the trainer's transport types (recipe/loop/sg_types.py at ow-template
                                a102d7f = trainer-gspo 49fd6762): TrainingSample / TrainingBatch, msgspec array_like
                                (POSITIONAL: the schema must match the loop's byte for byte)
Loading the functions / classes by AST (as the kit's own trainer-patches/gspo-token-aggregation/test_patch.py does) keeps
the arithmetic the kit's and avoids importing surogate.grpo.* of this tree (a different trainer lineage).
"""
from __future__ import annotations

import ast
import dataclasses
import hashlib
import logging
import os
from typing import Literal

import numpy as np

HERE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kit")
PINS = {
    "loss_041f2c0c.py.txt": "041f2c0cc9652d8b7e6999a3b6b1be4c75db560232568d4f3db133915c8b87fc",
    "config_ea32d191.py.txt": "ea32d1918ca33dbc0cd75d4fb4c0f2d12a13b72af8a92228294fd51e9302bd3c",
    "sg_types_ow_a102d7f.py.txt": "6aeeb60ea7dfc916efa912de77a14e69f7e2539542733582ca923f48966419f6",
}


def _read_pinned(name: str) -> str:
    path = os.path.join(HERE, name)
    raw = open(path, "rb").read()
    got = hashlib.sha256(raw).hexdigest()
    if got != PINS[name]:
        raise RuntimeError(f"{path}: sha256 {got} != pinned {PINS[name]}")
    return raw.decode()


def _load(name: str, names: list[str], env: dict) -> dict:
    src = _read_pinned(name)
    tree = ast.parse(src)
    keep = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names]
    missing = set(names) - {n.name for n in keep}
    if missing:
        raise RuntimeError(f"{name}: missing {missing}")
    exec(compile(ast.Module(body=keep, type_ignores=[]), os.path.join(HERE, name), "exec"), env)
    return env


_cache: dict = {}


def loss_api():
    """(GRPOLossConfig, compute_grpo_per_token_grads) exactly as the kit's trainer has them."""
    if "loss" not in _cache:
        cfg = _load("config_ea32d191.py.txt", ["GRPOLossConfig"],
                    {"dataclass": dataclasses.dataclass, "Literal": Literal, "logger": logging.getLogger("kit.config"),
                     "__name__": "surogate.inkling_rl.kit"})["GRPOLossConfig"]
        fns = _load("loss_041f2c0c.py.txt", ["_safe_mean", "_sigmoid", "_compute_sample_grads", "compute_grpo_per_token_grads"],
                    {"np": np, "_logger": logging.getLogger("kit.loss"), "GRPOLossConfig": cfg})
        _cache["loss"] = (cfg, fns["compute_grpo_per_token_grads"])
    return _cache["loss"]


def types_api():
    """(TrainingSample, TrainingBatch) structs of the kit's transport schema (needs msgspec)."""
    if "types" not in _cache:
        import msgspec

        env = {"msgspec": msgspec, "__name__": "surogate.inkling_rl.kit_types"}
        src = _read_pinned("sg_types_ow_a102d7f.py.txt")
        exec(compile(src, os.path.join(HERE, "sg_types_ow_a102d7f.py.txt"), "exec"), env)
        _cache["types"] = (env["TrainingSample"], env["TrainingBatch"])
    return _cache["types"]
