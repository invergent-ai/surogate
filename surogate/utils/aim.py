import math
import os
from typing import Optional, Dict, Any
import numpy as np
import aim
import torch
from transformers import TrainerCallback

def _to_number(v):
    # torch tensors
    if isinstance(v, torch.Tensor):
        if v.numel() == 1:
            v = v.item()
        else:
            return None
    # numpy scalars / arrays
    if isinstance(v, np.ndarray):
        if v.size == 1:
            v = v.item()
        else:
            return None
    elif isinstance(v, (np.floating, np.integer)):
        v = v.item()

    # plain python
    if isinstance(v, (int, float)):
        return float(v)
    return None

def _clean_name(name: str) -> str:
    return name.replace("/", ".").replace(" ", "_")

class AimCallback(TrainerCallback):
    def __init__(self, repo: Optional[str] = None, experiment: Optional[str] = None):
        self.repo = repo or os.environ.get("AIM_REPO", ".")
        self.experiment = experiment
        self.run = None

    def on_train_begin(self, args, state, control, **kwargs):
        self.run = aim.Run(repo=self.repo, experiment=self.experiment)
        # Basic metadata
        self.run["hparams"] = {
            "per_device_train_batch_size": args.per_device_train_batch_size,
            "learning_rate": args.learning_rate,
            "num_train_epochs": args.num_train_epochs,
        }

    def _track_dict(self, d: Dict[str, Any], step: int, phase: str):
        for k, v in (d or {}).items():
            val = _to_number(v)
            if val is None or math.isnan(val) or math.isinf(val):
                continue
            name = _clean_name(k)
            # add a phase context so you can filter in Aim (train vs eval)
            self.run.track(val, name=name, step=int(step), context={"phase": phase})

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if not self.run or not metrics:
            return
        step = metrics.get("eval_step", state.global_step)
        self._track_dict(metrics, step=step, phase="eval")

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not self.run or not logs:
            return
        self._track_dict(logs, step=state.global_step, phase="train")

    def on_train_end(self, args, state, control, **kwargs):
        if self.run:
            self.run.close()


