"""dispatch-PP's step loss is per valid token over the step, as its gradients are normalized and as the one-GPU path logs it.

It used to be a mean of the rows' means, so a short row weighed as much as a long one in the logged loss, in SFT's loss
guard and in its plateau logic, while the update itself was already per token.
"""

import numpy as np
import pytest
import torch

from tests.train.test_adam_step_count_gpu import common_yaml, make_model

pytestmark = [pytest.mark.gpu, pytest.mark.slow, pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")]


def test_step_loss_is_a_token_mean(tmp_path, monkeypatch):
    from surogate import _surogate as ext
    from surogate.core.config.loader import load_config
    from surogate.grpo.config import GRPOTrainConfig
    from surogate.grpo.trainer import GRPOTrainer

    model_dir, adapter = make_model(tmp_path)
    monkeypatch.setenv("SUROGATE_DISPATCH_STAGE_BLOCKS", "3")
    monkeypatch.setenv("SUROGATE_DISPATCH_LORA_LOCAL", "1")
    monkeypatch.setenv("SUROGATE_DISPATCH_STAGE_PLAN", "3")
    monkeypatch.setenv("SUROGATE_DISPATCH_PREFETCH_BLOCKS", "3")
    path = tmp_path / "train.yaml"
    path.write_text("\n".join(common_yaml(tmp_path, model_dir, adapter, True) + ["max_steps: 1", "save_steps: 0"]) + "\n")
    trainer = GRPOTrainer(load_config(GRPOTrainConfig, str(path)))

    # Rows of very different lengths, one over a narrow token range, so their mean losses differ.
    L = 512
    rng = np.random.default_rng(3)
    x, y = np.zeros((3, L), np.int32), np.full((3, L), -100, np.int32)
    for i, (n, top) in enumerate(((500, 500), (40, 8), (120, 60))):
        ids = rng.integers(3, 3 + top, n)
        x[i, :n] = ids
        y[i, : n - 1] = ids[1:]
    lp = np.asarray(trainer.trainer.dispatch_pp_forward_logprobs_multigpu(x.copy(), y.copy(), trainer._dpp_los, trainer._dpp_his, 3, keep_boundaries=False))
    per_row = [-lp[i][y[i] >= 0] for i in range(3)]
    token_mean = float(np.concatenate(per_row).mean())
    mean_of_means = float(np.mean([r.mean() for r in per_row]))
    assert abs(mean_of_means - token_mean) > 0.02, "the rows do not tell the two apart"

    options = ext.OptimizerConfig(optimizer="adamw_8bit", learning_rate=0.0)
    loss = float(trainer.trainer.dispatch_pp_train_step_multigpu(x, y, trainer._dpp_los, trainer._dpp_his, options, 0, False, 3))
    trainer.close()
    assert abs(loss - token_mean) < 5e-3, f"step loss {loss:.4f}, token mean {token_mean:.4f}, mean of the rows' means {mean_of_means:.4f}"
