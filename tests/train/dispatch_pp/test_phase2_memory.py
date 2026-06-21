"""Phase-2 gate: the dispatch-PP GPU weight-residency (memory-scaling) invariant.

The point of streaming block work-weights from pinned CPU is that a GPU's resident
weight footprint is bounded by the streaming double-buffer slot count (a small
constant), NOT by the model's layer count. This test pins that quantitatively via
``dispatch_pp_weight_residency`` introspection:

  resident  (offload_master=False): masters live on the GPU -> the whole model's
            weight bytes are device-resident; there is no weight manager / no
            streaming slots (slot_count == 0).
  streaming (offload_master=True):  masters are pinned on the CPU and blocks stream
            through ``kNumPrefetchBuffers`` double-buffer slots -> the device holds
            only ``slot_count`` blocks' work-weights, far less than the full model's
            ``NUM_LAYERS`` blocks.
"""

import gc
import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
sg = pytest.importorskip("surogate._surogate")

from surogate.dsl.ir_builder import build_dsl_ir_for_model
from surogate.utils.hf import get_model_weights_path
from tests.test_onboarding_qwen3 import (
    BATCH,
    NUM_LAYERS,
    SEQ_LEN,
    prepare_mini_model,
    resolve_model_path,
)

_MIN_FREE_BYTES = 6 * 1024**3  # the DSL stack alone is ~3 GB


def _enough_free_gpu():
    if torch.cuda.device_count() < 1:
        return False
    free, _ = torch.cuda.mem_get_info(0)
    return free >= _MIN_FREE_BYTES


pytestmark = pytest.mark.skipif(
    not _enough_free_gpu(), reason=f"needs 1 GPU with >= {_MIN_FREE_BYTES // 1024**3} GiB free"
)


def _residency(offload_master: bool):
    snapshot = resolve_model_path()
    if snapshot is None:
        pytest.skip("Qwen3 weights not found. Set QWEN3_MODEL_PATH or cache Qwen/Qwen3-0.6B.")
    model_dir = prepare_mini_model(snapshot)
    cfg = sg.PretrainedConfig.from_pretrained(str(model_dir), "bf16")
    opts = sg.RuntimeOptions(
        offload_residual=False,
        use_cuda_graphs=False,
        offload_master=offload_master,
        offload_grads=False,
        offload_optimizer=False,
        shard_gradients=False,
        use_zero_copy=False,
    )
    opts.dsl_ir_json = build_dsl_ir_for_model(str(model_dir))
    trainer = sg.SurogateTrainer(
        ngpu=1,
        config=cfg,
        options=opts,
        batch_size=BATCH,
        seq_len=SEQ_LEN,
        grad_accum=1,
        memcpy_all_gather=True,
        memcpy_send_recv=True,
        lora_config=None,
        qlora_config=None,
    )
    trainer.import_weights(get_model_weights_path(str(model_dir)))
    res = dict(trainer.dispatch_pp_weight_residency())
    del trainer
    gc.collect()
    torch.cuda.empty_cache()
    return res


def test_resident_has_no_streaming_slots():
    """Fully-resident weights => no weight manager / no streaming double-buffer."""
    resident = _residency(offload_master=False)
    assert resident["prefetch_slot_count"] == 0
    assert resident["gpu_prefetch_buffer_bytes"] == 0


def test_streaming_bounds_gpu_weight_residency():
    """Streaming holds only `slot_count` blocks on the device — a small constant,
    far fewer than the model's NUM_LAYERS blocks. This is the dispatch-PP memory
    invariant: GPU weight residency scales with the slot count, not the layer count."""
    streaming = _residency(offload_master=True)

    slots = streaming["prefetch_slot_count"]
    assert slots > 0
    assert slots < NUM_LAYERS  # the whole point: fewer resident blocks than layers

    prefetch = streaming["gpu_prefetch_buffer_bytes"]
    assert prefetch > 0
    per_block = prefetch // slots
    assert per_block > 0
    # The block double-buffer holds exactly `slots` blocks...
    assert prefetch == per_block * slots
    # ...far less than keeping every layer resident (the resident-weights cost).
    assert prefetch < per_block * NUM_LAYERS

    # The streaming GPU footprint (counted by total_persistent_bytes) includes the
    # prefetch slots; masters are pinned on the CPU so they are excluded.
    assert streaming["total_persistent_bytes"] >= prefetch
