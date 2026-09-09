# `qwen3_vl` Vision parity tool

Compares every plane the engine's vision tower produces against the tower the checkpoint
ships — the merged output and each deepstack tap.

```bash
python -m surogate.serve.tools.parity.qwen3_vl.vision \
  --model-dir ~/.cache/huggingface/hub/models--Qwen--Qwen3-VL-2B-Instruct/snapshots/<rev> \
  --device 0
```

Without `--weights` it converts the checkpoint through the transparent cache first, so the
usual invocation needs only the checkpoint.

Qwen3-VL taps its encoder partway up — vision layers 5, 11 and 17 on the 2B — and each tap
goes through a merger of its own before reaching the text stack as an addition at one of
its first layers. A plane that is subtly wrong therefore does not fail; it degrades, and
looks like a model that is worse at images than it should be.

The `qwen3_5` counterpart compares one plane, because that family's towers declare no
deepstack indexes and produce none. This one also compares the planes to *each other*:
three planes that were accidentally the same tensor pass every per-plane check, since each
would then be compared against a copy of the right answer.

Measured on Qwen3-VL-2B-Instruct, BF16 tower storage, random patches at three grid sizes:
worst cosine 0.9916 (the merged plane), deepstack planes 0.9985–0.9999.
