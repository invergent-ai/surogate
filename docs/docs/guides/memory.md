# Memory

This guide collects the main knobs that affect GPU memory use.

## First-line levers

- Reduce `sequence_len`
- Reduce `per_device_train_batch_size`
- Increase `gradient_accumulation_steps` (amortizes optimizer step)

## Framework knobs

- Recomputation (`recompute`)
- Chunking (`lmhead_chunks`, `attn_bwd_chunks`)
- Offloading (`offload_*`)
- Quantization / QLoRA (`qlora_*`)

## See also

- [Offloading](offloading.md)
- [Debugging](debugging.md)
- [Config reference](../reference/config.md)
- [Back to docs index](../index.mdx)
