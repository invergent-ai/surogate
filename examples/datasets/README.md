# Local datasets and column mapping

[mixed.yaml](mixed.yaml) combines two local training files and a separate validation
file. All paths are relative to the repository root:

```bash
surogate tokenize examples/datasets/mixed.yaml --debug
surogate sft examples/datasets/mixed.yaml
```

- [instructions.jsonl](instructions.jsonl): maps `task`, `context`, and `answer` to
  instruction/input/output and supplies a fixed system prompt.
- [conversations.jsonl](conversations.jsonl): maps the `dialog` column to chat messages.
- [validation.jsonl](validation.jsonl): a separate held-out example, so training data
  is not split again for validation.

The small fixtures show data shapes and token masking. Replace them for real
training. `sample_packing: false` keeps the tiny samples separate for inspection;
most model recipes enable packing. The tokenizer automatically formats instruction
and conversation data and masks non-assistant tokens under the default loss scale.
Raw `type: text` data is shown in [pretraining](../pt/README.md), `type: preference`
in [DPO](../dpo/README.md), and automatic dataset detection in the model recipes.

For chat datasets split into prompt and completion message arrays, use
`messages_field: prompt` and `completion_field: completion`, as in the
[reverse-text recipe](../sft/reverse-text-qwen3.yaml). Conversation datasets may
also set `system_field`, `tools_field`, and `message_property_mappings`; see the
[dataset guide](../../docs/guides/datasets.md) for their expected shapes.
