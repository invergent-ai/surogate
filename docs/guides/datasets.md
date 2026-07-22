# Datasets

Surogate supports multiple dataset formats configured under `datasets` (and optionally `validation_datasets`).

Dataset `type` can be:
- `text` (pretraining / raw text)
- `instruction` (instruction/output)
- `conversation` (chat messages)
- `preference` (`{prompt, chosen, rejected}` pairs for offline DPO)
- `auto` (auto-detect)

Every type loads through the same pipeline: `path` may be a local JSONL/JSON/parquet/CSV
file, a dataset directory, or a HuggingFace hub repo, and `subset`/`split`/`samples`
apply uniformly. Column names that differ from the defaults are mapped with per-type
field options (e.g. `text_field`, `messages_field`, `chosen_field`).

See the full schema and examples in the config reference.

## See also

- [Config reference: datasets](../reference/config.md)
- [Quickstart: SFT](../getting-started/quickstart-sft.md)
- [Quickstart: DPO](../getting-started/quickstart-dpo.md)
- [Back to docs index](../index.mdx)
