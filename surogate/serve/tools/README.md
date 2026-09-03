# SInfer maintainer tools

`tools/` contains the project-owner workflows that stand beside the engine: benchmarks,
evaluation, parity checks, probes and smoke runs, independent
Python references, numerical parity diagnostics, benchmark orchestration, and serving smoke checks.
These tools are not part of the public download-and-run path; normal users should start with the
[project README](../README.md).

Run commands from the repository root with a Python 3.11 environment containing the dependencies
for the selected tool.

## Task index

| Task | Location |
|---|---|
| Build the 27B artifact | [`../convert/qwen3_6_27b/`](../convert/qwen3_6_27b/) |
| Build the Qwen3.8-27B artifact | [`../convert/qwen3_8_27b/`](../convert/qwen3_8_27b/) |
| Build the 35B-A3B artifact | [`../convert/qwen3_6_35b_a3b/`](../convert/qwen3_6_35b_a3b/) |
| Inspect artifact metadata and objects | [`../artifact/inspect.py`](../artifact/inspect.py) |
| Measure perplexity the way llama-perplexity does | [`eval/perplexity.py`](eval/perplexity.py) |
| Run the 27B Python reference | [`reference/qwen3_6_27b/`](reference/qwen3_6_27b/README.md) |
| Run the 35B-A3B Python reference | [`reference/qwen3_6_35b_a3b/`](reference/qwen3_6_35b_a3b/README.md) |
| Compare 27B artifact/source Vision activations | [`parity/qwen3_6_27b/`](parity/qwen3_6_27b/README.md) |
| Run benchmark matrices | [`bench/`](bench/README.md) |
| Exercise a resident HTTP server | [`smoke/serve_contract.py`](smoke/serve_contract.py) |
| Exercise thinking preservation through a managed server | [`smoke/serve_thinking_preservation.py`](smoke/serve_thinking_preservation.py) |

## Artifact workflow

The converters consume an official local BF16 checkpoint and write one complete `.sinfer`
artifact. The paths below are placeholders for the maintainer's local checkpoint checkouts:

```bash
python -m surogate.serve.convert.qwen3_6_27b.convert \
  --model /path/to/Qwen3.6-27B \
  --out out/qwen3_6_27b.sinfer

python -m surogate.serve.convert.qwen3_8_27b.convert \
  --model /path/to/Qwen3.8-27B \
  --out out/qwen3_8_27b.sinfer

python -m surogate.serve.convert.qwen3_6_35b_a3b.convert \
  --model /path/to/Qwen3.6-35B-A3B-base \
  --dflash-model /path/to/Qwen3.6-35B-A3B-DFlash \
  --out out/qwen3_6_35b_a3b.sinfer
```

Inspect either result:

```bash
python -m surogate.serve.artifact.inspect out/qwen3_6_27b.sinfer --objects
```

The inventories, formats and conversion recipes live beside the converters themselves, in
[`../convert/`](../convert/). Nobody downloads an artifact: `surogate serve model.gguf` reads
the GGUF where it lies and builds whatever index it needs in a local cache.

## Python references and parity

```bash
python3 -m tools.reference.qwen3_6_27b \
  --weights out/qwen3_6_27b.sinfer \
  --prompt "请简短介绍一下你自己。" --decode 128

python3 -m tools.reference.qwen3_6_35b_a3b \
  --weights out/qwen3_6_35b_a3b.sinfer \
  --prompt "请简短介绍一下你自己。" --decode 128
```

The Python implementations are independent diagnostic references, not alternate public inference
products or generated-token goldens for the C++ engine. See the parity README for the direct 27B
artifact/source Vision comparison command.

## Benchmark orchestration

`tools/bench/run_sinfer_bench_matrix.py` builds and runs the public-Engine benchmark matrix and
writes ignored local reports below `profiles/bench/`:

```bash
python3 tools/bench/run_sinfer_bench_matrix.py --preset core --dry-run
python3 tools/bench/run_sinfer_bench_matrix.py --preset core
```

See [`tools/bench/README.md`](bench/README.md) and [`bench/README.md`](../bench/README.md) for the
orchestrator and executable contracts.

## Serving smoke

After starting `sinfer-serve` in another terminal:

```bash
python3 -m tools.smoke.serve_contract \
  --base-url http://127.0.0.1:18080 \
  --model qwen3.6-27b
```

The client exercises OpenAI, Anthropic, streaming, usage, multimodal, and tool-call response
surfaces against the resident process.

For typed rewrite-checkpoint and thinking-history behavior, the managed smoke script launches a
real server and consumes the repository fixture:

```bash
python3 tools/smoke/serve_thinking_preservation.py \
  --artifact out/qwen3_6_27b.sinfer --backend mtp
```
