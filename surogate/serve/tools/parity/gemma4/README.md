# `gemma4` decoder parity tool

Compares the engine's per-layer probes against `transformers`, which is the arbiter: it is the
implementation the checkpoints were published against.

```bash
# A small model with the 12B's head geometry, ~180M parameters.
python -m tools.parity.gemma4.fixture models/gemma-4-tiny
python -m surogate.serve.convert.gemma4.convert \
    --model models/gemma-4-tiny --out models/gemma-4-tiny.sinfer --device cuda

# One eager prefill, dumping every layer's intermediates.
SUROGATE_SERVE_RAW_PROMPT=1 \
SUROGATE_SERVE_DUMP_RESIDUAL=/tmp/probe SUROGATE_SERVE_DUMP_COLUMNS=13 \
  ./csrc/build-serve/surogate-engine-cli models/gemma-4-tiny.sinfer \
  --prompt "The capital of France is Paris and the capital of Italy is" \
  --max-new 1 --greedy --raw-output --kv-capacity 512 --max-context 512 --no-cuda-graph

python -m tools.parity.gemma4.decoder models/gemma-4-tiny /tmp/probe
```

Every probe should sit above 0.995. On the fixture the worst is ~0.9973, at the last layer's
output, and that is int8 weight quantisation accumulating over six layers rather than a defect.

## Why a fixture rather than the 12B

The published dense sizes are 12 GB and 24 GB of weights; this runs anywhere and exercises the
same machinery, because it keeps the part that matters — the 12B's *head geometry*, 16 query
heads of 256 through the window against 16 of 512 over one key/value head globally, with
`attention_k_eq_v` — and shrinks only the residual width, the depth and the FFN.

`SUROGATE_SERVE_DUMP_COLUMNS` matters: the probe files are named by tag and occurrence, so a
prefill and the decode rounds after it write the same names and the last one wins. Pinning the
column count to the prompt length keeps the prefill's.

## What it caught

The embedding scale. Gemma multiplies its embedding lookup by `sqrt(hidden)`, which is 62.0 at
the 12B's 3840 and 73.5 at the 31B's 5376 — so a target serving both cannot compile it, and the
runtime was applying its compiled constant rather than the artifact's declared one. Every norm
divides that factor straight back out, so the projections and their norms all agreed at cosine
0.9999 and only the residual carried the error: the layer's own output came out at 0.925. It is
exactly the failure this ladder exists to find, and it is invisible from the top.
