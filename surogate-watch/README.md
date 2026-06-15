# @surogate/watch

A live terminal dashboard for [surogate](https://surogate.ai) training runs —
built with **Ink** (React for CLIs) and **real raster graphs** (not ASCII/braille).

It is a standalone Node package: it does not import surogate, it just **tails the
JSONL metrics feed** surogate writes (`report_to: [surogate]`). Run it in a second
terminal — or over SSH — alongside training.

The loss chart is a real image (ECharts → resvg → PNG) shown inline via the
**Kitty graphics protocol** (or iTerm2 / Sixel), with a colored half-block
fallback on terminals without a graphics protocol.

## Requirements

- Node ≥ 20
- A terminal with a graphics protocol for real graphs: **Kitty, Ghostty, WezTerm,
  iTerm2** (others fall back to colored half-blocks). Over SSH, the *local*
  terminal must support a protocol (and enable tmux passthrough if used).

## Install & build

```bash
cd surogate-watch
npm install
npm run build      # -> dist/cli.js
```

## Use

Enable the feed in your surogate training config:

```yaml
report_to: [surogate]
log_gpu_util: 5
logging_steps: 1
```

Then, in another terminal:

```bash
node dist/cli.js                 # live tail of /tmp/surogate_metrics.jsonl
node dist/cli.js <path>          # a specific feed file
node dist/cli.js --from-start    # replay a finished run from the top
node dist/cli.js --repo-root /path/to/surogate   # for example train files in Launch
```

(After `npm link` or publish: `surogate-watch` instead of `node dist/cli.js`.)

Keys: **q** quit · **m** monitor · **l** launch · **p** pause.

## Tabs

- **Monitor** — live loss chart (train area + eval diamonds), vitals (loss/eval/lr/
  grad-norm/throughput/phase), per-GPU device table (temp/power/util/memory), log tail.
- **Launch** — pick GPUs, choose a precision recipe, review the resolved
  `CUDA_VISIBLE_DEVICES=… surogate sft …` command, and start a real run (it writes a
  `watch-run.yaml`, spawns surogate, and switches to Monitor).

## Dev

```bash
npm run dev        # run from source (tsx)
npm run typecheck
npm test           # node:test + ink-testing-library
```
