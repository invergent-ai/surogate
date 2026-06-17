# jackalope

A live terminal dashboard for [surogate](https://surogate.ai) — watch training in
real time, browse and pick models/datasets from HuggingFace, and **launch runs**
(local, SSH, or cloud) without leaving the terminal. Built with **Ink** (React for
CLIs) and real graphs.

jackalope is a standalone, Node-free TUI. It doesn't import surogate — it **tails
the JSONL metrics feed** surogate writes (`report_to: [surogate]`) and shells out to
the `surogate` CLI to start runs. Run it in a second terminal, or on your laptop to
drive remote/cloud training over SSH — no GPU required on the machine showing the UI.

```
┌ ◆ jackalope ─────────────────────────────────────────────────────────────┐
│  ▸ Monitor │  TRAINING LOSS                                  ● live        │
│    GPUs    │   1.8 ┤╮                                                       │
│    Models  │       ╰╮___                                                    │
│    Datasets│   0.9 ┤   ╰─╮____                                              │
│    Launch  │       ┤        ╰──╮________                                    │
│    Runs    │   0.2 ┼──────────────────────────  step 420/2000  21%         │
│            │   gpu0 ███████░ 92%   gpu1 ██████░░ 81%                        │
└────────────┴────────────────────────────────────────────────────────────┘
```

## Install

**Via surogate (recommended).** surogate bundles the Linux-x64 binary and fetches
updates on its own release channel:

```bash
surogate jackalope                 # launch the dashboard
surogate jackalope --update        # pull the latest build
```

**Standalone binary.** Grab the binary for your OS from the `jackalope-latest`
GitHub release (Linux x64/arm64, macOS arm64/x64, Windows x64) — no Node needed.
Handy for running the dashboard on a laptop to drive remote training over SSH.

**From source.**

```bash
cd jackalope
npm install
npm run build          # -> dist/cli.js   (or: npm run dev  to run via tsx)
```

## Quick start

Enable the feed in your surogate training config:

```yaml
report_to: [surogate]
logging_steps: 1
log_gpu_util: 5
```

Then watch (and launch) from another terminal:

```bash
jackalope                                  # live tail of /tmp/surogate_metrics.jsonl
jackalope <path>                           # a specific feed file
jackalope --from-start                     # replay a finished run from the top
jackalope --repo-root /path/to/surogate    # so Launch can find example configs + envs
jackalope --surogate-bin /path/to/surogate # if surogate isn't on PATH
```

## What it does

- **Monitor** — live loss chart (braille by default; train line + eval markers),
  step/epoch progress, lr · grad-norm · throughput · phase, and per-GPU
  temp/power/util/memory meters. Real errors and run logs stream inline.
- **Models / Datasets** — search HuggingFace or type a local path (`/…`, `~/…`,
  `./…`). Each model shows a **"trainable by surogate"** badge resolved from its
  architecture; pick one to line it up for a launch.
- **Launch** — choose **SFT, GRPO, or RULER**, pick a compute target, select GPUs
  and a precision recipe, edit parameters, and start a real run. jackalope writes
  the config, spawns surogate, and switches to Monitor.
- **GPUs** — browse local devices and saved compute providers.
- **Runs / Logs / Files** — past runs, live logs, and run artifacts (fetch cloud
  outputs back to disk).
- **Setup** — a guided first-run flow to pick compute and install surogate.

### Compute targets

| Target | What runs |
|--------|-----------|
| **Local** | your own GPUs |
| **SSH**   | a remote box with surogate installed |
| **Modal** | a serverless GPU sandbox (streams metrics back; stop terminates it) |
| **dstack**| a cloud backend you've configured (`dstack apply`) |

Cloud/SSH credentials are never stored by jackalope — they go to each tool's own
native store (Modal → `~/.modal.toml`, dstack → `~/.dstack/...`, SSH → `~/.ssh`).

### Training modes

- **SFT** — supervised fine-tuning.
- **GRPO** — RL with a vLLM rollout server and the trainer on disjoint GPUs.
- **RULER** — GRPO with an additional frozen LLM **judge** (3-way GPU split).

GRPO/RULER need the vLLM stack in surogate's venv; jackalope detects a missing
stack and offers a one-key install before launching.

### HuggingFace token

Authenticated Hub requests avoid anonymous rate limits. jackalope uses your
`huggingface-cli` login automatically (the `~/.cache/huggingface/token` file). No
token yet? Press **`^T`** in the Models/Datasets browser to paste one — it's
validated and saved to the standard location.

## Graphs

The loss chart is a crisp native **braille line chart** that works in any terminal,
including over SSH/tmux. On terminals with an inline-image protocol it can render a
real raster chart (ECharts → resvg → PNG via **Kitty/iTerm2**, or **Sixel**):

```bash
jackalope --graphics image     # or: sixel | auto | braille
```

Best image support: **Kitty, Ghostty, WezTerm, iTerm2**. Others fall back to
colored half-blocks. Over SSH the *local* terminal must support a protocol (and
tmux passthrough if used).

## Keys

**q** quit · **↑↓** navigate · **⏎** select · **p** pause · **x** stop a run ·
**t** theme. Context-specific keys are shown in the footer of each tab.

## Requirements

- To **show** the dashboard: a terminal (Node ≥ 20 only if running from source;
  the binary needs nothing). For real graphs, an image-capable terminal.
- To **launch** training: `surogate` reachable (locally, over SSH, or in the cloud).

## Dev

```bash
npm run dev          # run from source (tsx)
npm run typecheck    # tsc --noEmit
npm test             # node:test + ink-testing-library
npm run build:binary # standalone binary via Bun --compile (see scripts/)
```
