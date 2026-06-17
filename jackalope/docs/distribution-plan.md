# Distribution plan (TODO)

How surogate users actually get & run jackalope. Today it's build-from-source
(`npm install && npm run build && npm link`) — fine for devs, but surogate users
are Python/ML people, often with **no Node**, often inside **Docker or over SSH**.
The goal: **"however you got surogate, `jackalope` (or `surogate watch`) just works."**

## Surogate's 3 install modes (from surogate/README.md)
- **A — Docker** (recommended): 3 CUDA images, `docker run … <IMAGE> sft config.yaml`.
- **B — install.sh**: `curl -LsSf …/install.sh | bash` → installs a wheel into `.venv/`.
- **C — source**: `uv pip install -e .`.

## How users run jackalope (UX is already done)
- **On the GPU box**: `surogate sft config.yaml` (terminal 1) + `jackalope` (terminal 2 / tmux).
  Their config needs `report_to: [surogate]` (+ `logging_steps: 1`).
- **From a laptop** (no local GPU): `jackalope` locally → Launch → Compute → SSH / Cloud (dstack);
  training runs remotely, the metrics feed is mirrored back. (This is what the SSH/dstack work enables.)

## Distribution work to do (ranked by value)
1. **`surogate watch [config|feed]` subcommand** — tightest integration. surogate's CLI already
   has clean subparsers (`surogate/cli/main.py`); add a `watch` parser that just execs the bundled
   jackalope binary. Then: `surogate sft config.yaml` to train, `surogate watch` to watch. (Change is
   in the surogate repo, not jackalope.)
2. **Standalone, Node-less binary + `curl …/install.sh | sh`** — for Docker/laptops/no-Node users.
   - Per-platform via **Bun compile** (or Node SEA): linux-x64, mac-arm64/x64, win-x64.
   - **Wrinkle:** the default **braille chart is pure JS (no native deps)**; only the optional
     `--graphics image|sixel` raster path uses `@resvg/resvg-js` (native). Make resvg a **lazy
     import** (load only when image/sixel mode is selected) so the core binary ships with zero native
     deps. Ship binaries as GitHub release assets.
3. **`npm publish`** — instant `npx jackalope` / `npm i -g jackalope`. Fastest to ship; only helps
   users who already have Node ≥ 20.
4. **Bundle into surogate Docker images + install.sh** — bake the binary into the 3 images and have
   install.sh drop it into `.venv/bin` (or `~/.local/bin`), so it's present wherever surogate is.

## Suggested first cut
`surogate watch` (#1) + standalone binary/install.sh (#2) cover everyone; npm publish (#3) is a cheap
extra for the Node crowd. Docker bundling (#4) follows once the binary exists.
