#!/usr/bin/env node
import React from "react";
import process from "node:process";
import path from "node:path";
import { render } from "ink";
import supportsTerminalGraphics from "supports-terminal-graphics";
import { resolveFeedPath } from "./launch.ts";
import { setSixelPreferred } from "./term-image.ts";
import { App } from "./ui/App.tsx";

interface Args {
  path?: string;
  fromStart: boolean;
  surogateBin: string;
  repoRoot: string;
  sixel: "auto" | "on" | "off";
}

function parseArgs(argv: string[]): Args {
  const out: Args = { fromStart: false, surogateBin: "surogate", repoRoot: process.cwd(), sixel: "auto" };
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i]!;
    if (a === "--from-start") out.fromStart = true;
    else if (a === "--surogate-bin") out.surogateBin = argv[++i] ?? "surogate";
    else if (a === "--repo-root") out.repoRoot = path.resolve(argv[++i] ?? ".");
    else if (a === "--sixel") out.sixel = "on";
    else if (a === "--no-sixel") out.sixel = "off";
    else if (a === "-h" || a === "--help") {
      printHelp();
      process.exit(0);
    } else if (!a.startsWith("-")) out.path = a;
  }
  return out;
}

/** Use Sixel only when the terminal supports it but NOT Kitty/iTerm2 (those are
 *  higher quality and handled by terminal-image). Honor explicit overrides. */
function configureGraphics(mode: "auto" | "on" | "off"): void {
  if (mode === "off") return;
  const g = supportsTerminalGraphics.stdout;
  if (mode === "on") {
    setSixelPreferred(true);
    return;
  }
  if (!g.kitty && !g.iterm2 && g.sixel) setSixelPreferred(true);
}

function printHelp() {
  process.stdout.write(
    [
      "jackrabbit — live training dashboard for surogate (Ink, real graphs)",
      "",
      "Usage: jackrabbit [path] [options]",
      "",
      "  path                metrics JSONL file (default: $SUROGATE_METRICS_PATH or /tmp/surogate_metrics.jsonl)",
      "  --from-start        replay the feed from the beginning",
      "  --surogate-bin <b>  surogate executable for the Launch tab (default: surogate)",
      "  --repo-root <dir>   surogate repo root for example train files (default: cwd)",
      "  --sixel/--no-sixel  force/disable Sixel graphs (default: auto-detect)",
      "",
      "Keys: q quit · ↑↓ nav · ⏎ select · p pause · x stop run · c compare",
      "",
    ].join("\n"),
  );
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  configureGraphics(args.sixel);
  const feedPath = resolveFeedPath(args.path);

  if (!process.stdout.isTTY) {
    process.stderr.write(
      `jackrabbit needs an interactive terminal.\nMetrics feed: ${feedPath}\n` +
        "Tail it directly, or use wandb/aim (report_to in your training config).\n",
    );
    process.exit(0);
  }

  const { waitUntilExit } = render(
    <App
      initialFeedPath={feedPath}
      fromStart={args.fromStart}
      surogateBin={args.surogateBin}
      repoRoot={args.repoRoot}
      version="0.1.0"
    />,
    { exitOnCtrlC: true },
  );
  await waitUntilExit();
}

main().catch((err) => {
  process.stderr.write(`${err?.stack ?? err}\n`);
  process.exit(1);
});
