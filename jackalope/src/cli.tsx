#!/usr/bin/env node
import React from "react";
import process from "node:process";
import path from "node:path";
import { render } from "ink";
import supportsTerminalGraphics from "supports-terminal-graphics";
import { discoverFeedPath } from "./launch.ts";
import { setGraphicsMode, type GraphicsMode } from "./render.ts";
import { enableSynchronizedOutput } from "./sync-output.ts";
import { App } from "./ui/App.tsx";

type GraphicsArg = "auto" | GraphicsMode;

interface Args {
  path?: string;
  fromStart: boolean;
  surogateBin: string;
  repoRoot: string;
  graphics: GraphicsArg;
  sync: boolean;
}

function parseArgs(argv: string[]): Args {
  const out: Args = { fromStart: false, surogateBin: "surogate", repoRoot: process.cwd(), graphics: "auto", sync: true };
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i]!;
    if (a === "--from-start") out.fromStart = true;
    else if (a === "--surogate-bin") out.surogateBin = argv[++i] ?? "surogate";
    else if (a === "--repo-root") out.repoRoot = path.resolve(argv[++i] ?? ".");
    else if (a === "--graphics") out.graphics = (argv[++i] as GraphicsArg) ?? "auto";
    else if (a === "--sixel") out.graphics = "sixel";
    else if (a === "--no-sixel" || a === "--braille") out.graphics = "braille";
    else if (a === "--no-sync") out.sync = false;
    else if (a === "-h" || a === "--help") {
      printHelp();
      process.exit(0);
    } else if (!a.startsWith("-")) out.path = a;
  }
  return out;
}

/** Pick the chart renderer. Default (auto) uses the native braille line chart
 *  everywhere, upgrading to inline images only on terminals that truly support
 *  the Kitty/iTerm2 protocols. Sixel and braille can be forced explicitly. */
function configureGraphics(arg: GraphicsArg): void {
  if (arg !== "auto") {
    setGraphicsMode(arg);
    return;
  }
  const g = supportsTerminalGraphics.stdout;
  setGraphicsMode(g.kitty || g.iterm2 ? "image" : "braille");
}

function printHelp() {
  process.stdout.write(
    [
      "jackalope — live training dashboard for surogate (Ink, real graphs)",
      "",
      "Usage: jackalope [path] [options]",
      "",
      "  path                  metrics JSONL file (default: $SUROGATE_METRICS_PATH or /tmp/surogate_metrics.jsonl)",
      "  --from-start          replay the feed from the beginning",
      "  --surogate-bin <b>    surogate executable for the Launch tab (default: surogate)",
      "  --repo-root <dir>     surogate repo root for example train files (default: cwd)",
      "  --graphics <mode>     chart renderer: auto | braille | image | sixel (default: auto)",
      "                        auto = crisp braille lines, inline images on Kitty/iTerm2",
      "  --braille / --sixel   force the braille or sixel renderer",
      "",
      "Keys: q quit · ↑↓ nav · ⏎ select · p pause · x stop run · c compare",
      "",
    ].join("\n"),
  );
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  configureGraphics(args.graphics);
  const feedPath = discoverFeedPath(args.path);

  if (!process.stdout.isTTY) {
    process.stderr.write(
      `jackalope needs an interactive terminal.\nMetrics feed: ${feedPath}\n` +
        "Tail it directly, or use wandb/aim (report_to in your training config).\n",
    );
    process.exit(0);
  }

  // Run in the alternate screen buffer (like vim/htop): a fixed full-screen
  // viewport with no scrollback, so scrolling/resizing can't smear the UI, and
  // the user's terminal is left exactly as it was on exit.
  const tty = process.stdout.isTTY;
  let restored = false;
  const enterAlt = () => {
    if (tty) process.stdout.write("\x1b[?1049h\x1b[H");
  };
  const leaveAlt = () => {
    if (tty && !restored) {
      restored = true;
      process.stdout.write("\x1b[?1049l");
    }
  };
  // ensure we always restore, even on a hard signal
  process.on("exit", leaveAlt);
  enterAlt();

  const restoreSync = args.sync ? enableSynchronizedOutput() : () => {};
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
  try {
    await waitUntilExit();
  } finally {
    restoreSync();
    leaveAlt();
  }
}

main().catch((err) => {
  process.stderr.write(`${err?.stack ?? err}\n`);
  process.exit(1);
});
