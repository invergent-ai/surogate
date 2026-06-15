#!/usr/bin/env node
import React from "react";
import process from "node:process";
import path from "node:path";
import { render } from "ink";
import { Feed } from "./feed.ts";
import { resolveFeedPath } from "./launch.ts";
import { App } from "./ui/App.tsx";

interface Args {
  path?: string;
  fromStart: boolean;
  surogateBin: string;
  repoRoot: string;
}

function parseArgs(argv: string[]): Args {
  const out: Args = { fromStart: false, surogateBin: "surogate", repoRoot: process.cwd() };
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i]!;
    if (a === "--from-start") out.fromStart = true;
    else if (a === "--surogate-bin") out.surogateBin = argv[++i] ?? "surogate";
    else if (a === "--repo-root") out.repoRoot = path.resolve(argv[++i] ?? ".");
    else if (a === "-h" || a === "--help") {
      printHelp();
      process.exit(0);
    } else if (!a.startsWith("-")) out.path = a;
  }
  return out;
}

function printHelp() {
  process.stdout.write(
    [
      "surogate-watch — live training dashboard (Ink, real graphs)",
      "",
      "Usage: surogate-watch [path] [options]",
      "",
      "  path                metrics JSONL file (default: $SUROGATE_METRICS_PATH or /tmp/surogate_metrics.jsonl)",
      "  --from-start        replay the feed from the beginning",
      "  --surogate-bin <b>  surogate executable for the Launch tab (default: surogate)",
      "  --repo-root <dir>   surogate repo root for example train files (default: cwd)",
      "",
      "Keys: q quit · m monitor · l launch · p pause",
      "",
    ].join("\n"),
  );
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  const feedPath = resolveFeedPath(args.path);

  if (!process.stdout.isTTY) {
    process.stderr.write(
      `surogate-watch needs an interactive terminal.\nMetrics feed: ${feedPath}\n` +
        "Tail it directly, or use wandb/aim (report_to in your training config).\n",
    );
    process.exit(0);
  }

  const feed = new Feed(feedPath, args.fromStart);
  const { waitUntilExit } = render(
    <App feed={feed} feedPath={feedPath} surogateBin={args.surogateBin} repoRoot={args.repoRoot} version="0.1.0" />,
    { exitOnCtrlC: true },
  );
  await waitUntilExit();
}

main().catch((err) => {
  process.stderr.write(`${err?.stack ?? err}\n`);
  process.exit(1);
});
