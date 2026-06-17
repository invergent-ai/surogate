// Shared by the streaming compute backends (dstack + modal): pipe a child's
// stdout into a run feed — valid metric JSONL lines go to the feed, everything
// (including stderr) to the run log. Closes the fds when the child exits.
import { type ChildProcess } from "node:child_process";
import fs from "node:fs";
import { parseLine } from "./records.ts";

export function pipeMetrics(child: ChildProcess, feedPath: string, logPath: string): void {
  const feedFd = fs.openSync(feedPath, "a");
  const logFd = fs.openSync(logPath, "a");
  let buf = "";
  child.stdout?.on("data", (chunk: Buffer) => {
    try {
      fs.writeSync(logFd, chunk);
    } catch {
      /* ignore */
    }
    buf += chunk.toString("utf8");
    let nl: number;
    while ((nl = buf.indexOf("\n")) >= 0) {
      const line = buf.slice(0, nl);
      buf = buf.slice(nl + 1);
      if (parseLine(line)) {
        try {
          fs.writeSync(feedFd, line + "\n");
        } catch {
          /* ignore */
        }
      }
    }
  });
  child.stderr?.on("data", (c: Buffer) => {
    try {
      fs.writeSync(logFd, c);
    } catch {
      /* ignore */
    }
  });
  child.on("exit", () => {
    try {
      fs.closeSync(feedFd);
      fs.closeSync(logFd);
    } catch {
      /* ignore */
    }
  });
}
