// Live, read-only tailer for the surogate JSONL metrics feed.
//
// Uses @logdna/tail-file (handles rotation/truncation via inode tracking) and
// buffers partial trailing lines until a newline arrives. Malformed lines are
// skipped. `fromStart` replays existing content before tailing new appends.

import fs from "node:fs";
import readline from "node:readline";
import TailFile from "@logdna/tail-file";
import { parseLine, type Record_ } from "./records.ts";

export type OnRecords = (records: Record_[]) => void;

export class Feed {
  private buf = "";
  private tail: TailFile | null = null;

  constructor(
    private readonly path: string,
    private readonly fromStart = false,
  ) {}

  exists(): boolean {
    try {
      return fs.statSync(this.path).isFile();
    } catch {
      return false;
    }
  }

  /** Read the entire current file once and return parsed records (snapshot mode). */
  async snapshot(): Promise<Record_[]> {
    if (!this.exists()) return [];
    const out: Record_[] = [];
    const rl = readline.createInterface({ input: fs.createReadStream(this.path, "utf8"), crlfDelay: Infinity });
    for await (const line of rl) {
      const r = parseLine(line);
      if (r) out.push(r);
    }
    return out;
  }

  /** Begin streaming. Calls onRecords with each batch of newly parsed records. */
  async start(onRecords: OnRecords): Promise<void> {
    if (this.fromStart) {
      const initial = await this.snapshot();
      if (initial.length) onRecords(initial);
    }

    const tail = new TailFile(this.path, { encoding: "utf8", pollFileIntervalMs: 250 });
    this.tail = tail;

    tail.on("data", (chunk: string | Buffer) => {
      this.buf += chunk.toString();
      const out: Record_[] = [];
      let nl: number;
      while ((nl = this.buf.indexOf("\n")) !== -1) {
        const line = this.buf.slice(0, nl);
        this.buf = this.buf.slice(nl + 1);
        const r = parseLine(line);
        if (r) out.push(r);
      }
      if (out.length) onRecords(out);
    });

    tail.on("truncated", () => {
      this.buf = "";
    });

    // Don't let a transient read error crash the app; the tailer retries.
    tail.on("tail_error", () => {});
    tail.on("error", () => {});

    // Start the tailer in the background — don't block startup on it. The feed
    // file may not exist yet (the tailer waits for it / errors are swallowed),
    // so the UI shows a waiting state and picks up data once it appears.
    void tail.start().catch(() => {});
  }

  async stop(): Promise<void> {
    if (this.tail) {
      // quit() can hang if the tailer is still waiting for a not-yet-existent
      // file — fire-and-forget so callers never block.
      void this.tail.quit().catch(() => {});
      this.tail = null;
    }
  }
}
