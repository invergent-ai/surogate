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

// "waiting": file not here yet (still polling). "streaming": data is flowing.
// "unavailable": the file never appeared within the retry budget — gave up.
export type FeedStatus = "waiting" | "streaming" | "unavailable";
export type OnStatus = (status: FeedStatus) => void;

// ~30s of polling (60 * 500ms) for a launched run's file to appear before we
// declare the feed unavailable rather than spinning forever.
const MAX_RETRIES = 60;

export class Feed {
  private buf = "";
  private tail: TailFile | null = null;
  private stopped = false;
  private retryTimer: ReturnType<typeof setTimeout> | null = null;
  private attempts = 0;
  private status: FeedStatus = "waiting";
  private onStatus: OnStatus = () => {};

  constructor(
    private readonly path: string,
    private readonly fromStart = false,
  ) {}

  /** Notify the listener only on an actual status transition. */
  private emit(status: FeedStatus): void {
    if (status === this.status) return;
    this.status = status;
    this.onStatus(status);
  }

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

  /** Begin streaming. Calls onRecords with each batch of newly parsed records;
   *  onStatus (optional) reports waiting / streaming / unavailable transitions. */
  async start(onRecords: OnRecords, onStatus?: OnStatus): Promise<void> {
    if (onStatus) this.onStatus = onStatus;
    const deliver = (records: Record_[]) => {
      this.emit("streaming");
      onRecords(records);
    };
    // fromStart + file already here → snapshot now and tail only new appends.
    // fromStart + file missing → replay from byte 0 once the run creates it.
    const replayWhenReady = this.fromStart && !this.exists();
    if (this.fromStart && !replayWhenReady) {
      const initial = await this.snapshot();
      // A stop() during the snapshot await means this feed was switched away —
      // don't create a tailer that would never be cleaned up.
      if (this.stopped) return;
      if (initial.length) deliver(initial);
    }
    if (this.stopped) return;
    this.beginTail(deliver, replayWhenReady);
  }

  private beginTail(deliver: OnRecords, fromZero = false): void {
    if (this.stopped) return;
    // On a retry tick the file may still be absent — skip the TailFile setup and
    // just poll again, rather than allocating one only to have start() reject.
    if (!this.exists()) {
      this.scheduleRetry(deliver, fromZero);
      return;
    }
    this.attempts = 0; // file is present — reset the give-up budget
    const tail = new TailFile(this.path, {
      encoding: "utf8",
      pollFileIntervalMs: 250,
      ...(fromZero ? { startPos: 0 } : {}),
    });
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
      if (out.length) deliver(out);
    });

    tail.on("truncated", () => {
      this.buf = "";
    });

    // Don't let a transient read error crash the app; the tailer retries.
    tail.on("tail_error", () => {});
    tail.on("error", () => {});

    // TailFile.start() opens the file eagerly and REJECTS with ENOENT if it
    // doesn't exist yet — and its ENOENT-tolerant polling only engages after a
    // successful first open. A run's metrics file is usually created a few
    // seconds after we switch to it, so on failure we poll for the file to
    // appear and retry, instead of silently giving up forever.
    void tail.start().catch(() => {
      if (this.tail === tail) this.tail = null;
      void tail.quit().catch(() => {});
      this.scheduleRetry(deliver, fromZero);
    });
  }

  private scheduleRetry(deliver: OnRecords, fromZero: boolean): void {
    if (this.stopped || this.retryTimer) return;
    if (++this.attempts > MAX_RETRIES) {
      this.emit("unavailable"); // file never showed up — stop polling
      return;
    }
    this.emit("waiting");
    this.retryTimer = setTimeout(() => {
      this.retryTimer = null;
      this.beginTail(deliver, fromZero);
    }, 500);
  }

  async stop(): Promise<void> {
    this.stopped = true;
    if (this.retryTimer) {
      clearTimeout(this.retryTimer);
      this.retryTimer = null;
    }
    if (this.tail) {
      // quit() can hang if the tailer is still waiting for a not-yet-existent
      // file — fire-and-forget so callers never block.
      const t = this.tail;
      this.tail = null;
      void t.quit().catch(() => {});
    }
  }
}
