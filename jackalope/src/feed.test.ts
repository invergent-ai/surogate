import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { Feed } from "./feed.ts";

test("snapshot reads existing records; missing file → []", async () => {
  const p = path.join(os.tmpdir(), `feed-${process.pid}.jsonl`);
  fs.writeFileSync(p, '{"type":"config","ts":0,"recipe":"bf16"}\n{"step":1,"ts":1,"train/loss":2}\n');
  const recs = await new Feed(p, true).snapshot();
  assert.equal(recs.length, 2);
  fs.rmSync(p, { force: true });
  assert.deepEqual(await new Feed(p, true).snapshot(), []);
});

test("start() on a missing feed does not throw", async () => {
  const feed = new Feed("/tmp/jb-definitely-missing.jsonl", true);
  let records = 0;
  await feed.start(() => (records += 1));
  await feed.stop();
  assert.equal(records, 0);
});

test("start() tails a feed file that only appears after start (retries ENOENT)", async () => {
  const p = path.join(os.tmpdir(), `feed-late-${process.pid}.jsonl`);
  fs.rmSync(p, { force: true });
  const feed = new Feed(p, true);
  const got: number[] = [];
  await feed.start((recs) => recs.forEach((r) => r.kind === "step" && got.push(r.step)));
  // file appears only now — the tailer must notice and pick it up
  fs.writeFileSync(p, '{"step":1,"ts":1,"train/loss":2}\n');
  await new Promise((r) => setTimeout(r, 1500));
  await feed.stop();
  fs.rmSync(p, { force: true });
  assert.ok(got.includes(1), `expected step 1 to be tailed, got ${JSON.stringify(got)}`);
});
