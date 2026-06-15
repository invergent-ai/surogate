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
