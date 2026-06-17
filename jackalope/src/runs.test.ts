import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { listRuns, newRunFeedPath, isActive } from "./runs.ts";

test("newRunFeedPath puts the feed in a slugged per-run folder", () => {
  const a = newRunFeedPath("sft-fp8-hybrid", 1700000000000);
  assert.match(a, /\.surogate-watch\/runs\/\d{8}-\d{6}-sft-fp8-hybrid\/metrics\.jsonl$/);
});

test("listRuns includes extra paths, newest first, with active flag", () => {
  const tmp = path.join(os.tmpdir(), `runs-${process.pid}.jsonl`);
  fs.writeFileSync(tmp, '{"step":1}\n');
  const now = Date.now();
  const runs = listRuns([tmp], now);
  assert.ok(runs.some((r) => r.path === path.resolve(tmp)));
  const r = runs.find((x) => x.path === path.resolve(tmp))!;
  assert.equal(isActive(r), true); // just written
  fs.rmSync(tmp, { force: true });
});
