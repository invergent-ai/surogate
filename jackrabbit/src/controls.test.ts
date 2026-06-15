import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { readRunPid, pidAlive, runControllable, stopRun } from "./controls.ts";

test("read pid + liveness + controllable", () => {
  const feed = path.join(os.tmpdir(), `ctl-${process.pid}.jsonl`);
  fs.writeFileSync(`${feed}.pid`, String(process.pid));
  assert.equal(readRunPid(feed), process.pid);
  assert.equal(pidAlive(process.pid), true);
  assert.equal(runControllable(feed), true);
  fs.rmSync(`${feed}.pid`, { force: true });
});

test("missing pid file → null / not controllable", () => {
  const feed = path.join(os.tmpdir(), `nope-${process.pid}.jsonl`);
  assert.equal(readRunPid(feed), null);
  assert.equal(runControllable(feed), false);
});

test("stopRun on a dead pid returns false (safe)", () => {
  const feed = path.join(os.tmpdir(), `dead-${process.pid}.jsonl`);
  fs.writeFileSync(`${feed}.pid`, "2147480000"); // not a live pid
  assert.equal(stopRun(feed), false);
  fs.rmSync(`${feed}.pid`, { force: true });
});
