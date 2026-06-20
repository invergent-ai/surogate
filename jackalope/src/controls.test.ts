import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { readRunPid, pidAlive, runControllable, stopRun, procStartTime } from "./controls.ts";

test("read pid + liveness + controllable", () => {
  const feed = path.join(os.tmpdir(), `ctl-${process.pid}.jsonl`);
  const start = procStartTime(process.pid);
  fs.writeFileSync(`${feed}.pid`, JSON.stringify({ pid: process.pid, start }));
  assert.deepEqual(readRunPid(feed), { pid: process.pid, start });
  assert.equal(pidAlive(process.pid), true);
  assert.equal(runControllable(feed), true); // start-time matches → it's "our" run
  fs.rmSync(`${feed}.pid`, { force: true });
});

test("recycled PID guard: matching pid but wrong start-time → not controllable", () => {
  const feed = path.join(os.tmpdir(), `recycled-${process.pid}.jsonl`);
  fs.writeFileSync(`${feed}.pid`, JSON.stringify({ pid: process.pid, start: "1" })); // bogus start-time
  // On Linux the start-time mismatch rejects it; off Linux there's no /proc so
  // this guard can't apply — only assert the strict behavior where it exists.
  if (process.platform === "linux") assert.equal(runControllable(feed), false);
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
