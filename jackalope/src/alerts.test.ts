import { test } from "node:test";
import assert from "node:assert/strict";
import { WatchState } from "./state.ts";
import { parseLine } from "./records.ts";
import { AlertEngine } from "./alerts.ts";

function st(lines: string[]) {
  const s = new WatchState();
  s.ingest(lines.map(parseLine).filter((r): r is NonNullable<typeof r> => r !== null));
  return s;
}

test("alert: complete fires once at max_steps", () => {
  const s = st(['{"type":"config","ts":0,"max_steps":3}', '{"step":3,"ts":3,"train/loss":1}']);
  const e = new AlertEngine();
  assert.equal(e.check(s, "r", 1000)?.kind, "complete");
  assert.equal(e.check(s, "r", 1000), null); // deduped
});

test("alert: diverging loss", () => {
  const lines = ['{"type":"config","ts":0,"max_steps":100}'];
  for (let i = 1; i <= 12; i++) lines.push(`{"step":${i},"ts":${i},"train/loss":${1.0 + i * 0.2}}`);
  const e = new AlertEngine();
  assert.equal(e.check(st(lines), "r", 1000)?.kind, "diverge");
});

test("alert: stalled when feed goes quiet", () => {
  const s = st(['{"type":"config","ts":0,"max_steps":100}', '{"step":5,"ts":10,"train/loss":1}']);
  const e = new AlertEngine();
  // now = 10s feed ts + 40s later
  assert.equal(e.check(s, "r", 50000)?.kind, "stalled");
});
