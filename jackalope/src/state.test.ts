import { test } from "node:test";
import assert from "node:assert/strict";
import { WatchState } from "./state.ts";
import { parseLine } from "./records.ts";

function ingestLines(s: WatchState, ...lines: string[]) {
  s.ingest(lines.map(parseLine).filter((r): r is NonNullable<typeof r> => r !== null));
}

test("config sets header; positive max_steps only", () => {
  const s = new WatchState();
  ingestLines(s, '{"type":"config","ts":0,"recipe":"fp8-hybrid","model":"Qwen/Qwen3-0.6B","max_steps":200,"lora":true}');
  assert.equal(s.recipe, "fp8-hybrid");
  assert.equal(s.model, "Qwen/Qwen3-0.6B");
  assert.equal(s.maxSteps, 200);
  assert.equal(s.lora, true);
});

test("steps update history + eval history + phase", () => {
  const s = new WatchState();
  ingestLines(
    s,
    '{"step":1,"ts":1,"train/loss":4.0,"train/lr":1e-4,"train/norm":1.0,"train/epoch":0.1,"train/phase":"warmup"}',
    '{"step":2,"ts":2,"train/loss":3.0}',
    '{"step":2,"ts":2,"eval/loss":3.2}',
  );
  assert.equal(s.step, 2);
  assert.equal(s.latestTrainLoss, 3.0);
  assert.deepEqual(s.lossHistory, [4.0, 3.0]);
  assert.deepEqual(s.lossSteps, [1, 2]);
  assert.equal(s.gradNorm, 1.0);
  assert.equal(s.phase, "warmup");
  assert.deepEqual(s.evalHistory, [[2, 3.2]]);
});

test("eta from median step duration", () => {
  const s = new WatchState();
  ingestLines(s, '{"type":"config","ts":0,"max_steps":5}');
  ingestLines(s, '{"step":1,"ts":10,"train/loss":1}', '{"step":2,"ts":20,"train/loss":1}', '{"step":3,"ts":30,"train/loss":1}');
  assert.equal(s.etaSeconds(), 20); // ~10s/step * 2 remaining
});

test("gpus keyed by id, latest wins", () => {
  const s = new WatchState();
  ingestLines(
    s,
    '{"type":"gpu","gpu_id":0,"step":1,"ts":1,"temperature":60,"gpu_util":50,"mem_util":40,"power":300000}',
    '{"type":"gpu","gpu_id":1,"step":1,"ts":1,"temperature":62}',
    '{"type":"gpu","gpu_id":0,"step":2,"ts":2,"temperature":61}',
  );
  const g = s.gpusSorted();
  assert.deepEqual(g.map((x) => x.gpuId), [0, 1]);
  assert.equal(g[0]!.temp, 61); // latest for gpu0
  assert.equal(s.hasGpus, true);
  // telemetry history accumulates per gpu
  assert.deepEqual(s.gpuHistory(0)?.temp, [60, 61]);
});

test("derived: eval gap, smoothed loss, best loss, tps history", () => {
  const s = new WatchState();
  ingestLines(
    s,
    '{"step":1,"ts":1,"train/loss":2.0,"train/tokens_per_second":1000}',
    '{"step":2,"ts":2,"train/loss":1.0,"eval/loss":1.5,"train/tokens_per_second":1200}',
    '{"step":3,"ts":3,"train/loss":1.2}',
  );
  // eval 1.5 vs train 1.2 → +25%
  assert.equal(Math.round(s.evalGapPct()!), 25);
  const best = s.bestLoss()!;
  assert.equal(best.loss, 1.0);
  assert.equal(best.step, 2);
  assert.deepEqual(s.tpsHistory, [1000, 1200]);
  const sm = s.smoothedLoss(0.5);
  assert.equal(sm.length, 3);
  assert.equal(sm[0], 2.0); // first equals raw
  assert.ok(sm[1]! > 1.0 && sm[1]! < 2.0); // EMA between
});
