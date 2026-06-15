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
});
