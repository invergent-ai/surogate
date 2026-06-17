import { test } from "node:test";
import assert from "node:assert/strict";
import { parseLine } from "./records.ts";

test("parses a config line", () => {
  const r = parseLine('{"type":"config","ts":1,"recipe":"fp8-hybrid","output_dir":"./out"}');
  assert.equal(r?.kind, "config");
  if (r?.kind === "config") {
    assert.equal(r.recipe, "fp8-hybrid");
    assert.equal(r.fields["output_dir"], "./out");
  }
});

test("parses a real step line (train/norm, train/epoch)", () => {
  const r = parseLine(
    '{"step":120,"ts":10,"train/loss":1.14,"train/lr":1.8e-4,"train/norm":0.7,"train/epoch":2.1,"train/tokens_per_second":18000,"train/phase":"converging"}',
  );
  assert.equal(r?.kind, "step");
  if (r?.kind === "step") {
    assert.equal(r.step, 120);
    assert.equal(r.trainLoss, 1.14);
    assert.equal(r.gradNorm, 0.7); // from train/norm
    assert.equal(r.epoch, 2.1); // from train/epoch
    assert.equal(r.tokensPerSecond, 18000);
    assert.equal(r.phase, "converging");
  }
});

test("parses a real gpu line (power mW->W, util percent, mem fraction)", () => {
  const r = parseLine('{"type":"gpu","gpu_id":0,"step":120,"ts":10,"temperature":55,"power":325152,"gpu_util":98,"mem_util":31}');
  assert.equal(r?.kind, "gpu");
  if (r?.kind === "gpu") {
    assert.equal(r.temp, 55);
    assert.equal(r.power, 325.152); // mW -> W
    assert.equal(r.smUtil, 98); // percent
    assert.ok(Math.abs((r.memUtil ?? 0) - 0.31) < 1e-9); // 31% -> 0.31
  }
});

test("malformed / blank / unknown -> null", () => {
  assert.equal(parseLine(""), null);
  assert.equal(parseLine("{not json"), null);
  assert.equal(parseLine("[1,2,3]"), null);
  assert.equal(parseLine('{"type":"allocator","step":1}'), null);
});
