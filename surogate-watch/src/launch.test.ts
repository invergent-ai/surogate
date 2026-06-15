import { test } from "node:test";
import assert from "node:assert/strict";
import { buildCommand, buildConfigYaml, DEFAULT_FIELDS, discoverGpus, resolveFeedPath } from "./launch.ts";

test("buildCommand with/without gpus", () => {
  assert.equal(buildCommand([0, 1, 3], "run.yaml"), "CUDA_VISIBLE_DEVICES=0,1,3 surogate sft run.yaml");
  assert.equal(buildCommand([], "run.yaml"), "surogate sft run.yaml");
});

test("buildConfigYaml includes core fields + feed", () => {
  const y = buildConfigYaml(DEFAULT_FIELDS, 2);
  assert.match(y, /model: Qwen\/Qwen3-0\.6B/);
  assert.match(y, /recipe: fp8-hybrid/);
  assert.match(y, /gpus: 2/);
  assert.match(y, /lora: true/);
  assert.match(y, /lora_rank: 16/);
  assert.match(y, /report_to: \[surogate\]/);
  assert.match(y, /q_proj/);
});

test("buildConfigYaml omits lora block when disabled", () => {
  const y = buildConfigYaml({ ...DEFAULT_FIELDS, lora: false }, 1);
  assert.match(y, /lora: false/);
  assert.doesNotMatch(y, /lora_rank/);
});

test("resolveFeedPath precedence", () => {
  assert.equal(resolveFeedPath("/x.jsonl"), "/x.jsonl");
  const prev = process.env["SUROGATE_METRICS_PATH"];
  process.env["SUROGATE_METRICS_PATH"] = "/env.jsonl";
  assert.equal(resolveFeedPath(), "/env.jsonl");
  delete process.env["SUROGATE_METRICS_PATH"];
  assert.equal(resolveFeedPath(), "/tmp/surogate_metrics.jsonl");
  if (prev !== undefined) process.env["SUROGATE_METRICS_PATH"] = prev;
});

test("discoverGpus never empty", () => {
  const g = discoverGpus();
  assert.ok(g.length >= 1);
  assert.equal(typeof g[0]!.id, "number");
});
