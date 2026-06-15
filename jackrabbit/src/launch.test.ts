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

import { buildGrpoCommand, estimateRunVramGB, fitOnGpu, paramsBFromModel } from "./launch.ts";

test("paramsBFromModel parses sizes incl MoE active", () => {
  assert.equal(paramsBFromModel("Qwen/Qwen3-0.6B"), 0.6);
  assert.equal(paramsBFromModel("meta-llama/Llama-3.1-8B"), 8);
  assert.equal(paramsBFromModel("Qwen/Qwen3-30B-A3B"), 3); // active params
  assert.equal(paramsBFromModel("no-size-here"), null);
});

test("estimateRunVramGB is a positive ballpark", () => {
  const gb = estimateRunVramGB(DEFAULT_FIELDS);
  assert.ok(gb !== null && gb > 0 && gb < 30);
});

test("fitOnGpu verdicts", () => {
  assert.equal(fitOnGpu(6, 30), "fits");
  assert.equal(fitOnGpu(6, 7), "tight");
  assert.equal(fitOnGpu(6, 3), "risk");
  assert.equal(fitOnGpu(null, 30), "unknown");
});

test("buildGrpoCommand has split gpus + 3 configs", () => {
  const c = { train: "t.yaml", infer: "i.yaml", orch: "o.yaml" };
  const cmd = buildGrpoCommand([4, 5], [0, 1, 2, 3], c, "surogate");
  assert.match(cmd, /surogate grpo --train t\.yaml --infer i\.yaml --orch o\.yaml/);
  assert.match(cmd, /--trainer-gpus 4,5 --vllm-gpus 0,1,2,3/);
});

import { buildGrpoCommand as bgc, exampleRulerConfigs } from "./launch.ts";

test("RULER command adds judge-infer + judge-gpus", () => {
  const c = { train: "t.yaml", infer: "i.yaml", orch: "o.yaml", judge: "j.yaml" };
  const cmd = bgc([4, 5], [0, 1], c, "surogate", [6, 7]);
  assert.match(cmd, /--trainer-gpus 4,5 --vllm-gpus 0,1/);
  assert.match(cmd, /--judge-infer j\.yaml --judge-gpus 6,7/);
});

test("GRPO without judge omits judge args", () => {
  const c = { train: "t", infer: "i", orch: "o" };
  assert.doesNotMatch(bgc([1], [2], c, "surogate", []), /--judge/);
});

test("exampleRulerConfigs points at examples/ruler with judge", () => {
  const c = exampleRulerConfigs("/repo");
  assert.match(c.train, /examples\/ruler\/train\.yaml$/);
  assert.ok(c.judge && /examples\/ruler\/judge\.yaml$/.test(c.judge));
});
