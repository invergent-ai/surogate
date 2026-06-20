import { test } from "node:test";
import assert from "node:assert/strict";
import { buildCommand, buildConfigYaml, DEFAULT_FIELDS, discoverGpus, resolveFeedPath } from "./launch.ts";

test("buildCommand with/without gpus", () => {
  assert.equal(buildCommand([0, 1, 3], "run.yaml"), "CUDA_VISIBLE_DEVICES=0,1,3 surogate sft run.yaml");
  assert.equal(buildCommand([], "run.yaml"), "surogate sft run.yaml");
});

// DEFAULT_FIELDS now starts with empty model/dataset (no preselection); tests
// that exercise a real config supply them via FIELDS.
const FIELDS = { ...DEFAULT_FIELDS, model: "Qwen/Qwen3-0.6B", datasetPath: "OpenLLM-Ro/ro_gsm8k" };

test("buildConfigYaml includes core fields + feed", () => {
  const y = buildConfigYaml(FIELDS, 2);
  assert.match(y, /model: "Qwen\/Qwen3-0\.6B"/); // quoted so ids/paths with colons/spaces are valid YAML
  assert.match(y, /output_dir: "/);
  assert.match(y, /recipe: fp8-hybrid/);
  assert.match(y, /gpus: 2/);
  assert.match(y, /lora: true/);
  assert.match(y, /lora_rank: 16/);
  assert.match(y, /report_to: \[surogate\]/);
  assert.match(y, /q_proj/);
});

test("buildConfigYaml writes the surogate schema field names (guards typos)", () => {
  const y = buildConfigYaml(FIELDS, 1);
  // exact keys from surogate's SFTConfig — a typo here silently misconfigures runs
  assert.match(y, /optimizer: adamw_8bit/);
  assert.match(y, /lr_scheduler_type: linear/);
  assert.match(y, /weight_decay: 0/);
  assert.match(y, /max_grad_norm: 1\.0/);
  assert.match(y, /recompute: true/); // surogate names grad-checkpointing "recompute"
  assert.match(y, /save_steps: 50/);
  assert.match(y, /save_total_limit: 5/);
  assert.match(y, /resume_from_checkpoint: false/);
  assert.doesNotMatch(y, /merge_adapter/); // only written when enabled
  assert.match(buildConfigYaml({ ...FIELDS, mergeAdapter: true }, 1), /merge_adapter: true/);
});

test("buildConfigYaml omits lora block when disabled", () => {
  const y = buildConfigYaml({ ...FIELDS, lora: false }, 1);
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

test("discoverGpus returns parsed GPUs, or [] when there's no NVIDIA driver", () => {
  const g = discoverGpus();
  assert.ok(Array.isArray(g));
  if (g.length) {
    assert.equal(typeof g[0]!.id, "number");
    assert.ok("sm" in g[0]!);
  }
});

import { buildGrpoCommand, estimateRunVramGB, fitOnGpu, paramsBFromModel } from "./launch.ts";

test("paramsBFromModel parses sizes incl MoE active", () => {
  assert.equal(paramsBFromModel("Qwen/Qwen3-0.6B"), 0.6);
  assert.equal(paramsBFromModel("meta-llama/Llama-3.1-8B"), 8);
  assert.equal(paramsBFromModel("Qwen/Qwen3-30B-A3B"), 3); // active params
  assert.equal(paramsBFromModel("no-size-here"), null);
});

test("estimateRunVramGB is a positive ballpark", () => {
  const gb = estimateRunVramGB(FIELDS);
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

import { buildGrpoCommand as bgc, ensureRlConfigs } from "./launch.ts";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";

// Write to a throwaway dir so tests never touch the user's real ~/.surogate-watch.
const TMP_CFG = fs.mkdtempSync(path.join(os.tmpdir(), "jk-rl-"));

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

test("ensureRlConfigs(grpo) generates runnable configs with reconciled ports", () => {
  const c = ensureRlConfigs("grpo", "/nonexistent-repo", TMP_CFG);
  assert.ok(!c.judge); // GRPO has no judge
  const infer = fs.readFileSync(c.infer, "utf8");
  const orch = fs.readFileSync(c.orch, "utf8");
  assert.match(infer, /port: 8007/);
  assert.match(orch, /localhost:8007\/v1/); // orch client points at the student vLLM port
  assert.match(orch, /id: markdown-table-qa/); // a locally-importable env (no "/" → no hub install)
});

test("ensureRlConfigs(ruler) reconciles disjoint student/judge ports", () => {
  const c = ensureRlConfigs("ruler", "/nonexistent-repo", TMP_CFG);
  assert.ok(c.judge);
  const judge = fs.readFileSync(c.judge!, "utf8");
  const orch = fs.readFileSync(c.orch, "utf8");
  assert.match(judge, /port: 8001/); // judge vLLM binds 8001…
  assert.match(orch, /localhost:8001\/v1/); // …and the RULER judge base_url matches
  assert.match(orch, /localhost:8007\/v1/); // student rollout still on 8007 (disjoint)
});
