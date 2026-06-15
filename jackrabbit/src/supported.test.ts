import { test } from "node:test";
import assert from "node:assert/strict";
import { checkTrainable } from "./supported.ts";

test("supported model by architecture", () => {
  const t = checkTrainable(["Qwen3ForCausalLM"], "qwen3");
  assert.equal(t.supported, true);
  assert.equal(t.family?.label, "Qwen3");
  assert.ok(t.recipes.includes("fp8-hybrid"));
});

test("MoE + vision flags", () => {
  assert.equal(checkTrainable(["Qwen3MoeForCausalLM"], "qwen3_moe").moe, true);
  assert.equal(checkTrainable(["Qwen3VLForConditionalGeneration"], "qwen3_vl").vision, true);
  assert.equal(checkTrainable(["GptOssForCausalLM"], "gpt_oss").moe, true);
});

test("model_type fallback when architecture unknown", () => {
  const t = checkTrainable(["SomethingNew"], "gemma4_text");
  assert.equal(t.supported, true);
  assert.equal(t.family?.label, "Gemma4");
});

test("unsupported model", () => {
  const t = checkTrainable(["MixtralForCausalLM"], "mixtral");
  assert.equal(t.supported, false);
  assert.match(t.reason, /not in surogate/);
});
