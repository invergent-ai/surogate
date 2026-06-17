import { test } from "node:test";
import assert from "node:assert/strict";
import { searchModels, modelDetail, paramsB } from "./hf.ts";

const realFetch = globalThis.fetch;
function mockFetch(payload: unknown, status = 200) {
  globalThis.fetch = (async () => ({ ok: status < 400, status, json: async () => payload })) as unknown as typeof fetch;
}

test("searchModels parses hits", async () => {
  mockFetch([{ id: "Qwen/Qwen3-0.6B", downloads: 1000, likes: 50, pipeline_tag: "text-generation", gated: false }]);
  const hits = await searchModels("qwen3");
  assert.equal(hits[0]!.id, "Qwen/Qwen3-0.6B");
  assert.equal(hits[0]!.downloads, 1000);
  globalThis.fetch = realFetch;
});

test("modelDetail extracts config + params", async () => {
  mockFetch({ config: { architectures: ["Qwen3ForCausalLM"], model_type: "qwen3" }, safetensors: { total: 1_200_000_000 }, downloads: 9, gated: "manual" });
  const d = await modelDetail("Qwen/Qwen3-0.6B");
  assert.deepEqual(d.architectures, ["Qwen3ForCausalLM"]);
  assert.equal(d.modelType, "qwen3");
  assert.equal(paramsB(d), 0.6);
  assert.equal(d.gated, "manual");
  globalThis.fetch = realFetch;
});

test("429 throws a helpful error", async () => {
  mockFetch({}, 429);
  await assert.rejects(() => searchModels("x"), /rate limited/);
  globalThis.fetch = realFetch;
});
