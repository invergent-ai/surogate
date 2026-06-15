import { test } from "node:test";
import assert from "node:assert/strict";
import { WatchState } from "./state.ts";
import { parseLine } from "./records.ts";
import { computeInsights } from "./insights.ts";

function load(...lines: string[]) {
  const s = new WatchState();
  s.ingest(lines.map(parseLine).filter((r): r is NonNullable<typeof r> => r !== null));
  return s;
}

test("health: descending loss is steady, % and ETA present", () => {
  const s = load(
    '{"type":"config","ts":0,"recipe":"bf16","max_steps":100}',
    '{"step":10,"ts":10,"train/loss":2.0,"train/tokens_per_second":29000}',
    '{"step":20,"ts":20,"train/loss":1.0,"train/tokens_per_second":29000}',
  );
  const g = computeInsights(s);
  assert.ok(g.health.some((i) => /loss ↓/.test(i.text)));
  assert.ok(g.health.some((i) => /%/.test(i.text)));
  assert.ok(g.health.some((i) => /tok\/s/.test(i.text)));
});

test("alerts: hot gpu flagged; nominal otherwise", () => {
  const hot = load('{"type":"gpu","gpu_id":0,"step":1,"ts":1,"temperature":82,"gpu_util":90,"mem_util":50}');
  assert.ok(computeInsights(hot).alerts.some((i) => /hot/.test(i.text) && i.color === "red"));
  const cool = load('{"type":"gpu","gpu_id":0,"step":5,"ts":1,"temperature":55,"gpu_util":90,"mem_util":50}');
  assert.ok(computeInsights(cool).alerts.some((i) => /nominal/.test(i.text)));
});

test("tips: bf16 suggests fp8; facts include model", () => {
  const s = load('{"type":"config","ts":0,"recipe":"bf16","model":"Qwen/Qwen3-0.6B"}');
  const g = computeInsights(s);
  assert.ok(g.tips.some((i) => /fp8/.test(i.text)));
  assert.ok(g.facts.some((i) => /Qwen\/Qwen3-0\.6B/.test(i.text)));
});
