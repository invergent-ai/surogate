import { test } from "node:test";
import assert from "node:assert/strict";
import { dstackTaskYaml } from "./dstack.ts";

const CFG = { gpu: "H100", count: 2, image: "ghcr.io/acme/surogate:latest" };

test("dstackTaskYaml has type/name/image/gpu resource", () => {
  const y = dstackTaskYaml("sur-run1", CFG, 'model: "x"\nmax_steps: 10\n', "surogate");
  assert.match(y, /^type: task$/m);
  assert.match(y, /^name: sur-run1$/m);
  assert.match(y, /^image: ghcr\.io\/acme\/surogate:latest$/m);
  assert.match(y, /gpu: "H100:2"/);
});

test("dstackTaskYaml embeds the config via heredoc + tails metrics to stdout", () => {
  const y = dstackTaskYaml("r", CFG, 'model: "x"\nmax_steps: 10\n', "surogate");
  assert.match(y, /cat > config\.yaml <<'JACKALOPE_YAML'/);
  assert.match(y, /model: "x"/); // config content embedded
  assert.match(y, /SUROGATE_METRICS_PATH=metrics\.jsonl surogate sft config\.yaml/);
  assert.match(y, /tail -n \+1 -F metrics\.jsonl/);
});

test("dstackTaskYaml pins backend + region when given", () => {
  const y = dstackTaskYaml("r", { ...CFG, backend: "runpod", region: "us-east-1" }, "model: x\n", "surogate");
  assert.match(y, /backends: \[runpod\]/);
  assert.match(y, /regions: \[us-east-1\]/);
  // omitted when not set
  const y2 = dstackTaskYaml("r", CFG, "model: x\n", "surogate");
  assert.ok(!/backends:/.test(y2) && !/regions:/.test(y2));
});
