import { test } from "node:test";
import assert from "node:assert/strict";
import { dstackTaskYaml, mergeDstackBackend } from "./dstack.ts";

const CFG = { gpu: "H100", count: 2, image: "ghcr.io/acme/surogate:latest" };

test("mergeDstackBackend creates main/backends on an empty config", () => {
  const doc = mergeDstackBackend(null, "runpod", { api_key: "rp_key" }) as any;
  const main = doc.projects.find((p: any) => p.name === "main");
  assert.equal(main.backends.length, 1);
  assert.equal(main.backends[0].type, "runpod");
  assert.equal(main.backends[0].creds.api_key, "rp_key");
});

test("mergeDstackBackend preserves other projects and other backends", () => {
  const existing = {
    projects: [
      { name: "other", backends: [{ type: "aws", creds: { type: "access_key", access_key: "a", secret_key: "s" } }] },
      { name: "main", backends: [{ type: "vastai", creds: { type: "api_key", api_key: "keep_me" } }] },
    ],
  };
  const doc = mergeDstackBackend(existing, "runpod", { api_key: "rp_key" }) as any;
  // untouched project survives
  assert.ok(doc.projects.find((p: any) => p.name === "other"));
  const main = doc.projects.find((p: any) => p.name === "main");
  // existing different-type backend in main is kept, new one appended
  const types = main.backends.map((b: any) => b.type).sort();
  assert.deepEqual(types, ["runpod", "vastai"]);
  assert.equal(main.backends.find((b: any) => b.type === "vastai").creds.api_key, "keep_me");
});

test("mergeDstackBackend refuses a non-list projects rather than clobbering it", () => {
  // a malformed-but-present config must not be silently overwritten
  assert.throws(() => mergeDstackBackend({ projects: { default: "x" } as any }, "runpod", { api_key: "k" }), /non-list `projects`/);
});

test("mergeDstackBackend replaces a same-type backend (idempotent re-config)", () => {
  const existing = { projects: [{ name: "main", backends: [{ type: "runpod", creds: { type: "api_key", api_key: "old" } }] }] };
  const doc = mergeDstackBackend(existing, "runpod", { api_key: "new" }) as any;
  const main = doc.projects.find((p: any) => p.name === "main");
  assert.equal(main.backends.length, 1); // not duplicated
  assert.equal(main.backends[0].creds.api_key, "new");
});

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
