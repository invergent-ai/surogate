import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { localDatasetInfo, localModelInfo, looksLikePath, resolveLocalPath } from "./local.ts";

test("looksLikePath distinguishes paths from Hub ids", () => {
  for (const p of ["/abs/x", "./rel", "../up", "~/home"]) assert.equal(looksLikePath(p), true, p);
  for (const id of ["Qwen/Qwen3-0.6B", "mistralai/Mixtral", "bert"]) assert.equal(looksLikePath(id), false, id);
});

test("resolveLocalPath expands ~ to home", () => {
  assert.equal(resolveLocalPath("~/x"), path.join(os.homedir(), "x"));
});

test("localModelInfo reads config.json → trainability", () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "jb-model-"));
  fs.writeFileSync(path.join(dir, "config.json"), JSON.stringify({ architectures: ["Qwen3ForCausalLM"], model_type: "qwen3" }));
  const m = localModelInfo(dir);
  assert.equal(m.exists, true);
  assert.equal(m.isDir, true);
  assert.deepEqual(m.architectures, ["Qwen3ForCausalLM"]);
  assert.equal(m.trainability.supported, true);
  fs.rmSync(dir, { recursive: true, force: true });
});

test("localModelInfo: missing path and non-folder are flagged", () => {
  assert.equal(localModelInfo("/tmp/jb-nope-xyz").exists, false);
  const f = path.join(os.tmpdir(), `jb-file-${process.pid}`);
  fs.writeFileSync(f, "x");
  const m = localModelInfo(f);
  assert.equal(m.isDir, false);
  assert.match(m.error ?? "", /folder|directory/);
  fs.rmSync(f, { force: true });
});

test("localDatasetInfo: file infers loader type, dir is dir", () => {
  const f = path.join(os.tmpdir(), `jb-ds-${process.pid}.jsonl`);
  fs.writeFileSync(f, '{"a":1}\n');
  const fi = localDatasetInfo(f);
  assert.equal(fi.kind, "file");
  assert.equal(fi.dsType, "json");
  assert.ok(fi.sizeBytes > 0);
  fs.rmSync(f, { force: true });

  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "jb-dsdir-"));
  assert.equal(localDatasetInfo(dir).kind, "dir");
  fs.rmSync(dir, { recursive: true, force: true });

  assert.equal(localDatasetInfo("/tmp/jb-nope-xyz.parquet").exists, false);
});
