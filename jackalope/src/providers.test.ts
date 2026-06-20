import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { deleteProvider, getProvider, loadProviders, providerId, saveProvider } from "./providers.ts";

function withTempHome(fn: () => void): void {
  const prev = process.env["HOME"];
  process.env["HOME"] = fs.mkdtempSync(path.join(os.tmpdir(), "prov-"));
  try {
    fn();
  } finally {
    if (prev === undefined) delete process.env["HOME"];
    else process.env["HOME"] = prev;
  }
}

test("providers: empty by default", () => {
  withTempHome(() => assert.deepEqual(loadProviders(), []));
});

test("providers: save / get / upsert / delete round-trip", () => {
  withTempHome(() => {
    const ssh = { id: providerId("ssh", "gpu@box"), kind: "ssh" as const, label: "box", host: "gpu@box", addedAt: "t" };
    saveProvider(ssh);
    assert.equal(loadProviders().length, 1);
    assert.equal(getProvider(ssh.id)?.host, "gpu@box");

    // upsert by id: same target re-added updates, doesn't duplicate
    saveProvider({ ...ssh, label: "renamed" });
    assert.equal(loadProviders().length, 1);
    assert.equal(getProvider(ssh.id)?.label, "renamed");

    const modal = { id: providerId("modal", "myws"), kind: "modal" as const, label: "myws", workspace: "myws", gpu: "H100", addedAt: "t" };
    saveProvider(modal);
    assert.equal(loadProviders().length, 2);

    deleteProvider(ssh.id);
    const left = loadProviders();
    assert.equal(left.length, 1);
    assert.equal(left[0]!.kind, "modal");
  });
});

test("providerId: verbatim key (no lossy slug → no collisions)", () => {
  assert.equal(providerId("ssh", " user@host:22 "), "ssh:user@host:22");
  assert.equal(providerId("dstack", "runpod"), "dstack:runpod");
  assert.equal(providerId("modal", "  "), "modal:default");
  // distinct hosts that a slug would have collapsed stay distinct
  assert.notEqual(providerId("ssh", "a-1"), providerId("ssh", "a_1"));
});
