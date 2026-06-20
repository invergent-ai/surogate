import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import {
  dstackInstallPlan,
  isSurogateSource,
  loadOnboarding,
  localInstallPlans,
  remoteInstallPlan,
  saveOnboarding,
} from "./setup.ts";

test("isSurogateSource: true only for a pyproject + surogate package", () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "sur-src-"));
  assert.equal(isSurogateSource(dir), false);
  fs.writeFileSync(path.join(dir, "pyproject.toml"), "[project]\n");
  assert.equal(isSurogateSource(dir), false); // pyproject alone isn't enough
  fs.mkdirSync(path.join(dir, "surogate"));
  assert.equal(isSurogateSource(dir), true);
});

test("localInstallPlans: script-first off a checkout, source-first in a checkout", () => {
  const plain = fs.mkdtempSync(path.join(os.tmpdir(), "plain-"));
  assert.equal(localInstallPlans(plain)[0]!.mode, "script");

  const src = fs.mkdtempSync(path.join(os.tmpdir(), "src-"));
  fs.writeFileSync(path.join(src, "pyproject.toml"), "[project]\n");
  fs.mkdirSync(path.join(src, "surogate"));
  const plans = localInstallPlans(src);
  assert.equal(plans[0]!.mode, "source");
  assert.equal(plans[0]!.command, "uv pip install -e .");
  assert.equal(plans[0]!.cwd, src);
});

test("install plans carry the documented commands", () => {
  assert.match(localInstallPlans("/nope")[0]!.command, /curl -LsSf https:\/\/surogate\.ai\/install\.sh \| sh/);
  assert.match(remoteInstallPlan().command, /install\.sh \| sh/);
  assert.match(dstackInstallPlan().command, /dstack\[all\]/);
});

test("onboarding state round-trips through $HOME", () => {
  const home = fs.mkdtempSync(path.join(os.tmpdir(), "home-"));
  const prev = process.env["HOME"];
  process.env["HOME"] = home;
  try {
    assert.equal(loadOnboarding(), null);
    saveOnboarding({ completed: true, compute: "ssh", sshHost: "gpu@box", surogateOk: true, ts: "t" });
    const got = loadOnboarding();
    assert.equal(got?.completed, true);
    assert.equal(got?.compute, "ssh");
    assert.equal(got?.sshHost, "gpu@box");
  } finally {
    if (prev === undefined) delete process.env["HOME"];
    else process.env["HOME"] = prev;
  }
});
