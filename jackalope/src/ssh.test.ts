import { test } from "node:test";
import assert from "node:assert/strict";
import { parseSshTarget, remoteLaunchCommand, sshBaseArgs } from "./ssh.ts";
import { parseGpuCsv } from "./launch.ts";

test("parseSshTarget splits host/port, keeps aliases", () => {
  assert.deepEqual(parseSshTarget("ubuntu@1.2.3.4"), { host: "ubuntu@1.2.3.4" });
  assert.deepEqual(parseSshTarget("ubuntu@1.2.3.4:2222"), { host: "ubuntu@1.2.3.4", port: 2222 });
  assert.deepEqual(parseSshTarget("gpu-box"), { host: "gpu-box" });
});

test("sshBaseArgs is non-interactive with keepalives + optional port/key", () => {
  const a = sshBaseArgs({ host: "h", port: 2222, identityFile: "~/k" });
  assert.ok(a.includes("BatchMode=yes"));
  assert.ok(a.includes("ServerAliveInterval=15"));
  assert.deepEqual(a.slice(-4), ["-p", "2222", "-i", "~/k"]);
  // no port/key → none of those flags
  const b = sshBaseArgs({ host: "h" });
  assert.ok(!b.includes("-p") && !b.includes("-i"));
});

test("remoteLaunchCommand runs surogate detached in tmux with the metrics path", () => {
  const cmd = remoteLaunchCommand("~/.surogate-watch/remote/run1", "sur_run1", "surogate");
  assert.match(cmd, /tmux new-session -d -s sur_run1/);
  assert.match(cmd, /SUROGATE_METRICS_PATH=~\/\.surogate-watch\/remote\/run1\/metrics\.jsonl/);
  assert.match(cmd, /surogate sft config\.yaml > train\.log 2>&1 < \/dev\/null/);
});

test("parseGpuCsv handles [Not Supported] + blank lines", () => {
  const out = "0, NVIDIA RTX 5090, 32768, 1024, 12\n\n1, NVIDIA RTX 5090, 32768, [Not Supported], [N/A]\n";
  const g = parseGpuCsv(out);
  assert.equal(g.length, 2);
  assert.equal(g[0]!.memMB, 32768);
  assert.equal(g[0]!.util, 12);
  assert.equal(g[1]!.memUsedMB, null); // [Not Supported] → null
  assert.equal(g[1]!.util, null);
});
