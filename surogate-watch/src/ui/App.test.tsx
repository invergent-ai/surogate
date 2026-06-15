import { test } from "node:test";
import assert from "node:assert/strict";
import React from "react";
import { render } from "ink-testing-library";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { Feed } from "../feed.ts";
import { App } from "./App.tsx";

test("App renders Monitor without crashing", async () => {
  const p = path.join(os.tmpdir(), `smoke-${process.pid}.jsonl`);
  fs.writeFileSync(
    p,
    '{"type":"config","ts":0,"recipe":"fp8-hybrid","model":"Qwen/Qwen3-0.6B","max_steps":200,"lora":true}\n' +
      '{"step":1,"ts":1,"train/loss":2.0,"train/lr":2e-4,"train/norm":0.8,"train/epoch":0.1}\n' +
      '{"step":2,"ts":2,"train/loss":1.6,"train/lr":2e-4}\n' +
      '{"type":"gpu","gpu_id":0,"step":2,"ts":2,"temperature":55,"gpu_util":97,"mem_util":31,"power":325000}\n',
  );
  const feed = new Feed(p, true);
  const { lastFrame, stdin, unmount } = render(
    React.createElement(App, { feed, feedPath: p, surogateBin: "surogate", repoRoot: process.cwd(), version: "0.1.0" }),
  );
  await new Promise((r) => setTimeout(r, 200));

  // splash first: mascot menu with tips + devices
  const splash = lastFrame() ?? "";
  assert.match(splash, /DEVICES/);
  assert.match(splash, /TIPS/);
  assert.match(splash, /enter to open the dashboard/);

  // enter -> dashboard
  stdin.write("\r");
  await new Promise((r) => setTimeout(r, 300));
  const frame = lastFrame() ?? "";
  assert.match(frame, /surogate/);
  assert.match(frame, /Monitor/); // sidebar nav
  assert.match(frame, /Qwen\/Qwen3-0\.6B/);
  assert.match(frame, /INSIGHTS/); // right rail
  assert.match(frame, /devices/); // main page panel
  unmount();
  fs.rmSync(p, { force: true });
});
