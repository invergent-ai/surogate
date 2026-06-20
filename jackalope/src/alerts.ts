// Alert engine: detects notable training events from WatchState transitions and
// fires each one once (per run). The UI rings the bell, shows a banner, and
// best-effort desktop-notifies.

import { spawn } from "node:child_process";
import type { WatchState } from "./state.ts";

export type AlertKind = "diverge" | "plateau" | "complete" | "stalled";

export interface Alert {
  kind: AlertKind;
  text: string;
  ts: number;
  color: "red" | "warm" | "green";
}

const STALL_SECONDS = 30;

export class AlertEngine {
  private fired = new Set<string>(); // `${runId}:${kind}`

  /** Inspect state; return a newly-fired alert (or null). `runId` scopes dedupe. */
  check(s: WatchState, runId: string, nowMs: number): Alert | null {
    const once = (kind: AlertKind): boolean => {
      const key = `${runId}:${kind}`;
      if (this.fired.has(key)) return false;
      this.fired.add(key);
      return true;
    };

    // complete: reached the planned step budget
    if (s.maxSteps && s.maxSteps > 0 && s.step >= s.maxSteps && once("complete")) {
      return { kind: "complete", text: `run complete — ${s.step} steps`, ts: nowMs, color: "green" };
    }

    const loss = s.latestTrainLoss ?? 1;
    const trend = s.lossTrend(10);
    const rel = Math.abs(trend) / (Math.abs(loss) || 1);

    // diverge: loss climbing meaningfully
    if (s.lossHistory.length >= 10 && trend > 0 && rel > 0.08 && once("diverge")) {
      return { kind: "diverge", text: "loss diverging — lower LR / clip grads", ts: nowMs, color: "red" };
    }

    // plateau: long flat stretch
    if (s.lossHistory.length >= 30 && rel <= 0.003 && s.step > 20 && once("plateau")) {
      return { kind: "plateau", text: "loss plateaued — more data / lower LR?", ts: nowMs, color: "warm" };
    }

    // stalled: feed went quiet before finishing (possible crash / OOM)
    if (
      s.lastTs !== null &&
      s.step > 1 &&
      (!s.maxSteps || s.step < s.maxSteps) &&
      nowMs / 1000 - s.lastTs > STALL_SECONDS &&
      once("stalled")
    ) {
      return { kind: "stalled", text: "feed stalled >30s — run stopped or OOM?", ts: nowMs, color: "warm" };
    }

    return null;
  }

  /** Re-arm an alert kind for a run (e.g. after the feed resumes). */
  rearm(runId: string, kind: AlertKind): void {
    this.fired.delete(`${runId}:${kind}`);
  }
}

/** Best-effort desktop notification (notify-send on Linux, osascript on macOS).
 *  Never throws; silently no-ops where unsupported. */
export function desktopNotify(title: string, body: string): void {
  const fire = (cmd: string, args: string[]) => {
    try {
      const c = spawn(cmd, args, { stdio: "ignore", detached: true });
      c.on("error", () => {}); // swallow "command not found" etc. — never crash
      c.unref();
    } catch {
      /* ignore */
    }
  };
  if (process.platform === "darwin") {
    fire("osascript", ["-e", `display notification ${JSON.stringify(body)} with title ${JSON.stringify(title)}`]);
  } else if (process.platform === "linux") {
    fire("notify-send", [title, body]);
  }
}
