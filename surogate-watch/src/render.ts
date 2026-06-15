// Bridge: WatchState -> chart PNG (echarts/resvg) -> terminal string
// (terminal-image: Kitty/iTerm2 protocol where supported, colored half-block
// fallback otherwise). Throttled so we don't regenerate every frame.

import terminalImage from "terminal-image";
import { lossChartPng, type LossChartData } from "./chart.ts";
import type { WatchState } from "./state.ts";

export function chartDataFromState(s: WatchState): LossChartData {
  return { title: "training loss", steps: s.lossSteps, train: s.lossHistory, evals: s.evalHistory };
}

/** Render the loss chart to a terminal string sized to `cols` x `rows` cells. */
export async function renderChartString(s: WatchState, cols: number, rows: number): Promise<string> {
  if (s.lossHistory.length === 0 || cols < 4 || rows < 2) return "";
  const pngW = Math.max(600, cols * 12);
  const pngH = Math.max(220, rows * 24);
  const png = lossChartPng(chartDataFromState(s), pngW, pngH, pngW);
  return terminalImage.buffer(png, { width: cols, height: rows, preserveAspectRatio: true });
}

/** A throttled chart renderer: regenerates at most once per `intervalMs`. */
export class ChartRenderer {
  private last = 0;
  private rendering = false;
  private cache = "";
  private cacheKey = "";

  constructor(private readonly intervalMs = 700) {}

  current(): string {
    return this.cache;
  }

  /** Returns true if a new image was produced (so the caller can re-render). */
  async maybeRender(s: WatchState, cols: number, rows: number, nowMs: number): Promise<boolean> {
    const key = `${s.lossHistory.length}:${s.evalHistory.length}:${cols}x${rows}`;
    if (this.rendering) return false;
    if (key === this.cacheKey && this.cache) return false;
    if (nowMs - this.last < this.intervalMs && this.cache) return false;
    this.rendering = true;
    try {
      const img = await renderChartString(s, cols, rows);
      this.cache = img;
      this.cacheKey = key;
      this.last = nowMs;
      return true;
    } finally {
      this.rendering = false;
    }
  }
}
