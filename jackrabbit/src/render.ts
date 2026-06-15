// Bridge: WatchState -> chart PNG (echarts/resvg) -> terminal string
// (terminal-image: Kitty/iTerm2 protocol where supported; Sixel or colored
// half-block fallback otherwise). Throttled so we don't regenerate every frame.

import { lossChartPng, lossChartRgba, type LossChartData } from "./chart.ts";
import type { WatchState } from "./state.ts";
import { sixelPreferred, viaSixel, viaTerminalImage } from "./term-image.ts";

export interface CompareRun {
  label: string;
  steps: number[];
  train: number[];
}

export function chartDataFromState(s: WatchState, compare?: CompareRun): LossChartData {
  return {
    title: "training loss",
    steps: s.lossSteps,
    train: s.lossHistory,
    evals: s.evalHistory,
    ...(compare ? { compare } : {}),
  };
}

/** Render the loss chart to a terminal string sized to `cols` x `rows` cells. */
export async function renderChartString(
  s: WatchState,
  cols: number,
  rows: number,
  compare?: CompareRun,
): Promise<string> {
  if (s.lossHistory.length === 0 || cols < 4 || rows < 2) return "";
  const data = chartDataFromState(s, compare);
  // Render at the panel's pixel aspect (cell ≈ 9x19 px).
  const logicalW = Math.min(2400, cols * 9);
  const logicalH = Math.min(1200, rows * 19);

  if (sixelPreferred()) {
    // Sixel: render to RGBA at ~panel pixel size (cell ≈ 10x20 px) and encode.
    const px = lossChartRgba(data, logicalW, logicalH, Math.min(2000, cols * 10));
    return viaSixel(px.data, px.width, px.height);
  }
  // Kitty / iTerm2 / half-block: PNG at 2x supersample for crispness.
  const png = lossChartPng(data, logicalW, logicalH, Math.min(3200, logicalW * 2));
  return viaTerminalImage(png, cols, rows);
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
  async maybeRender(s: WatchState, cols: number, rows: number, nowMs: number, compare?: CompareRun): Promise<boolean> {
    const cmpKey = compare ? `c${compare.label}:${compare.train.length}` : "";
    const key = `${s.lossHistory.length}:${s.evalHistory.length}:${cols}x${rows}:${cmpKey}`;
    if (this.rendering) return false;
    if (key === this.cacheKey && this.cache) return false;
    if (nowMs - this.last < this.intervalMs && this.cache) return false;
    this.rendering = true;
    try {
      this.cache = await renderChartString(s, cols, rows, compare);
      this.cacheKey = key;
      this.last = nowMs;
      return true;
    } finally {
      this.rendering = false;
    }
  }
}
