// Bridge: WatchState -> chart string. Three renderers:
//   braille  — native Unicode line chart (default; crisp lines, any terminal)
//   image    — ECharts/resvg PNG via the Kitty/iTerm2 graphics protocol
//   sixel    — ECharts/resvg RGBA encoded to Sixel
// The image/sixel paths regenerate a raster per frame, so they're throttled.

import type { LossChartData } from "./chart.ts";
import { renderLineChart } from "./linechart.ts";
import type { WatchState } from "./state.ts";
import { getTheme } from "./ui/theme.ts";

// chart.ts (resvg native addon) and term-image.ts (sixel) are imported lazily —
// only when a raster graphics mode is actually selected. The default braille
// renderer needs neither, so a standalone build never loads them at startup.

export type GraphicsMode = "braille" | "image" | "sixel";

let _mode: GraphicsMode = "braille";
export function setGraphicsMode(m: GraphicsMode): void {
  _mode = m;
}
export function graphicsMode(): GraphicsMode {
  return _mode;
}

export interface CompareRun {
  label: string;
  steps: number[];
  train: number[];
}

export function chartDataFromState(s: WatchState, compare?: CompareRun, smooth = false): LossChartData {
  return {
    title: smooth ? "training loss (smoothed)" : "training loss",
    steps: s.lossSteps,
    train: smooth ? s.smoothedLoss() : s.lossHistory,
    evals: s.evalHistory,
    ...(compare ? { compare } : {}),
  };
}

interface PixelOpts {
  cellW: number;
  cellH: number;
  capW: number;
  capH: number;
  superSample?: number; // pixelW = logicalW * superSample, unless pixelW is given
  capPixelW?: number;
  pixelW?: number;
}

/** Derive logical (SVG) and raster pixel sizes from a cell grid + per-cell pixel
 *  estimate, shared by the image and sixel paths so their budgets stay legible. */
function termPixelSize(cols: number, rows: number, o: PixelOpts): { logicalW: number; logicalH: number; pixelW: number } {
  const logicalW = Math.min(o.capW, cols * o.cellW);
  const logicalH = Math.min(o.capH, rows * o.cellH);
  let pixelW = o.pixelW ?? Math.round(logicalW * (o.superSample ?? 1));
  if (o.capPixelW) pixelW = Math.min(o.capPixelW, pixelW);
  return { logicalW, logicalH, pixelW };
}

// The raster renderers (resvg + sixel) load on first use, then are cached — the
// chart re-renders ~1-2×/sec in image/sixel mode and shouldn't re-resolve the
// dynamic imports per frame.
let rasterMods: Promise<[typeof import("./chart.ts"), typeof import("./term-image.ts")]> | null = null;
function loadRaster() {
  if (!rasterMods) rasterMods = Promise.all([import("./chart.ts"), import("./term-image.ts")]);
  return rasterMods;
}

/** Render the loss chart to a terminal string sized to `cols` x `rows` cells. */
export async function renderChartString(
  s: WatchState,
  cols: number,
  rows: number,
  compare?: CompareRun,
  smooth = false,
): Promise<string> {
  if (s.lossHistory.length === 0 || cols < 4 || rows < 2) return "";
  const data = chartDataFromState(s, compare, smooth);

  if (_mode === "braille") return renderLineChart(data, cols, rows);

  if (_mode === "sixel") {
    // image2sixel quantizes the whole buffer synchronously, so keep it small.
    const [{ lossChartRgba }, { viaSixel }] = await loadRaster();
    const plan = termPixelSize(cols, rows, { cellW: 7, cellH: 14, capW: 1400, capH: 700, pixelW: Math.min(1100, cols * 7) });
    const px = lossChartRgba(data, plan.logicalW, plan.logicalH, plan.pixelW);
    return viaSixel(px.data, px.width, px.height);
  }

  // image: Kitty / iTerm2 — PNG at 2x supersample for crispness.
  const [{ lossChartPng }, { viaTerminalImage }] = await loadRaster();
  const plan = termPixelSize(cols, rows, { cellW: 9, cellH: 19, capW: 2400, capH: 1200, superSample: 2, capPixelW: 3200 });
  const png = lossChartPng(data, plan.logicalW, plan.logicalH, plan.pixelW);
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

  /** Force the next maybeRender to regenerate (e.g. after a theme change), while
   *  keeping the current image on screen until the new one is ready (no blink). */
  invalidate(): void {
    this.cacheKey = "";
    this.last = 0;
  }

  /** Returns true if a new image was produced (so the caller can re-render). */
  async maybeRender(
    s: WatchState,
    cols: number,
    rows: number,
    nowMs: number,
    compare?: CompareRun,
    smooth = false,
  ): Promise<boolean> {
    const cmpKey = compare ? `c${compare.label}:${compare.train.length}` : "";
    const key = `${getTheme()}:${_mode}:${smooth ? "s" : ""}:${s.lossHistory.length}:${s.evalHistory.length}:${cols}x${rows}:${cmpKey}`;
    if (this.rendering) return false;
    if (key === this.cacheKey && this.cache) return false;
    if (nowMs - this.last < this.intervalMs && this.cache) return false;
    this.rendering = true;
    try {
      this.cache = await renderChartString(s, cols, rows, compare, smooth);
      this.cacheKey = key;
      this.last = nowMs;
      return true;
    } finally {
      this.rendering = false;
    }
  }
}
