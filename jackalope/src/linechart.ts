// Native terminal line chart drawn with Unicode Braille (2x4 sub-dots per cell).
//
// Each character cell packs an 8-dot grid, so a `cols x rows` panel gives a
// `2*cols x 4*rows` plotting surface — enough for crisp, vector-like curves with
// axes and labels in ANY utf-8 terminal (ssh, tmux, no image protocol needed).
// Unlike a raster image it needs no echarts/resvg/sixel per frame: it's a few
// Bresenham lines, so it's fast and flicker-free for the live dashboard.

import type { LossChartData } from "./chart.ts";
import { fgHex as fg, RESET } from "./ansi.ts";
import { C, COMPARE } from "./ui/theme.ts";

// Series + axis colors read from the live theme so the braille chart recolors
// with dark/light. (COMPARE stays fixed — it's a distinct dashed overlay.)

const ANSI = /\x1b\[[0-9;]*m/g;
const ANSI_AT = /\x1b\[[0-9;]*m/y; // sticky: match an escape at a given index
function paint(s: string, hex: string): string {
  return `${fg(hex)}${s}${RESET}`;
}
function visLen(s: string): number {
  return s.replace(ANSI, "").length;
}
/** Pad (or truncate, preserving escapes) a styled string to `n` visible cells. */
function fitVisible(s: string, n: number): string {
  const len = visLen(s);
  if (len === n) return s;
  if (len < n) return s + " ".repeat(n - len);
  let out = "";
  let count = 0;
  let i = 0;
  while (i < s.length && count < n) {
    if (s[i] === "\x1b") {
      ANSI_AT.lastIndex = i;
      const m = ANSI_AT.exec(s);
      if (m) {
        out += m[0];
        i += m[0].length;
        continue;
      }
    }
    out += s[i];
    count++;
    i++;
  }
  return out + RESET;
}

// Braille dot bit for a sub-pixel (x in {0,1}, y in {0..3}) within a cell.
const DOT = [
  [0x01, 0x08],
  [0x02, 0x10],
  [0x04, 0x20],
  [0x40, 0x80],
];

// Series color ids — a higher id draws on top where curves share a cell.
const C_COMPARE = 1;
const C_TRAIN = 2;
const C_EVAL = 3;
// Per-id ANSI foreground, read from the theme at render time.
function idFg(id: number): string {
  if (id === C_TRAIN) return fg(C.train);
  if (id === C_EVAL) return fg(C.eval);
  return fg(COMPARE);
}

class Braille {
  readonly subW: number;
  readonly subH: number;
  private mask: Uint8Array;
  private color: Uint8Array;

  constructor(
    readonly cols: number,
    readonly rows: number,
  ) {
    this.subW = cols * 2;
    this.subH = rows * 4;
    this.mask = new Uint8Array(cols * rows);
    this.color = new Uint8Array(cols * rows);
  }

  private plot(sx: number, sy: number, id: number): void {
    if (sx < 0 || sy < 0 || sx >= this.subW || sy >= this.subH) return;
    const i = (sy >> 2) * this.cols + (sx >> 1);
    this.mask[i]! |= DOT[sy & 3]![sx & 1]!;
    if (id >= this.color[i]!) this.color[i] = id;
  }

  /** A small diamond marker centered on a sub-pixel (for sparse eval points). */
  marker(cx: number, cy: number, id: number): void {
    cx = Math.round(cx);
    cy = Math.round(cy);
    this.plot(cx, cy, id);
    this.plot(cx - 1, cy, id);
    this.plot(cx + 1, cy, id);
    this.plot(cx, cy - 1, id);
    this.plot(cx, cy + 1, id);
  }

  /** Bresenham line between two sub-pixel coordinates. */
  line(x0: number, y0: number, x1: number, y1: number, id: number): void {
    x0 = Math.round(x0);
    y0 = Math.round(y0);
    x1 = Math.round(x1);
    y1 = Math.round(y1);
    const dx = Math.abs(x1 - x0);
    const dy = -Math.abs(y1 - y0);
    const sx = x0 < x1 ? 1 : -1;
    const sy = y0 < y1 ? 1 : -1;
    let err = dx + dy;
    for (;;) {
      this.plot(x0, y0, id);
      if (x0 === x1 && y0 === y1) break;
      const e2 = 2 * err;
      if (e2 >= dy) {
        err += dy;
        x0 += sx;
      }
      if (e2 <= dx) {
        err += dx;
        y0 += sy;
      }
    }
  }

  /** Colored braille for cell-row `cy`, exactly `cols` visible cells wide. */
  rowString(cy: number): string {
    let out = "";
    let cur = -1;
    for (let cx = 0; cx < this.cols; cx++) {
      const i = cy * this.cols + cx;
      const m = this.mask[i]!;
      if (m === 0) {
        if (cur !== -1) {
          out += RESET;
          cur = -1;
        }
        out += " ";
        continue;
      }
      const id = this.color[i]!;
      if (id !== cur) {
        out += idFg(id);
        cur = id;
      }
      out += String.fromCharCode(0x2800 + m);
    }
    if (cur !== -1) out += RESET;
    return out;
  }
}

function fmtY(v: number, range: number): string {
  const a = Math.abs(v);
  if (range >= 100 || a >= 1000) return Math.round(v).toString();
  if (range >= 10) return v.toFixed(1);
  if (range >= 1) return v.toFixed(2);
  return v.toFixed(3);
}

function points(xs: number[], ys: number[]): Array<[number, number]> {
  const out: Array<[number, number]> = [];
  for (let i = 0; i < ys.length; i++) {
    const x = xs[i];
    const y = ys[i];
    if (x !== undefined && y !== undefined && Number.isFinite(y)) out.push([x, y]);
  }
  return out;
}

/** Render the loss chart as a colored multi-line string of exactly `rows` lines,
 *  each `cols` visible cells wide. Returns "" when there's no data or no room. */
export function renderLineChart(data: LossChartData, cols: number, rows: number): string {
  const train = points(data.steps, data.train);
  const evals = data.evals.filter(([, y]) => Number.isFinite(y));
  const cmp = data.compare ? points(data.compare.steps, data.compare.train) : [];
  if (train.length === 0 && evals.length === 0) return "";

  const P = rows - 3; // plot cell-rows (title + axis underline + x-labels take 3)
  if (P < 2 || cols < 24) return "";

  // data bounds (+ small vertical padding so curves don't touch the frame)
  let xMin = Infinity;
  let xMax = -Infinity;
  let yMin = Infinity;
  let yMax = -Infinity;
  for (const series of [train, evals, cmp]) {
    for (const [x, y] of series) {
      if (x < xMin) xMin = x;
      if (x > xMax) xMax = x;
      if (y < yMin) yMin = y;
      if (y > yMax) yMax = y;
    }
  }
  if (xMax <= xMin) xMax = xMin + 1;
  if (yMax <= yMin) {
    yMax = yMin + 1;
    yMin -= 1;
  }
  const padY = (yMax - yMin) * 0.06;
  yMin -= padY;
  yMax += padY;
  const yRange = yMax - yMin;

  // y-axis labels (evenly spaced rows); pick gutter width from the widest label
  const nLabels = Math.min(P, Math.max(3, Math.floor(P / 2)));
  const labelAt = new Map<number, string>();
  for (let k = 0; k < nLabels; k++) {
    const row = Math.round((k / (nLabels - 1)) * (P - 1));
    labelAt.set(row, fmtY(yMax - (row / (P - 1)) * yRange, yRange));
  }
  let G = 4;
  for (const s of labelAt.values()) G = Math.max(G, s.length);
  G = Math.min(G, 8);
  const Pw = cols - G - 1; // plot cells (1 col for the y-axis bar)
  if (Pw < 6) return "";

  const canvas = new Braille(Pw, P);
  const mapX = (x: number) => ((x - xMin) / (xMax - xMin)) * (canvas.subW - 1);
  const mapY = (y: number) => (1 - (y - yMin) / yRange) * (canvas.subH - 1);
  const draw = (pts: Array<[number, number]>, id: number) => {
    if (pts.length === 1) {
      canvas.line(mapX(pts[0]![0]), mapY(pts[0]![1]), mapX(pts[0]![0]), mapY(pts[0]![1]), id);
    }
    for (let i = 1; i < pts.length; i++) {
      canvas.line(mapX(pts[i - 1]![0]), mapY(pts[i - 1]![1]), mapX(pts[i]![0]), mapY(pts[i]![1]), id);
    }
  };
  draw(cmp, C_COMPARE);
  draw(train, C_TRAIN);
  for (const [x, y] of evals) canvas.marker(mapX(x), mapY(y), C_EVAL); // sparse → markers

  // title + legend
  const title = data.compare ? `training loss   vs ${data.compare.label}` : (data.title ?? "training loss");
  let legend = `${paint("●", C.train)}${paint(" train", C.muted)}`;
  if (evals.length) legend += `  ${paint("◆", C.eval)}${paint(" eval", C.muted)}`;
  if (data.compare) legend += `  ${paint("┈", COMPARE)}${paint(" compare", C.muted)}`;

  const lines: string[] = [];
  lines.push(fitVisible(`${paint(title, C.muted)}   ${legend}`, cols));
  const axisBar = paint("│", C.dim);
  for (let cy = 0; cy < P; cy++) {
    const lbl = labelAt.get(cy);
    const gutter = lbl ? paint(lbl.padStart(G), C.muted) : " ".repeat(G);
    lines.push(gutter + axisBar + canvas.rowString(cy));
  }
  lines.push(" ".repeat(G) + paint("└" + "─".repeat(Pw), C.dim));
  lines.push(xAxisLabels(xMin, xMax, G, Pw));
  return lines.join("\n");
}

function xAxisLabels(xMin: number, xMax: number, G: number, Pw: number): string {
  const track = new Array<string>(Pw).fill(" ");
  const place = (str: string, start: number) => {
    start = Math.max(0, Math.min(Pw - str.length, start));
    for (let i = 0; i < str.length; i++) track[start + i] = str[i]!;
  };
  const left = Math.round(xMin).toString();
  const mid = Math.round((xMin + xMax) / 2).toString();
  const right = Math.round(xMax).toString();
  place(left, 0);
  place(mid, Math.floor((Pw - mid.length) / 2));
  place(right, Pw - right.length);
  return " ".repeat(G + 1) + paint(track.join(""), C.muted);
}
