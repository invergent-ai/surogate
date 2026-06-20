// Real raster chart: ECharts SSR (SVG) -> resvg -> PNG buffer.
// Pure-npm, no node-canvas / browser. The PNG is displayed inline via a
// terminal graphics protocol (Kitty/Sixel/iTerm2) by the caller.

import * as echarts from "echarts";
import { Resvg } from "@resvg/resvg-js";
import { C, COMPARE, getTheme, hexToRgb } from "./ui/theme.ts";

// Series identity for the raster renderer. COMPARE lives in theme.ts (re-exported
// here) so the braille renderer can use it without loading this module.
export const TRAIN = "#ffd15c"; // brand gold
export const EVAL = "#9fd0ff"; // cool blue
export { COMPARE };

function rgba(hex: string, a: number): string {
  const [r, g, b] = hexToRgb(hex);
  return `rgba(${r},${g},${b},${a})`;
}

// Canvas tones flip with the theme (the braille chart draws onto the terminal
// background; this raster owns its own canvas, so it needs its own bg/grid).
function pal() {
  const light = getTheme() === "light";
  // bg MATCHES the page (C.panel) so the chart blends instead of looking like a
  // dark box in light mode (or vice-versa).
  return {
    bg: light ? "#f4f4f5" : "#0b0b0b",
    grid: light ? "#e4e4e7" : "#1c1c1c",
    axis: light ? "#cfcfd4" : "#2a2a2a",
    muted: light ? "#6b6b70" : "#8a8a8a",
    train: C.train,
    eval: C.eval,
  };
}

export interface LossChartData {
  title?: string;
  steps: number[]; // x for each train point
  train: number[]; // train loss per step
  evals: Array<[number, number]>; // [step, eval loss]
  compare?: { label: string; steps: number[]; train: number[] }; // overlay run
}

/** Render the loss chart to an SVG string at logical width/height (cells*cellpx). */
export function lossChartSvg(data: LossChartData, width: number, height: number): string {
  const p = pal();
  const chart = echarts.init(null, null, { renderer: "svg", ssr: true, width, height });
  // bound the x-axis over BOTH train + eval steps, so eval-only points (logged
  // before the first train loss) aren't clipped off the chart.
  const xs = [...data.steps, ...data.evals.map((e) => e[0])];
  const minStep = xs.length ? Math.min(...xs) : 0;
  const maxStep = xs.length ? Math.max(...xs) : 1;
  // scale styling with the render height so high-DPI renders stay proportional
  const font = Math.max(11, Math.round(height / 22));
  const lineW = Math.max(2, Math.round(height / 95));
  const symbol = Math.max(8, Math.round(height / 24));
  chart.setOption({
    backgroundColor: p.bg,
    animation: false,
    grid: { left: font * 4, right: font * 1.5, top: font * 2, bottom: font * 2 },
    title: {
      text: data.compare ? `training loss   (vs ${data.compare.label})` : (data.title ?? "training loss"),
      left: font,
      top: Math.round(font / 2),
      textStyle: { color: p.muted, fontSize: font, fontWeight: 600 },
    },
    xAxis: {
      type: "value",
      min: minStep,
      max: maxStep,
      axisLine: { lineStyle: { color: p.axis, width: Math.max(1, Math.round(lineW / 2)) } },
      axisLabel: { color: p.muted, fontSize: font },
      splitLine: { show: false },
    },
    yAxis: {
      type: "value",
      scale: true,
      axisLine: { show: false },
      axisTick: { show: false },
      axisLabel: { color: p.muted, fontSize: font },
      splitLine: { lineStyle: { color: p.grid, width: Math.max(1, Math.round(lineW / 2)) } },
    },
    series: [
      {
        name: "train",
        type: "line",
        showSymbol: false,
        smooth: true,
        data: data.steps.map((s, i) => [s, data.train[i]]),
        lineStyle: { color: p.train, width: lineW },
        areaStyle: {
          color: {
            type: "linear",
            x: 0,
            y: 0,
            x2: 0,
            y2: 1,
            colorStops: [
              { offset: 0, color: rgba(p.train, 0.34) },
              { offset: 1, color: rgba(p.train, 0.02) },
            ],
          },
        },
      },
      ...(data.compare
        ? [
            {
              name: "compare",
              type: "line",
              showSymbol: false,
              smooth: true,
              data: data.compare.steps.map((s, i) => [s, data.compare!.train[i]]),
              lineStyle: { color: COMPARE, width: Math.max(1, lineW - 1), type: "dashed" },
            },
          ]
        : []),
      {
        name: "eval",
        type: "scatter",
        data: data.evals,
        symbol: "diamond",
        symbolSize: symbol,
        itemStyle: { color: p.eval, borderColor: p.bg, borderWidth: Math.max(1, Math.round(symbol / 7)) },
      },
    ],
  });
  const svg = chart.renderToSVGString();
  chart.dispose();
  return svg;
}

/** Render the loss chart to a PNG buffer at the given pixel width (height auto from aspect). */
export function lossChartPng(data: LossChartData, logicalW: number, logicalH: number, pixelW = 1840): Buffer {
  const svg = lossChartSvg(data, logicalW, logicalH);
  const png = new Resvg(svg, { fitTo: { mode: "width", value: pixelW }, background: pal().bg }).render().asPng();
  return Buffer.from(png);
}

/** Render the loss chart to raw RGBA pixels (for the Sixel path). */
export function lossChartRgba(
  data: LossChartData,
  logicalW: number,
  logicalH: number,
  pixelW = 1200,
): { data: Uint8Array; width: number; height: number } {
  const svg = lossChartSvg(data, logicalW, logicalH);
  const r = new Resvg(svg, { fitTo: { mode: "width", value: pixelW }, background: pal().bg }).render();
  return { data: r.pixels, width: r.width, height: r.height };
}
