// Real raster chart: ECharts SSR (SVG) -> resvg -> PNG buffer.
// Pure-npm, no node-canvas / browser. The PNG is displayed inline via a
// terminal graphics protocol (Kitty/Sixel/iTerm2) by the caller.

import * as echarts from "echarts";
import { Resvg } from "@resvg/resvg-js";

const BG = "#14120c"; // warm dark (brand)
const TRAIN = "#ffd15c"; // brand gold
const EVAL = "#9fd0ff"; // cool blue
const GRID = "#241f12";
const AXIS = "#3a3320";
const MUTED = "#8a8270";

export interface LossChartData {
  title?: string;
  steps: number[]; // x for each train point
  train: number[]; // train loss per step
  evals: Array<[number, number]>; // [step, eval loss]
}

/** Render the loss chart to an SVG string at logical width/height (cells*cellpx). */
export function lossChartSvg(data: LossChartData, width: number, height: number): string {
  const chart = echarts.init(null, null, { renderer: "svg", ssr: true, width, height });
  const minStep = data.steps.length ? data.steps[0]! : 0;
  const maxStep = data.steps.length ? data.steps[data.steps.length - 1]! : 1;
  // scale styling with the render height so high-DPI renders stay proportional
  const font = Math.max(11, Math.round(height / 22));
  const lineW = Math.max(2, Math.round(height / 95));
  const symbol = Math.max(8, Math.round(height / 24));
  chart.setOption({
    backgroundColor: BG,
    animation: false,
    grid: { left: font * 4, right: font * 1.5, top: font * 2, bottom: font * 2 },
    title: {
      text: data.title ?? "training loss",
      left: font,
      top: Math.round(font / 2),
      textStyle: { color: MUTED, fontSize: font, fontWeight: 600 },
    },
    xAxis: {
      type: "value",
      min: minStep,
      max: maxStep,
      axisLine: { lineStyle: { color: AXIS, width: Math.max(1, Math.round(lineW / 2)) } },
      axisLabel: { color: MUTED, fontSize: font },
      splitLine: { show: false },
    },
    yAxis: {
      type: "value",
      scale: true,
      axisLine: { show: false },
      axisTick: { show: false },
      axisLabel: { color: MUTED, fontSize: font },
      splitLine: { lineStyle: { color: GRID, width: Math.max(1, Math.round(lineW / 2)) } },
    },
    series: [
      {
        name: "train",
        type: "line",
        showSymbol: false,
        smooth: true,
        data: data.steps.map((s, i) => [s, data.train[i]]),
        lineStyle: { color: TRAIN, width: lineW },
        areaStyle: {
          color: {
            type: "linear",
            x: 0,
            y: 0,
            x2: 0,
            y2: 1,
            colorStops: [
              { offset: 0, color: "rgba(255,209,92,0.34)" },
              { offset: 1, color: "rgba(255,209,92,0.02)" },
            ],
          },
        },
      },
      {
        name: "eval",
        type: "scatter",
        data: data.evals,
        symbol: "diamond",
        symbolSize: symbol,
        itemStyle: { color: EVAL, borderColor: BG, borderWidth: Math.max(1, Math.round(symbol / 7)) },
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
  const png = new Resvg(svg, { fitTo: { mode: "width", value: pixelW }, background: BG }).render().asPng();
  return Buffer.from(png);
}
