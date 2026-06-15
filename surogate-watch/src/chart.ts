// Real raster chart: ECharts SSR (SVG) -> resvg -> PNG buffer.
// Pure-npm, no node-canvas / browser. The PNG is displayed inline via a
// terminal graphics protocol (Kitty/Sixel/iTerm2) by the caller.

import * as echarts from "echarts";
import { Resvg } from "@resvg/resvg-js";

const BG = "#0e1018";
const TRAIN = "#5ad1c4";
const EVAL = "#d18fff";
const GRID = "#1b1e28";
const AXIS = "#2a2e3a";
const MUTED = "#6b7280";

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
  chart.setOption({
    backgroundColor: BG,
    animation: false,
    grid: { left: 56, right: 22, top: 26, bottom: 28 },
    title: {
      text: data.title ?? "training loss",
      left: 12,
      top: 6,
      textStyle: { color: MUTED, fontSize: 13, fontWeight: 600 },
    },
    xAxis: {
      type: "value",
      min: minStep,
      max: maxStep,
      axisLine: { lineStyle: { color: AXIS } },
      axisLabel: { color: MUTED },
      splitLine: { show: false },
    },
    yAxis: {
      type: "value",
      scale: true,
      axisLine: { show: false },
      axisTick: { show: false },
      axisLabel: { color: MUTED },
      splitLine: { lineStyle: { color: GRID } },
    },
    series: [
      {
        name: "train",
        type: "line",
        showSymbol: false,
        smooth: true,
        data: data.steps.map((s, i) => [s, data.train[i]]),
        lineStyle: { color: TRAIN, width: 2.5 },
        areaStyle: {
          color: {
            type: "linear",
            x: 0,
            y: 0,
            x2: 0,
            y2: 1,
            colorStops: [
              { offset: 0, color: "rgba(90,209,196,0.35)" },
              { offset: 1, color: "rgba(90,209,196,0.02)" },
            ],
          },
        },
      },
      {
        name: "eval",
        type: "scatter",
        data: data.evals,
        symbol: "diamond",
        symbolSize: 11,
        itemStyle: { color: EVAL, borderColor: BG, borderWidth: 1.5 },
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
