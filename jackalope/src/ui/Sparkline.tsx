import React from "react";
import { Text } from "ink";
import { C } from "./theme.ts";

// A compact single-line block sparkline. Normalizes over its own min/max so any
// series (loss, tok/s, temp, power) reads as a shape. Optional gradient coloring.
const RAMP = ["▁", "▂", "▃", "▄", "▅", "▆", "▇", "█"];

export function Sparkline({
  values,
  width = 28,
  color = C.accent,
}: {
  values: number[];
  width?: number;
  color?: string;
}) {
  if (values.length === 0) return <Text color={C.dim}>{"·".repeat(width)}</Text>;
  // take the most recent `width` samples
  const v = values.slice(-width);
  const min = Math.min(...v);
  const max = Math.max(...v);
  const span = max - min || 1;
  const chars = v.map((x) => RAMP[Math.max(0, Math.min(7, Math.round(((x - min) / span) * 7)))]!).join("");
  return <Text color={color}>{chars}</Text>;
}
