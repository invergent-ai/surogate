// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: AGPL-3.0-only
//
import { BENCHMARKS } from "./evaluations-data";
import type { Score } from "./evaluations-data";

interface RadarLabel {
  id: string;
  short: string;
}

interface RadarChartProps {
  scores: Record<string, Score>;
  compareScores: Record<string, { value: number | null }> | null;
  labels: RadarLabel[];
  size?: number;
  color?: string;
  compareColor?: string;
}

export function RadarChart({
  scores,
  compareScores,
  labels,
  size = 260,
  color = "#8B5CF6",
  compareColor = "#4B5468",
}: RadarChartProps) {
  const cx = size / 2;
  const cy = size / 2;
  const r = size / 2 - 30;
  const n = labels.length;
  const angleStep = (2 * Math.PI) / n;
  const startAngle = -Math.PI / 2;

  const getPoint = (index: number, value: number, maxVal: number) => {
    const angle = startAngle + index * angleStep;
    const dist = (value / maxVal) * r;
    return { x: cx + dist * Math.cos(angle), y: cy + dist * Math.sin(angle) };
  };

  const gridLevels = [0.25, 0.5, 0.75, 1.0];
  const maxVals = labels.map((l) => {
    const bench = BENCHMARKS.find((b) => b.id === l.id);
    return bench?.metric === "score/10"
      ? 10
      : bench?.metric === "score/5"
        ? 5
        : 100;
  });

  const mainPoints = labels.map((l, i) => {
    const val = scores[l.id]?.value ?? 0;
    return getPoint(i, val, maxVals[i]);
  });

  const comparePoints = compareScores
    ? labels.map((l, i) => {
        const val = compareScores[l.id]?.value ?? 0;
        return getPoint(i, val, maxVals[i]);
      })
    : null;

  return (
    <svg width={size} height={size} className="block">
      {/* Grid */}
      {gridLevels.map((level, li) => {
        const pts = labels
          .map((_, i) => {
            const p = getPoint(i, level * maxVals[i], maxVals[i]);
            return `${p.x},${p.y}`;
          })
          .join(" ");
        return (
          <polygon
            key={li}
            points={pts}
            fill="none"
            className="stroke-border/50"
            strokeWidth="0.5"
          />
        );
      })}
      {/* Axes */}
      {labels.map((_, i) => {
        const p = getPoint(i, maxVals[i], maxVals[i]);
        return (
          <line
            key={i}
            x1={cx}
            y1={cy}
            x2={p.x}
            y2={p.y}
            className="stroke-border/30"
            strokeWidth="0.5"
          />
        );
      })}
      {/* Compare area */}
      {comparePoints && (
        <polygon
          points={comparePoints.map((p) => `${p.x},${p.y}`).join(" ")}
          fill={compareColor}
          fillOpacity="0.05"
          stroke={compareColor}
          strokeWidth="1"
          strokeDasharray="3,3"
        />
      )}
      {/* Main area */}
      <polygon
        points={mainPoints.map((p) => `${p.x},${p.y}`).join(" ")}
        fill={color}
        fillOpacity="0.12"
        stroke={color}
        strokeWidth="1.5"
      />
      {/* Dots */}
      {mainPoints.map((p, i) => (
        <circle key={i} cx={p.x} cy={p.y} r="3" fill={color} />
      ))}
      {/* Labels */}
      {labels.map((l, i) => {
        const angle = startAngle + i * angleStep;
        const lx = cx + (r + 18) * Math.cos(angle);
        const ly = cy + (r + 18) * Math.sin(angle);
        return (
          <text
            key={i}
            x={lx}
            y={ly}
            className="fill-muted-foreground text-[8px] font-display"
            textAnchor={
              lx < cx - 5 ? "end" : lx > cx + 5 ? "start" : "middle"
            }
            dominantBaseline={
              ly < cy - 5 ? "auto" : ly > cy + 5 ? "hanging" : "central"
            }
          >
            {l.short}
          </text>
        );
      })}
    </svg>
  );
}
