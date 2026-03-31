// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: AGPL-3.0-only
//
import { COMBINED_RPS, RPS_CHART_LEGEND, type TimeRangeId } from "./monitoring-data";

interface AreaChartProps {
  data: Record<string, number>[];
  colors: Record<string, string>;
  keys: string[];
  h?: number;
  w?: number;
}

function AreaChart({ data, colors, keys, h = 120, w = 700 }: AreaChartProps) {
  const allVals = data.flatMap((d) => keys.map((k) => (d[k] as number) ?? 0));
  const max = Math.max(...allVals) * 1.1 || 1;

  return (
    <svg width={w} height={h} className="block w-full" viewBox={`0 0 ${w} ${h}`} preserveAspectRatio="none">
      {[0.25, 0.5, 0.75].map((f) => (
        <line key={f} x1="0" y1={h - f * (h - 8) - 4} x2={w} y2={h - f * (h - 8) - 4} stroke="var(--border)" strokeWidth="1" />
      ))}
      {keys.map((key) => {
        const color = colors[key] || "#6B7585";
        const pts = data.map((d, i) => `${(i / (data.length - 1)) * w},${h - ((d[key] as number) / max) * (h - 8) - 4}`).join(" ");
        const area = pts + ` ${w},${h} 0,${h}`;
        const gid = `ac-${key}`;
        return (
          <g key={key}>
            <defs>
              <linearGradient id={gid} x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor={color} stopOpacity="0.12" />
                <stop offset="100%" stopColor={color} stopOpacity="0" />
              </linearGradient>
            </defs>
            <polygon points={area} fill={`url(#${gid})`} />
            <polyline points={pts} fill="none" stroke={color} strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
          </g>
        );
      })}
    </svg>
  );
}

interface RequestRateChartProps {
  timeRange: TimeRangeId;
}

export function RequestRateChart({ timeRange }: RequestRateChartProps) {
  const colors = Object.fromEntries(RPS_CHART_LEGEND.map((l) => [l.key, l.color]));

  return (
    <section className="rounded-lg border border-border bg-card overflow-hidden">
      <div className="px-4 py-3 border-b border-border flex items-center justify-between">
        <div className="flex items-center gap-2">
          <span className="text-primary text-xs">&#x25C9;</span>
          <span className="text-[13px] font-semibold text-foreground font-display">Request Rate</span>
        </div>
        <div className="flex gap-3 text-[9px]">
          {RPS_CHART_LEGEND.map((l) => (
            <div key={l.key} className="flex items-center gap-1">
              <div className="w-2 h-[3px] rounded-sm" style={{ background: l.color }} />
              <span className="text-muted-foreground font-display">{l.label}</span>
            </div>
          ))}
        </div>
      </div>
      <div className="px-4 pt-3 pb-2">
        <AreaChart data={COMBINED_RPS} colors={colors} keys={RPS_CHART_LEGEND.map((l) => l.key)} />
        <div className="flex justify-between text-[9px] text-muted-foreground mt-1.5 font-display">
          <span>-{timeRange}</span>
          <span>now</span>
        </div>
      </div>
    </section>
  );
}
