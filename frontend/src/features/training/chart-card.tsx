// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: AGPL-3.0-only
//
import { cn } from "@/utils/cn";

interface Dataset {
  data: number[];
  color: string;
  opacity?: number;
  dashed?: boolean;
}

interface ChartSVGProps {
  datasets: Dataset[];
  h?: number;
  xLabel?: string;
  yLabel?: string;
  gridLines?: number;
}

export function ChartSVG({ datasets, h = 100, xLabel, yLabel, gridLines = 4 }: ChartSVGProps) {
  const svgW = 560;
  const validDatasets = datasets.filter(ds => ds.data && ds.data.length > 0);

  if (validDatasets.length === 0) {
    return (
      <div className="flex items-center justify-center text-[10px] text-muted-foreground/40" style={{ height: h }}>
        No data yet
      </div>
    );
  }

  const allVals = validDatasets.flatMap(ds => ds.data);
  const max = Math.max(...allVals);
  const min = Math.min(...allVals);
  const range = max - min || 1;
  const pad = range * 0.05;

  return (
    <div className="relative">
      <svg width="100%" height={h} viewBox={`0 0 ${svgW} ${h}`} preserveAspectRatio="none" className="block">
        {Array.from({ length: gridLines }).map((_, i) => {
          const y = (i + 1) * (h / (gridLines + 1));
          return <line key={i} x1="0" y1={y} x2={svgW} y2={y} stroke="currentColor" strokeWidth="0.5" className="text-border/50" />;
        })}
        {validDatasets.map((ds, di) => {
          const pts = ds.data.map((v, i) => {
            const x = (i / (ds.data.length - 1)) * svgW;
            const y = h - ((v - min + pad) / (range + pad * 2)) * (h - 8) - 4;
            return `${x},${y}`;
          }).join(" ");
          const gid = `cg${di}${ds.color.replace("#", "")}`;
          const area = pts + ` ${svgW},${h} 0,${h}`;
          const lastPt = pts.split(" ").pop()!.split(",");

          return (
            <g key={di}>
              {validDatasets.length === 1 && (
                <>
                  <defs>
                    <linearGradient id={gid} x1="0" y1="0" x2="0" y2="1">
                      <stop offset="0%" stopColor={ds.color} stopOpacity="0.1" />
                      <stop offset="100%" stopColor={ds.color} stopOpacity="0" />
                    </linearGradient>
                  </defs>
                  <polygon points={area} fill={`url(#${gid})`} />
                </>
              )}
              <polyline
                points={pts}
                fill="none"
                stroke={ds.color}
                strokeWidth={validDatasets.length > 1 ? "1.5" : "2"}
                strokeLinecap="round"
                strokeLinejoin="round"
                opacity={ds.opacity || 1}
                strokeDasharray={ds.dashed ? "4,3" : "none"}
              />
              {validDatasets.length <= 2 && (
                <circle cx={parseFloat(lastPt[0])} cy={parseFloat(lastPt[1])} r="3" fill={ds.color} />
              )}
            </g>
          );
        })}
      </svg>
      <div className="flex justify-between text-[9px] text-muted-foreground/40 mt-1">
        <span>{xLabel || "Step 0"}</span>
        <span>{yLabel || ""}</span>
      </div>
    </div>
  );
}

interface ChartCardProps {
  title: string;
  value?: string;
  valueColor?: string;
  datasets: Dataset[];
  h?: number;
  xLabel?: string;
  className?: string;
}

export function ChartCard({ title, value, valueColor, datasets, h = 100, xLabel, className }: ChartCardProps) {
  return (
    <section className={cn("bg-card rounded-lg ring-1 ring-foreground/10 overflow-hidden", className)}>
      <div className="px-4 py-3 border-b border-border flex justify-between items-center">
        <span className="text-xs font-semibold text-foreground font-display">{title}</span>
        {value && (
          <span className="text-[13px] font-bold" style={{ color: valueColor }}>
            {value}
          </span>
        )}
      </div>
      <div className="px-4 py-3">
        <ChartSVG datasets={datasets} h={h} xLabel={xLabel} />
      </div>
    </section>
  );
}
