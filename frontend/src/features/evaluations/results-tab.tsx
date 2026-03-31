// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: AGPL-3.0-only
//
import { BENCHMARKS, CATEGORY_COLORS } from "./evaluations-data";
import { RadarChart } from "./radar-chart";
import type { EvalRun, BenchmarkCategory } from "./evaluations-data";

// ── Score bar ─────────────────────────────────────────────────

function ScoreBar({
  value,
  max = 100,
  color,
  compare,
}: {
  value: number;
  max?: number;
  color: string;
  compare?: number | null;
}) {
  const pct = Math.min((value / max) * 100, 100);
  const cPct =
    compare != null ? Math.min((compare / max) * 100, 100) : null;

  return (
    <div className="relative h-1.5 bg-muted rounded-sm overflow-hidden">
      {cPct !== null && (
        <div
          className="absolute h-full bg-foreground/5 rounded-sm"
          style={{ width: `${cPct}%` }}
        />
      )}
      <div
        className="h-full rounded-sm transition-[width] duration-500 ease-out"
        style={{ width: `${pct}%`, backgroundColor: color }}
      />
    </div>
  );
}

// ── Results tab ───────────────────────────────────────────────

export function ResultsTab({ run }: { run: EvalRun }) {
  const radarLabels = run.benchmarks.map((bid) => {
    const b = BENCHMARKS.find((bk) => bk.id === bid);
    return { id: bid, short: b?.name || bid };
  });

  return (
    <div className="animate-in fade-in duration-150">
      <div className="grid grid-cols-[280px_1fr] gap-5">
        {/* Radar chart */}
        <div className="bg-muted/40 border border-border rounded-lg p-4 flex flex-col items-center">
          <div className="text-xs font-semibold text-foreground font-display mb-2 self-start">
            Radar Overview
          </div>
          <RadarChart
            scores={run.scores}
            compareScores={
              run.compareModel
                ? Object.fromEntries(
                    Object.entries(run.scores).map(([k, v]) => [
                      k,
                      { value: v.previous },
                    ]),
                  )
                : null
            }
            labels={radarLabels}
            size={240}
          />
          {run.compareLabel && (
            <div className="flex gap-3 mt-2 text-[9px] text-muted-foreground">
              <span className="flex items-center gap-1">
                <span className="w-2 h-0.5 bg-violet-500 rounded-sm" />
                {run.modelLabel}
              </span>
              <span className="flex items-center gap-1">
                <span className="w-2 h-0.5 bg-muted-foreground rounded-sm" />
                {run.compareLabel}
              </span>
            </div>
          )}
        </div>

        {/* Score bars */}
        <div className="flex flex-col gap-2.5">
          {run.benchmarks.map((bid) => {
            const b = BENCHMARKS.find((bk) => bk.id === bid);
            const s = run.scores[bid];
            if (!s) return null;
            const cc =
              CATEGORY_COLORS[b?.category as BenchmarkCategory] ||
              CATEGORY_COLORS.custom;
            const max =
              b?.metric === "score/10"
                ? 10
                : b?.metric === "score/5"
                  ? 5
                  : 100;

            return (
              <div
                key={bid}
                className="bg-muted/40 border border-border rounded-lg px-4 py-3"
              >
                <div className="flex items-center justify-between mb-1.5">
                  <div className="flex items-center gap-2">
                    <span className="text-sm" style={{ color: cc.fg }}>
                      {b?.icon}
                    </span>
                    <span className="text-xs font-semibold text-foreground font-display">
                      {b?.name || bid}
                    </span>
                    <span
                      className="text-[8px] px-1.5 py-px rounded font-medium uppercase tracking-wide"
                      style={{
                        background: cc.bg,
                        color: cc.fg,
                        border: `1px solid ${cc.border}`,
                      }}
                    >
                      {b?.category}
                    </span>
                  </div>
                  <div className="flex items-baseline gap-1.5">
                    <span className="text-xl font-bold text-foreground tracking-tight">
                      {s.value}
                    </span>
                    <span className="text-[10px] text-muted-foreground">
                      {b?.metric === "score/10"
                        ? "/10"
                        : b?.metric === "score/5"
                          ? "/5"
                          : "%"}
                    </span>
                    {s.delta !== null && (
                      <span
                        className="text-[10px] font-semibold"
                        style={{
                          color:
                            s.delta > 0
                              ? "#22C55E"
                              : s.delta < 0
                                ? "#EF4444"
                                : undefined,
                        }}
                      >
                        {s.delta > 0 ? "+" : ""}
                        {s.delta.toFixed(1)}
                      </span>
                    )}
                  </div>
                </div>
                <ScoreBar
                  value={s.value}
                  max={max}
                  color={cc.fg}
                  compare={s.previous}
                />
                {s.previous !== null && (
                  <div className="text-[9px] text-muted-foreground/40 mt-1 text-right">
                    previous: {s.previous}
                    {b?.metric === "score/10"
                      ? "/10"
                      : b?.metric === "score/5"
                        ? "/5"
                        : "%"}
                  </div>
                )}
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}
