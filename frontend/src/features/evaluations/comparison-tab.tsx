// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: AGPL-3.0-only
//
import { BENCHMARKS } from "./evaluations-data";
import type { EvalRun } from "./evaluations-data";

export function ComparisonTab({ run }: { run: EvalRun }) {
  if (!run.compareLabel) {
    return (
      <div className="py-10 text-center text-muted-foreground/30 text-xs">
        No comparison model specified for this run. Select an eval with a
        comparison to see side-by-side results.
      </div>
    );
  }

  const wins = Object.values(run.scores).filter(
    (s) => s.delta !== null && s.delta > 0,
  ).length;
  const losses = Object.values(run.scores).filter(
    (s) => s.delta !== null && s.delta < 0,
  ).length;
  const ties = Object.values(run.scores).filter(
    (s) => s.delta !== null && s.delta === 0,
  ).length;

  return (
    <div className="animate-in fade-in duration-150">
      <section className="bg-muted/40 border border-border rounded-lg overflow-hidden">
        <div className="px-4 py-3 border-b border-border">
          <span className="text-[13px] font-semibold text-foreground font-display">
            {run.modelLabel} vs {run.compareLabel}
          </span>
        </div>
        <table className="w-full border-collapse">
          <thead>
            <tr className="border-b border-border">
              <th className="px-4 py-2 text-left text-[9px] font-medium text-muted-foreground/40 uppercase tracking-wide font-display">
                Benchmark
              </th>
              <th className="px-4 py-2 text-right text-[9px] font-medium text-violet-500 tracking-wide font-display">
                {run.modelLabel}
              </th>
              <th className="px-4 py-2 text-right text-[9px] font-medium text-muted-foreground tracking-wide font-display">
                {run.compareLabel}
              </th>
              <th className="px-4 py-2 text-right text-[9px] font-medium text-muted-foreground/40 uppercase tracking-wide font-display">
                Delta
              </th>
              <th className="px-4 py-2 text-center text-[9px] font-medium text-muted-foreground/40 uppercase tracking-wide font-display">
                Winner
              </th>
            </tr>
          </thead>
          <tbody>
            {run.benchmarks.map((bid) => {
              const s = run.scores[bid];
              if (!s || s.previous === null || s.delta === null) return null;
              const b = BENCHMARKS.find((bk) => bk.id === bid);
              const winner =
                s.delta > 0 ? "new" : s.delta < 0 ? "old" : "tie";
              return (
                <tr key={bid} className="border-b border-border/50">
                  <td className="px-4 py-2.5 text-xs text-foreground font-medium">
                    {b?.name || bid}
                  </td>
                  <td className="px-4 py-2.5 text-sm text-violet-500 font-bold text-right">
                    {s.value}
                  </td>
                  <td className="px-4 py-2.5 text-sm text-muted-foreground font-medium text-right">
                    {s.previous}
                  </td>
                  <td
                    className="px-4 py-2.5 text-xs font-semibold text-right"
                    style={{
                      color: s.delta > 0 ? "#22C55E" : "#EF4444",
                    }}
                  >
                    {s.delta > 0 ? "+" : ""}
                    {s.delta.toFixed(1)}
                  </td>
                  <td className="px-4 py-2.5 text-center">
                    <span
                      className="text-sm"
                      style={{
                        color:
                          winner === "new"
                            ? "#22C55E"
                            : winner === "old"
                              ? "#EF4444"
                              : "#6B7585",
                      }}
                    >
                      {winner === "new"
                        ? "\u2713"
                        : winner === "old"
                          ? "\u2717"
                          : "="}
                    </span>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
        <div className="px-4 py-3 border-t border-border flex justify-between items-center">
          <span className="text-[10px] text-muted-foreground">
            Won:{" "}
            <span className="text-green-500 font-semibold">{wins}</span>
            {" \u00B7 "}Lost:{" "}
            <span className="text-red-500 font-semibold">{losses}</span>
            {" \u00B7 "}Tied:{" "}
            <span className="text-muted-foreground">{ties}</span>
          </span>
          <span
            className="text-[10px] font-semibold"
            style={{ color: wins >= losses ? "#22C55E" : "#EF4444" }}
          >
            {wins >= losses
              ? "\u2713 New model wins overall"
              : "\u2717 Previous model wins overall"}
          </span>
        </div>
      </section>
    </div>
  );
}
