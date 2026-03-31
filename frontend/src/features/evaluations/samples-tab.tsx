// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: AGPL-3.0-only
//
import { useState } from "react";
import { cn } from "@/utils/cn";
import { BENCHMARKS, CATEGORY_COLORS } from "./evaluations-data";
import type { EvalRun, BenchmarkCategory } from "./evaluations-data";

type SampleFilter = "all" | "correct" | "incorrect";

export function SamplesTab({ run }: { run: EvalRun }) {
  const [filter, setFilter] = useState<SampleFilter>("all");

  if (run.samples.length === 0) {
    return (
      <div className="py-10 text-center text-muted-foreground/30 text-xs">
        No sample-level data available for this run. Re-run with sample logging
        enabled.
      </div>
    );
  }

  const filteredSamples = run.samples.filter((s) => {
    if (filter === "correct") return s.correct;
    if (filter === "incorrect") return !s.correct;
    return true;
  });

  return (
    <div className="animate-in fade-in duration-150">
      <section className="bg-muted/40 border border-border rounded-lg overflow-hidden">
        <div className="px-4 py-3 border-b border-border flex items-center justify-between">
          <span className="text-[13px] font-semibold text-foreground font-display">
            Sample Inspection
          </span>
          <div className="flex gap-1">
            {(["all", "correct", "incorrect"] as const).map((f) => (
              <button
                key={f}
                onClick={() => setFilter(f)}
                className={cn(
                  "px-2 py-0.5 rounded text-[9px] font-medium font-display border transition-colors cursor-pointer capitalize",
                  filter === f
                    ? "border-border bg-muted text-foreground/70"
                    : "border-transparent text-muted-foreground hover:bg-muted/50",
                )}
              >
                {f}
              </button>
            ))}
          </div>
        </div>
        {filteredSamples.map((s) => {
          const b = BENCHMARKS.find((bk) => bk.id === s.benchmark);
          const cc =
            CATEGORY_COLORS[b?.category as BenchmarkCategory];
          return (
            <div
              key={s.id}
              className="px-4 py-3.5 border-b border-border/50 last:border-b-0"
            >
              <div className="flex items-center gap-2 mb-1.5">
                <span
                  className="w-[18px] h-[18px] rounded flex items-center justify-center text-[10px] font-bold border"
                  style={{
                    background: s.correct ? "#22C55E12" : "#EF444412",
                    color: s.correct ? "#22C55E" : "#EF4444",
                    borderColor: s.correct ? "#22C55E30" : "#EF444430",
                  }}
                >
                  {s.correct ? "\u2713" : "\u2717"}
                </span>
                <code className="text-[10px] text-muted-foreground">
                  {s.id}
                </code>
                {cc && (
                  <span
                    className="text-[8px] px-1.5 py-px rounded"
                    style={{ background: cc.bg, color: cc.fg }}
                  >
                    {b?.name}
                  </span>
                )}
              </div>
              <div className="bg-background border border-border rounded-md px-3 py-2.5 mb-1.5">
                <div className="text-[8px] text-muted-foreground/40 uppercase tracking-wide mb-1 font-display">
                  Input
                </div>
                <div className="text-[11px] text-foreground/70 leading-snug">
                  {s.input}
                </div>
              </div>
              <div className="grid grid-cols-2 gap-2">
                <div className="bg-background border border-border rounded-md px-3 py-2.5">
                  <div className="text-[8px] text-green-500 uppercase tracking-wide mb-1 font-display">
                    Expected
                  </div>
                  <div className="text-[11px] text-foreground/70 leading-snug">
                    {s.expected}
                  </div>
                </div>
                <div
                  className="bg-background rounded-md px-3 py-2.5 border"
                  style={{
                    borderColor: s.correct ? undefined : "#EF444420",
                  }}
                >
                  <div
                    className="text-[8px] uppercase tracking-wide mb-1 font-display"
                    style={{ color: s.correct ? "#3B82F6" : "#EF4444" }}
                  >
                    Predicted
                  </div>
                  <div className="text-[11px] text-foreground/70 leading-snug">
                    {s.predicted}
                  </div>
                </div>
              </div>
            </div>
          );
        })}
      </section>
    </div>
  );
}
