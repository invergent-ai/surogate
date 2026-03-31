// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: AGPL-3.0-only
//
import { useState, useMemo } from "react";
import { Button } from "@/components/ui/button";
import { cn } from "@/utils/cn";
import { BENCHMARKS, CATEGORY_COLORS } from "./evaluations-data";
import type { BenchmarkCategory } from "./evaluations-data";

export function BenchmarkSuite() {
  const [filter, setFilter] = useState("all");
  const categories = useMemo(
    () => [...new Set(BENCHMARKS.map((b) => b.category))],
    [],
  );

  const filtered =
    filter === "all"
      ? BENCHMARKS
      : BENCHMARKS.filter((b) => b.category === filter);

  return (
    <div className="flex-1 overflow-y-auto px-7 py-5">
      <div className="flex items-center justify-between mb-4">
        <div className="flex gap-1">
          {["all", ...categories].map((c) => {
            const cc = CATEGORY_COLORS[c as BenchmarkCategory];
            const isActive = filter === c;
            return (
              <button
                key={c}
                onClick={() => setFilter(c)}
                className={cn(
                  "px-2.5 py-1 rounded text-[10px] font-medium font-display border transition-colors cursor-pointer capitalize",
                  !isActive &&
                    "border-transparent text-muted-foreground hover:bg-muted/50",
                )}
                style={
                  isActive
                    ? {
                        borderColor: (cc?.fg || "#8B5CF6") + "33",
                        color: cc?.fg || "#8B5CF6",
                        background: (cc?.fg || "#8B5CF6") + "10",
                      }
                    : undefined
                }
              >
                {c}
              </button>
            );
          })}
        </div>
        <Button variant="outline" size="sm">
          + Custom Benchmark
        </Button>
      </div>

      <div className="grid grid-cols-3 gap-3">
        {filtered.map((b) => {
          const cc = CATEGORY_COLORS[b.category];
          return (
            <div
              key={b.id}
              className="bg-muted/40 border border-border rounded-lg p-4 cursor-pointer hover:bg-muted/60 transition-colors"
            >
              <div className="flex items-center gap-2 mb-2">
                <span className="text-xl" style={{ color: cc.fg }}>
                  {b.icon}
                </span>
                <div>
                  <div className="text-[13px] font-bold text-foreground font-display">
                    {b.name}
                  </div>
                  <span
                    className="text-[8px] px-1.5 py-px rounded font-medium uppercase tracking-wide"
                    style={{
                      background: cc.bg,
                      color: cc.fg,
                      border: `1px solid ${cc.border}`,
                    }}
                  >
                    {b.category}
                  </span>
                </div>
              </div>
              <p className="text-[10px] text-muted-foreground leading-snug mb-2.5">
                {b.description}
              </p>
              <div className="flex justify-between text-[9px] text-muted-foreground/40">
                <span>{b.samples.toLocaleString()} samples</span>
                <span>
                  metric:{" "}
                  <span className="text-muted-foreground">{b.metric}</span>
                </span>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
