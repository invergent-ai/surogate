// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: AGPL-3.0-only
//
import { useState, useMemo } from "react";
import { cn } from "@/utils/cn";
import { LEADERBOARD } from "./evaluations-data";

type MetricKey = "gsm8k" | "mmlu" | "humaneval" | "mtbench";

const METRICS: { id: MetricKey; label: string }[] = [
  { id: "gsm8k", label: "GSM8K" },
  { id: "mmlu", label: "MMLU" },
  { id: "humaneval", label: "HumanEval" },
  { id: "mtbench", label: "MT-Bench" },
];

const RANK_STYLES = [
  { bg: "#F59E0B18", fg: "#F59E0B", border: "#F59E0B30" },
  { bg: "#A0A8B810", fg: "#A0A8B8", border: "#A0A8B820" },
  { bg: "#CD7F3218", fg: "#CD7F32", border: "#CD7F3230" },
];

export function Leaderboard() {
  const [metric, setMetric] = useState<MetricKey>("gsm8k");

  const sorted = useMemo(
    () =>
      [...LEADERBOARD].sort((a, b) => {
        const av = a[metric] ?? -1;
        const bv = b[metric] ?? -1;
        return bv - av;
      }),
    [metric],
  );

  return (
    <div className="flex-1 overflow-y-auto px-7 py-5">
      <div className="flex items-center justify-between mb-4">
        <div className="text-sm font-bold text-foreground font-display">
          Model Leaderboard
        </div>
        <div className="flex gap-1">
          {METRICS.map((m) => (
            <button
              key={m.id}
              onClick={() => setMetric(m.id)}
              className={cn(
                "px-2.5 py-1 rounded text-[10px] font-medium font-display border transition-colors cursor-pointer",
                metric === m.id
                  ? "border-violet-500/20 bg-violet-500/10 text-violet-500"
                  : "border-transparent text-muted-foreground hover:bg-muted/50",
              )}
            >
              {m.label}
            </button>
          ))}
        </div>
      </div>

      <section className="bg-muted/40 border border-border rounded-lg overflow-hidden">
        <table className="w-full border-collapse">
          <thead>
            <tr className="border-b border-border">
              <th className="px-4 py-2.5 text-left text-[9px] font-medium text-muted-foreground/40 uppercase tracking-wide font-display w-10">
                #
              </th>
              <th className="px-4 py-2.5 text-left text-[9px] font-medium text-muted-foreground/40 uppercase tracking-wide font-display">
                Model
              </th>
              {METRICS.map((m) => (
                <th
                  key={m.id}
                  className="px-4 py-2.5 text-right text-[9px] font-medium text-muted-foreground/40 uppercase tracking-wide font-display"
                >
                  {m.label}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {sorted.map((m, i) => (
              <tr
                key={m.model}
                className="border-b border-border/50 hover:bg-muted/30 transition-colors cursor-pointer"
              >
                <td className="px-4 py-3">
                  <span
                    className={cn(
                      "w-[22px] h-[22px] rounded-[5px] inline-flex items-center justify-center text-[11px] font-bold border",
                      i >= 3 &&
                        "bg-muted border-border text-muted-foreground/40",
                    )}
                    style={
                      i < 3
                        ? {
                            background: RANK_STYLES[i].bg,
                            color: RANK_STYLES[i].fg,
                            borderColor: RANK_STYLES[i].border,
                          }
                        : undefined
                    }
                  >
                    {i + 1}
                  </span>
                </td>
                <td className="px-4 py-3">
                  <div className="flex items-center gap-2">
                    <div
                      className="w-1 h-5 rounded-sm shrink-0"
                      style={{ backgroundColor: m.color }}
                    />
                    <span className="text-xs font-semibold text-foreground font-display">
                      {m.label}
                    </span>
                  </div>
                </td>
                {METRICS.map((col) => (
                  <td
                    key={col.id}
                    className={cn(
                      "px-4 py-3 text-right text-[13px]",
                      m[col.id] !== null
                        ? col.id === metric
                          ? "font-semibold text-violet-500"
                          : "font-semibold text-foreground/70"
                        : "text-muted-foreground/30",
                    )}
                  >
                    {m[col.id] !== null ? m[col.id] : "\u2014"}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </section>
    </div>
  );
}
