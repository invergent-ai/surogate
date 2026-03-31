// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: AGPL-3.0-only
//
import { BENCHMARKS, CATEGORY_COLORS } from "./evaluations-data";
import type { EvalRun, BenchmarkCategory } from "./evaluations-data";

export function DetailsTab({ run }: { run: EvalRun }) {
  return (
    <div className="animate-in fade-in duration-150">
      <div className="grid grid-cols-2 gap-4">
        {/* Run Configuration */}
        <section className="bg-muted/40 border border-border rounded-lg p-3.5">
          <div className="text-xs font-semibold text-foreground font-display mb-3">
            Run Configuration
          </div>
          {[
            { label: "Run ID", value: run.id },
            { label: "Model", value: run.modelLabel },
            { label: "Compare", value: run.compareLabel || "\u2014" },
            { label: "Benchmarks", value: run.benchmarks.length },
            { label: "Status", value: run.status },
            { label: "Started", value: run.startedAt },
            { label: "Duration", value: run.duration || "in progress" },
            { label: "Runner", value: run.runner },
            {
              label: "Compute",
              value: `${run.compute === "aws" ? "AWS" : "Local"} \u00B7 ${run.gpu}`,
            },
          ].map((f) => (
            <div
              key={f.label}
              className="flex justify-between py-1 border-b border-border/50"
            >
              <span className="text-[10px] text-muted-foreground/40">
                {f.label}
              </span>
              <span className="text-[10px] text-foreground/70">{f.value}</span>
            </div>
          ))}
        </section>

        {/* Benchmarks in Run */}
        <section className="bg-muted/40 border border-border rounded-lg p-3.5">
          <div className="text-xs font-semibold text-foreground font-display mb-3">
            Benchmarks in Run
          </div>
          {run.benchmarks.map((bid) => {
            const b = BENCHMARKS.find((bk) => bk.id === bid);
            const cc =
              CATEGORY_COLORS[b?.category as BenchmarkCategory] ||
              CATEGORY_COLORS.custom;
            const s = run.scores[bid];
            return (
              <div
                key={bid}
                className="flex items-center justify-between py-1.5 border-b border-border/50"
              >
                <div className="flex items-center gap-1.5">
                  <span className="text-xs" style={{ color: cc.fg }}>
                    {b?.icon}
                  </span>
                  <span className="text-[11px] text-foreground font-medium">
                    {b?.name}
                  </span>
                  <span className="text-[9px] text-muted-foreground/40">
                    {b?.samples} samples
                  </span>
                </div>
                <div className="flex items-center gap-1.5">
                  {s ? (
                    <span className="text-xs text-violet-500 font-bold">
                      {s.value}
                    </span>
                  ) : (
                    <span className="text-[9px] text-amber-500">pending</span>
                  )}
                </div>
              </div>
            );
          })}
        </section>
      </div>
    </div>
  );
}
