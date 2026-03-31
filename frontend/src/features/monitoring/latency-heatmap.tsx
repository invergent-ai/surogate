// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: AGPL-3.0-only
//
import { LATENCY_BUCKETS, LATENCY_HEATMAP } from "./monitoring-data";

export function LatencyHeatmap() {
  return (
    <section className="rounded-lg border border-border bg-card overflow-hidden">
      <div className="px-4 py-3 border-b border-border flex items-center gap-2">
        <span className="text-primary">&#x229E;</span>
        <span className="text-xs font-semibold text-foreground font-display">Latency Distribution</span>
      </div>
      <div className="px-3 pt-2.5 pb-2">
        <div className="flex gap-px">
          <div className="w-11 flex flex-col gap-px mr-1">
            {LATENCY_BUCKETS.map((b) => (
              <div key={b} className="h-3.5 flex items-center justify-end text-[7px] text-muted-foreground font-display pr-1">
                {b}
              </div>
            ))}
          </div>
          <div className="flex-1 flex flex-col gap-px">
            {LATENCY_HEATMAP.map((row, ri) => (
              <div key={ri} className="flex gap-px">
                {row.map((v, ci) => (
                  <div
                    key={ci}
                    className="flex-1 h-3.5 rounded-[1px]"
                    style={{
                      background:
                        v > 0.01
                          ? `rgba(20, ${Math.round(130 + v * 54)}, ${Math.round(120 + v * 46)}, ${0.15 + v * 0.6})`
                          : "var(--muted)",
                    }}
                  />
                ))}
              </div>
            ))}
          </div>
        </div>
        <div className="flex justify-between text-[8px] text-muted-foreground mt-1 pl-12 font-display">
          <span>-24h</span>
          <span>-12h</span>
          <span>now</span>
        </div>
      </div>
    </section>
  );
}
