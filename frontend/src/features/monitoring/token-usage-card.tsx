// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: AGPL-3.0-only
//
import { TOKEN_USAGE } from "./monitoring-data";

export function TokenUsageCard() {
  return (
    <section className="rounded-lg border border-border bg-card overflow-hidden">
      <div className="px-4 py-3 border-b border-border flex items-center gap-2">
        <span style={{ color: "#F59E0B" }}>&#x25A4;</span>
        <span className="text-xs font-semibold text-foreground font-display">Token Usage (24h)</span>
      </div>
      <div className="px-4 pt-3.5 pb-2">
        {/* stacked bar */}
        <div className="flex h-3 rounded-md overflow-hidden mb-3 gap-px">
          {TOKEN_USAGE.map((t) => (
            <div
              key={t.agent}
              className="transition-[width] duration-500 ease-out"
              style={{ width: `${t.pctOfTotal}%`, background: t.color, minWidth: t.pctOfTotal > 2 ? undefined : 4 }}
            />
          ))}
        </div>

        {/* breakdown rows */}
        {TOKEN_USAGE.map((t) => (
          <div key={t.agent} className="flex items-center justify-between py-1.5 border-b border-border/50">
            <div className="flex items-center gap-2">
              <div className="w-2 h-2 rounded-sm shrink-0" style={{ background: t.color }} />
              <span className="text-[11px] text-foreground/80 font-medium">{t.agent}</span>
            </div>
            <div className="flex items-center gap-3 text-[10px]">
              <span className="text-muted-foreground">
                <span className="text-green-500">&#x2193;</span>
                {(t.input / 1000).toFixed(0)}K{" "}
                <span className="text-blue-500">&#x2191;</span>
                {(t.output / 1000).toFixed(0)}K
              </span>
              <span className="text-foreground/60 font-medium w-10 text-right">{t.pctOfTotal}%</span>
            </div>
          </div>
        ))}

        <div className="flex justify-between mt-2.5 text-[10px] font-display">
          <span className="text-muted-foreground">Total</span>
          <span className="text-foreground font-bold">14.0M</span>
        </div>
      </div>
    </section>
  );
}
