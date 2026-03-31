// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: AGPL-3.0-only
//
import { StatusDot } from "@/components/ui/status-dot";
import { SOURCE_LABELS, toStatus } from "./datasets-data";
import type { Dataset } from "./datasets-data";

function MiniHistogram({ data, color, width = 500, height = 60 }: { data: number[]; color: string; width?: number; height?: number }) {
  const max = Math.max(...data);
  const barW = width / data.length - 1;
  return (
    <svg width="100%" height={height} viewBox={`0 0 ${width} ${height}`} preserveAspectRatio="none" className="block">
      {data.map((v, i) => {
        const bh = max > 0 ? (v / max) * (height - 2) : 0;
        return <rect key={i} x={i * (barW + 1)} y={height - bh - 1} width={barW} height={bh} rx="1" fill={color} fillOpacity="0.6" />;
      })}
    </svg>
  );
}

export function OverviewTab({ dataset }: { dataset: Dataset }) {
  return (
    <div className="animate-in fade-in duration-150">
      {/* KPI strip */}
      <div className="grid grid-cols-5 gap-2.5 mb-5">
        {[
          { label: "Samples", value: dataset.samples.toLocaleString(), color: "text-foreground" },
          { label: "Total Tokens", value: dataset.tokens, color: "text-violet-500" },
          { label: "Avg Tokens", value: dataset.stats.avgTokensPerSample.toLocaleString(), color: "text-blue-500" },
          { label: "Avg Turns", value: String(dataset.stats.avgTurns), color: "text-green-500" },
          { label: "Size", value: dataset.size, color: "text-amber-500" },
        ].map((m) => (
          <div key={m.label} className="bg-muted/40 border border-border rounded-lg px-3.5 py-3">
            <div className="text-[9px] text-muted-foreground/50 uppercase tracking-wide mb-1 font-display">{m.label}</div>
            <div className={`text-xl font-bold tracking-tight ${m.color}`}>{m.value}</div>
          </div>
        ))}
      </div>

      <div className="grid grid-cols-[1fr_300px] gap-4">
        {/* Left column */}
        <div className="flex flex-col gap-4">
          {/* Token distribution */}
          <section className="bg-muted/40 border border-border rounded-lg p-4">
            <div className="text-xs font-semibold text-foreground font-display mb-3">
              Token Distribution
            </div>
            <MiniHistogram data={dataset.stats.tokenHistogram} color={dataset.color} />
            <div className="flex justify-between text-[9px] text-muted-foreground/40 mt-1.5 font-display">
              <span>min: {dataset.stats.minTokens}</span>
              <span>avg: {dataset.stats.avgTokensPerSample}</span>
              <span>max: {dataset.stats.maxTokens}</span>
            </div>
          </section>

          {/* Content distribution */}
          <section className="bg-muted/40 border border-border rounded-lg p-4">
            <div className="text-xs font-semibold text-foreground font-display mb-3">
              Content Distribution
            </div>
            {dataset.stats.topTags.map((t, i) => {
              const maxC = dataset.stats.topTags[0].count;
              return (
                <div key={t.tag} className="mb-2 last:mb-0">
                  <div className="flex justify-between text-[10px] mb-1 font-display">
                    <span className="text-foreground/70">{t.tag}</span>
                    <span className="text-muted-foreground">{t.count}</span>
                  </div>
                  <div className="h-1 bg-border rounded-sm overflow-hidden">
                    <div
                      className="h-full rounded-sm transition-[width] duration-400"
                      style={{ width: `${(t.count / maxC) * 100}%`, backgroundColor: dataset.color, opacity: 1 - i * 0.12 }}
                    />
                  </div>
                </div>
              );
            })}
          </section>
        </div>

        {/* Right column */}
        <div className="flex flex-col gap-4">
          {/* Sentiment split */}
          <section className="bg-muted/40 border border-border rounded-lg p-4">
            <div className="text-xs font-semibold text-foreground font-display mb-3">
              Sentiment Split
            </div>
            <div className="flex h-2 rounded-full overflow-hidden mb-2.5">
              <div className="bg-green-500 transition-[width] duration-400" style={{ width: `${dataset.stats.sentimentDist.positive}%` }} />
              <div className="bg-muted-foreground transition-[width] duration-400" style={{ width: `${dataset.stats.sentimentDist.neutral}%` }} />
              <div className="bg-red-500 transition-[width] duration-400" style={{ width: `${dataset.stats.sentimentDist.negative}%` }} />
            </div>
            {[
              { label: "Positive", value: dataset.stats.sentimentDist.positive, status: "running" as const },
              { label: "Neutral", value: dataset.stats.sentimentDist.neutral, status: "stopped" as const },
              { label: "Negative", value: dataset.stats.sentimentDist.negative, status: "error" as const },
            ].map((s) => (
              <div key={s.label} className="flex items-center justify-between mb-1 last:mb-0">
                <div className="flex items-center gap-1.5">
                  <StatusDot status={s.status} />
                  <span className="text-[10px] text-muted-foreground font-display">{s.label}</span>
                </div>
                <span className="text-[11px] text-foreground font-semibold">{s.value}%</span>
              </div>
            ))}
          </section>

          {/* Used by */}
          <section className="bg-muted/40 border border-border rounded-lg p-4">
            <div className="text-xs font-semibold text-foreground font-display mb-2.5">
              Used By
            </div>
            {dataset.usedBy.length === 0 ? (
              <div className="text-[10px] text-muted-foreground/40 text-center py-2">
                Not used by any training job
              </div>
            ) : (
              dataset.usedBy.map((u) => (
                <div key={u.name} className="flex items-center justify-between py-1.5 border-b border-border/50 last:border-b-0">
                  <div className="flex items-center gap-1.5">
                    <span className="text-[11px] text-violet-500">&#x25EC;</span>
                    <span className="text-[11px] text-foreground font-medium">{u.name}</span>
                  </div>
                  <span className="flex items-center gap-1 text-[9px]">
                    <StatusDot status={toStatus(u.status)} />
                    <span className={u.status === "running" ? "text-green-500" : u.status === "completed" ? "text-blue-500" : "text-muted-foreground"}>
                      {u.status}
                    </span>
                  </span>
                </div>
              ))
            )}
          </section>

          {/* Details */}
          <section className="bg-muted/40 border border-border rounded-lg p-4">
            <div className="text-xs font-semibold text-foreground font-display mb-2.5">
              Details
            </div>
            {[
              { label: "Format", value: dataset.format },
              { label: "Source", value: SOURCE_LABELS[dataset.source] ?? dataset.source },
              { label: "Created", value: dataset.createdAt },
              { label: "Updated", value: dataset.updatedAt },
              { label: "Published", value: dataset.published ? "Yes" : "No" },
            ].map((f) => (
              <div key={f.label} className="flex justify-between py-0.5 text-[10px]">
                <span className="text-muted-foreground/40">{f.label}</span>
                <span className="text-foreground/70">{f.value}</span>
              </div>
            ))}
          </section>
        </div>
      </div>
    </div>
  );
}
