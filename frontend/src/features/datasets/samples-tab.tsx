// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: AGPL-3.0-only
//
import { useState } from "react";
import { Button } from "@/components/ui/button";
import { cn } from "@/utils/cn";
import type { Dataset } from "./datasets-data";

export function SamplesTab({ dataset }: { dataset: Dataset }) {
  const [expandedId, setExpandedId] = useState<string | null>(null);

  if (dataset.sampleData.length === 0) {
    return (
      <div className="animate-in fade-in duration-150">
        <div className="py-10 text-center text-muted-foreground/40">
          <div className="text-xs mb-2">No sample preview available for this dataset</div>
          <Button variant="outline" size="xs">Load Samples</Button>
        </div>
      </div>
    );
  }

  return (
    <div className="animate-in fade-in duration-150 flex flex-col gap-3">
      {dataset.sampleData.map((s) => {
        const isExpanded = expandedId === s.id;
        return (
          <section
            key={s.id}
            className={cn(
              "bg-muted/40 border border-border rounded-lg overflow-hidden",
              isExpanded && "border-l-2",
            )}
            style={isExpanded ? { borderLeftColor: dataset.color } : undefined}
          >
            {/* Sample header */}
            <button
              onClick={() => setExpandedId(isExpanded ? null : s.id)}
              className="w-full text-left px-4 py-3 flex items-center justify-between hover:bg-muted/30 transition-colors cursor-pointer"
            >
              <div className="flex items-center gap-2">
                <code className="text-[10px] text-muted-foreground">{s.id}</code>
                <div className="flex gap-1">
                  {s.metadata.tags.map((t) => (
                    <span key={t} className="text-[7px] px-1 py-px rounded bg-muted text-muted-foreground">
                      {t}
                    </span>
                  ))}
                </div>
                {s.metadata.quality_score != null && (
                  <span className={cn(
                    "text-[9px]",
                    s.metadata.quality_score > 0.9 ? "text-green-500"
                      : s.metadata.quality_score > 0.8 ? "text-amber-500"
                      : "text-red-500",
                  )}>
                    &#x2605; {s.metadata.quality_score}
                  </span>
                )}
              </div>
              <div className="flex items-center gap-2 text-[9px] text-muted-foreground/40">
                <span>{s.metadata.tokens} tok</span>
                <span>{s.metadata.turns}t</span>
                <span className={cn(
                  "text-xs text-muted-foreground/20 transition-transform duration-150",
                  isExpanded && "rotate-180",
                )}>
                  &#x25BE;
                </span>
              </div>
            </button>

            {/* Expanded content */}
            {isExpanded && (
              <div className="px-4 pb-4 animate-in fade-in duration-150">
                <div className="bg-background border border-border rounded-md px-3.5 py-3 mb-2">
                  <div className="text-[8px] text-amber-500 uppercase tracking-widest mb-1 font-display font-semibold">
                    {dataset.format === "DPO" ? "Prompt" : "Instruction"}
                  </div>
                  <div className="text-[11px] text-foreground/70 leading-relaxed whitespace-pre-wrap">
                    {s.instruction}
                  </div>
                </div>
                <div className="bg-background border border-border rounded-md px-3.5 py-3 mb-2">
                  <div className="text-[8px] text-green-500 uppercase tracking-widest mb-1 font-display font-semibold">
                    {dataset.format === "DPO" ? "Chosen Response" : "Response"}
                  </div>
                  <div className="text-[11px] text-foreground/70 leading-relaxed whitespace-pre-wrap">
                    {s.response}
                  </div>
                </div>
                <div className="flex gap-1.5">
                  <Button variant="outline" size="xs">Edit</Button>
                  <Button variant="outline" size="xs" className="text-destructive border-destructive/30 hover:bg-destructive/10">Remove</Button>
                  <Button variant="outline" size="xs">Duplicate</Button>
                </div>
              </div>
            )}
          </section>
        );
      })}
    </div>
  );
}
