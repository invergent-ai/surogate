// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: AGPL-3.0-only
//
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { cn } from "@/utils/cn";
import type { Dataset } from "./datasets-data";

export function VersionsTab({ dataset }: { dataset: Dataset }) {
  return (
    <div className="animate-in fade-in duration-150">
      <section className="bg-muted/40 border border-border rounded-lg overflow-hidden">
        <div className="px-4 py-3 border-b border-border flex items-center gap-2">
          <span className="text-[13px] font-semibold text-foreground font-display">
            Version History
          </span>
          <span className="text-[9px] px-1.5 py-px bg-muted rounded text-muted-foreground">
            {dataset.versions.length} versions
          </span>
        </div>
        {dataset.versions.map((v, i) => (
          <div
            key={v.version}
            className="px-4 py-3 border-b border-border/50 last:border-b-0 flex items-start gap-3 hover:bg-muted/30 transition-colors cursor-pointer"
          >
            {/* Timeline */}
            <div className="flex flex-col items-center pt-1 w-4 shrink-0">
              <div
                className={cn(
                  "w-2.5 h-2.5 rounded-full shrink-0 border-2",
                  i === 0
                    ? "bg-green-500 border-green-500"
                    : "bg-muted border-muted-foreground/30",
                )}
              />
              {i < dataset.versions.length - 1 && (
                <div className="w-px flex-1 bg-border min-h-[20px] mt-1" />
              )}
            </div>

            <div className="flex-1">
              <div className="flex items-center gap-2 mb-0.5">
                <code
                  className={cn(
                    "text-xs",
                    i === 0 ? "text-foreground font-semibold" : "text-muted-foreground",
                  )}
                >
                  {v.version}
                </code>
                {i === 0 && <Badge variant="active">CURRENT</Badge>}
                <span className="text-[10px] text-muted-foreground">
                  {v.samples.toLocaleString()} samples
                </span>
              </div>
              <div className="text-[11px] text-foreground/70 mb-0.5">{v.change}</div>
              <div className="text-[9px] text-muted-foreground/40">
                {v.author} &middot; {v.date}
              </div>
            </div>

            {i > 0 && (
              <Button variant="outline" size="xs" className="shrink-0">
                Restore
              </Button>
            )}
          </div>
        ))}
      </section>
    </div>
  );
}
