// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: AGPL-3.0-only
//
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { cn } from "@/utils/cn";
import type { Model } from "./models-data";

export function FinetunesTab({ model }: { model: Model }) {
  return (
    <div className="animate-in fade-in duration-150">
      <section className="bg-muted/40 border border-border rounded-lg overflow-hidden">
        <div className="px-4 py-3 border-b border-border flex items-center justify-between">
          <div className="flex items-center gap-2">
            <span className="text-amber-500">&#x25EC;</span>
            <span className="text-[13px] font-semibold text-foreground font-display">
              Fine-tune Lineage
            </span>
            <span className="text-[9px] px-1.5 py-px bg-muted rounded text-muted-foreground">
              {model.fineTunes.length}
            </span>
          </div>
          <Button variant="outline" size="xs">
            + New Fine-tune
          </Button>
        </div>

        {model.fineTunes.length === 0 ? (
          <div className="py-8 text-center">
            <div className="text-muted-foreground/40 text-xs mb-1">
              No fine-tunes for this model
            </div>
            <div className="text-[10px] text-muted-foreground/30">
              This is a base model. Start a fine-tuning job to create a
              specialized variant.
            </div>
          </div>
        ) : (
          model.fineTunes.map((ft, i) => (
            <div
              key={ft.name}
              className="px-4 py-3 border-b border-border/50 last:border-b-0 flex items-start gap-3 hover:bg-muted/30 transition-colors cursor-pointer"
            >
              {/* Timeline dot */}
              <div className="flex flex-col items-center pt-1 w-4 shrink-0">
                <div
                  className={cn(
                    "w-2.5 h-2.5 rounded-full shrink-0 border-2",
                    ft.status === "active"
                      ? "bg-green-500 border-green-500"
                      : "bg-muted border-muted-foreground/30",
                  )}
                />
                {i < model.fineTunes.length - 1 && (
                  <div className="w-px flex-1 bg-border min-h-[20px] mt-1" />
                )}
              </div>

              <div className="flex-1">
                <div className="flex items-center gap-2 mb-0.5">
                  <code
                    className={cn(
                      "text-xs",
                      ft.status === "active"
                        ? "text-foreground font-semibold"
                        : "text-muted-foreground",
                    )}
                  >
                    {ft.name}
                  </code>
                  {ft.status === "active" && (
                    <Badge variant="active">DEPLOYED</Badge>
                  )}
                  <span className="text-[9px] px-1.5 py-px rounded bg-muted text-muted-foreground">
                    {ft.method}
                  </span>
                </div>
                <div className="flex gap-3 text-[10px] text-muted-foreground/40">
                  <span>
                    dataset:{" "}
                    <span className="text-muted-foreground">{ft.dataset}</span>
                  </span>
                  <span>
                    loss:{" "}
                    <span className="text-muted-foreground">{ft.loss}</span>
                  </span>
                  <span>
                    hub: <span className="text-blue-500">{ft.hubRef}</span>
                  </span>
                </div>
                <div className="text-[9px] text-muted-foreground/40 mt-0.5">
                  {ft.date}
                </div>
              </div>

              <div className="shrink-0 flex gap-1">
                {ft.status === "previous" && (
                  <Button variant="outline" size="xs">
                    Promote
                  </Button>
                )}
                <Button variant="ghost" size="xs">
                  Compare
                </Button>
              </div>
            </div>
          ))
        )}
      </section>
    </div>
  );
}
