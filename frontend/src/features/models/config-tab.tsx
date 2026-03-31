// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: AGPL-3.0-only
//
import { Button } from "@/components/ui/button";
import { cn } from "@/utils/cn";
import type { Model } from "./models-data";

function ConfigKeyValue({
  entries,
  keyColor,
}: {
  entries: [string, unknown][];
  keyColor: string;
}) {
  return (
    <div className="font-mono">
      {entries.map(([k, v]) => (
        <div
          key={k}
          className="px-4 py-2 border-b border-border/50 last:border-b-0 flex items-center gap-4 text-xs"
        >
          <span className={cn("min-w-[220px]", keyColor)}>{k}</span>
          <span className="text-muted-foreground/20">=</span>
          <span
            className={cn(
              typeof v === "boolean"
                ? v
                  ? "text-green-500"
                  : "text-destructive"
                : "text-foreground/70",
            )}
          >
            {Array.isArray(v)
              ? JSON.stringify(v)
              : typeof v === "boolean"
                ? v
                  ? "true"
                  : "false"
                : String(v)}
          </span>
        </div>
      ))}
    </div>
  );
}

export function ConfigTab({ model }: { model: Model }) {
  if (!model.servingConfig) {
    return (
      <div className="animate-in fade-in duration-150">
        <div className="py-10 text-center text-muted-foreground/40 text-xs">
          Model not deployed &mdash; no serving configuration available
        </div>
      </div>
    );
  }

  return (
    <div className="animate-in fade-in duration-150 space-y-4">
      {/* Serving Parameters */}
      <section className="bg-muted/40 border border-border rounded-lg overflow-hidden">
        <div className="px-4 py-3 border-b border-border flex items-center justify-between">
          <span className="text-[13px] font-semibold text-foreground font-display">
            Serving Parameters
          </span>
          <Button variant="outline" size="xs">
            Edit
          </Button>
        </div>
        <ConfigKeyValue
          entries={Object.entries(model.servingConfig)}
          keyColor="text-blue-500"
        />
      </section>

      {/* Generation Defaults */}
      {model.generationDefaults && (
        <section className="bg-muted/40 border border-border rounded-lg overflow-hidden">
          <div className="px-4 py-3 border-b border-border flex items-center justify-between">
            <span className="text-[13px] font-semibold text-foreground font-display">
              Generation Defaults
            </span>
            <Button variant="outline" size="xs">
              Edit
            </Button>
          </div>
          <ConfigKeyValue
            entries={Object.entries(model.generationDefaults)}
            keyColor="text-violet-500"
          />
        </section>
      )}

      {/* Container */}
      <section className="bg-muted/40 border border-border rounded-lg p-4">
        <div className="text-xs font-semibold text-foreground font-display mb-2.5">
          Container
        </div>
        <div className="bg-background border border-border rounded-md px-3.5 py-2.5 font-mono text-[11px] text-muted-foreground mb-3">
          <span className="text-muted-foreground/40">image:</span>{" "}
          <span className="text-foreground/70">{model.image}</span>
        </div>
        {model.endpoint !== "\u2014" && (
          <div className="bg-background border border-border rounded-md px-3.5 py-2.5 font-mono text-[11px] text-muted-foreground">
            <span className="text-muted-foreground/40">endpoint:</span>{" "}
            <span className="text-foreground/70">{model.endpoint}</span>
          </div>
        )}
      </section>
    </div>
  );
}
