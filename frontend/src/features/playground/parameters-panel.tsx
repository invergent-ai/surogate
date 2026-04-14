// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: AGPL-3.0-only
//
import { Button } from "@/components/ui/button";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Separator } from "@/components/ui/separator";
import { cn } from "@/utils/cn";
import { ParameterSlider } from "./parameter-slider";
import {
  PRESETS,
  type PlaygroundParams,
} from "./playground-data";

interface ParametersPanelProps {
  params: PlaygroundParams;
  onParamChange: (key: keyof PlaygroundParams, value: number) => void;
  activePreset: string | null;
  onApplyPreset: (id: string) => void;
}

export function ParametersPanel({
  params,
  onParamChange,
  activePreset,
  onApplyPreset,
}: ParametersPanelProps) {
  return (
    <div className="flex w-[260px] min-w-[260px] flex-col overflow-hidden border-l border-border bg-card">
      <div className="border-b border-border px-3 py-2">
        <span className="font-display text-[10px] font-semibold text-muted-foreground">
          Parameters
        </span>
      </div>
      <ScrollArea className="flex-1">
        <ParametersTab
          params={params}
          onParamChange={onParamChange}
          activePreset={activePreset}
          onApplyPreset={onApplyPreset}
        />
      </ScrollArea>
    </div>
  );
}

function ParametersTab({
  params,
  onParamChange,
  activePreset,
  onApplyPreset,
}: Pick<
  ParametersPanelProps,
  "params" | "onParamChange" | "activePreset" | "onApplyPreset"
>) {
  return (
    <div className="p-3.5 pb-6">
      <ParameterSlider
        label="Temperature"
        value={params.temperature}
        onChange={(v) => onParamChange("temperature", v)}
        min={0}
        max={2}
        step={0.05}
      />
      <ParameterSlider
        label="Top P"
        value={params.topP}
        onChange={(v) => onParamChange("topP", v)}
        min={0}
        max={1}
        step={0.05}
      />
      <ParameterSlider
        label="Top K"
        value={params.topK}
        onChange={(v) => onParamChange("topK", v)}
        min={1}
        max={100}
        step={1}
      />
      <ParameterSlider
        label="Max Tokens"
        value={params.maxTokens}
        onChange={(v) => onParamChange("maxTokens", v)}
        min={64}
        max={16384}
        step={64}
      />
      <ParameterSlider
        label="Repetition Penalty"
        value={params.repPenalty}
        onChange={(v) => onParamChange("repPenalty", v)}
        min={1.0}
        max={2.0}
        step={0.05}
      />

      <Separator className="my-3" />
      
      <div className="mb-2 font-display text-[9px] uppercase tracking-widest text-muted-foreground/60">
        Presets
      </div>
      <div className="flex flex-col gap-1">
        {PRESETS.map((p) => (
          <Button
            key={p.id}
            variant="outline"
            size="sm"
            className={cn(
              "h-8 justify-between font-display text-[10px]",
              activePreset === p.id &&
                "border-primary/30 bg-primary/5 text-primary",
            )}
            onClick={() => onApplyPreset(p.id)}
          >
            <span className="font-medium">{p.name}</span>
            <span className="text-[8px] text-muted-foreground/60">
              t={p.temp}
            </span>
          </Button>
        ))}
      </div>
    </div>
  );
}

