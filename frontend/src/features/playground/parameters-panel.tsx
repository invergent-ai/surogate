// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: AGPL-3.0-only
//
import { Button } from "@/components/ui/button";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Separator } from "@/components/ui/separator";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import type { Model } from "@/types/model";
import { cn } from "@/utils/cn";
import {
  BoxIcon,
  DownloadIcon,
  FlaskConicalIcon,
  ScrollTextIcon,
  XIcon,
} from "lucide-react";
import { ParameterSlider } from "./parameter-slider";
import {
  DEFAULT_COLOR,
  PRESETS,
  type PlaygroundParams,
} from "./playground-data";

interface ParametersPanelProps {
  model: Model;
  params: PlaygroundParams;
  onParamChange: (key: keyof PlaygroundParams, value: number) => void;
  activePreset: string | null;
  onApplyPreset: (id: string) => void;
  stats: {
    messageCount: number;
    totalTokens: number;
    avgLatency: number;
    userTurns: number;
  };
}

export function ParametersPanel({
  model,
  params,
  onParamChange,
  activePreset,
  onApplyPreset,
  stats,
}: ParametersPanelProps) {
  return (
    <div className="flex w-[260px] min-w-[260px] flex-col overflow-hidden border-l border-border bg-card">
      <Tabs defaultValue="params" className="flex flex-1 flex-col gap-0">
        <TabsList
          variant="line"
          className="w-full rounded-none border-b border-border bg-transparent px-3 py-0"
        >
          <TabsTrigger value="params" className="flex-1 text-[10px]">
            Parameters
          </TabsTrigger>
          <TabsTrigger value="info" className="flex-1 text-[10px]">
            Model Info
          </TabsTrigger>
        </TabsList>

        <TabsContent value="params" className="flex-1 overflow-hidden">
          <ScrollArea className="h-full">
            <ParametersTab
              params={params}
              onParamChange={onParamChange}
              activePreset={activePreset}
              onApplyPreset={onApplyPreset}
            />
          </ScrollArea>
        </TabsContent>

        <TabsContent value="info" className="flex-1 overflow-hidden">
          <ScrollArea className="h-full">
            <ModelInfoTab model={model} stats={stats} />
          </ScrollArea>
        </TabsContent>
      </Tabs>
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
        Stop Sequences
      </div>
      <div className="mb-3 flex flex-wrap gap-1">
        {["</s>", "[END]", "\\n\\n"].map((s) => (
          <span
            key={s}
            className="flex items-center gap-1 rounded border border-border bg-muted px-1.5 py-0.5 font-mono text-[9px] text-muted-foreground"
          >
            {s}
            <button
              type="button"
              className="cursor-pointer text-muted-foreground/40 hover:text-foreground"
            >
              <XIcon className="size-2" />
            </button>
          </span>
        ))}
        <button
          type="button"
          className="cursor-pointer rounded border border-dashed border-border px-1.5 py-0.5 text-[9px] text-muted-foreground/60 hover:text-muted-foreground"
        >
          + add
        </button>
      </div>

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

const QUICK_ACTIONS = [
  { label: "View in Models", icon: BoxIcon },
  { label: "Run Evaluation", icon: FlaskConicalIcon },
  { label: "Export to Dataset", icon: DownloadIcon },
  { label: "View Logs", icon: ScrollTextIcon },
] as const;

function ModelInfoTab({
  model,
  stats,
}: Pick<ParametersPanelProps, "model" | "stats">) {
  const color = model.projectColor || DEFAULT_COLOR;

  return (
    <div className="p-3.5 pb-6">
      {/* Model icon */}
      <div
        className="mx-auto mt-2 mb-3 flex size-12 items-center justify-center rounded-xl text-[22px]"
        style={{
          backgroundColor: `${color}12`,
          border: `1px solid ${color}25`,
          color,
        }}
      >
        ◇
      </div>
      <div className="mb-4 text-center">
        <div className="font-display text-[13px] font-bold text-foreground">
          {model.displayName}
        </div>
        <div className="mt-0.5 text-[10px] text-muted-foreground">
          {model.family} · {model.paramCount}
        </div>
      </div>

      {/* Model fields */}
      <div className="flex justify-between border-b border-border/50 py-1.5 text-[10px]">
        <span className="text-muted-foreground/60">Status</span>
        <span style={{ color: "#22C55E" }}>{model.status}</span>
      </div>
      <div className="flex justify-between border-b border-border/50 py-1.5 text-[10px]">
        <span className="text-muted-foreground/60">Parameters</span>
        <span className="text-foreground/70">{model.paramCount}</span>
      </div>
      <div className="flex justify-between border-b border-border/50 py-1.5 text-[10px]">
        <span className="text-muted-foreground/60">Quantization</span>
        <span className="text-foreground/70">{model.quantization}</span>
      </div>
      <div className="flex justify-between border-b border-border/50 py-1.5 text-[10px]">
        <span className="text-muted-foreground/60">Throughput</span>
        <span className="text-foreground/70">{model.tps} tok/s</span>
      </div>
      <div className="flex justify-between border-b border-border/50 py-1.5 text-[10px]">
        <span className="text-muted-foreground/60">Endpoint</span>
        <span className="text-foreground/70">{model.endpoint || "—"}</span>
      </div>

      {/* Session Stats */}
      <div className="mt-4">
        <div className="mb-2 font-display text-[9px] uppercase tracking-widest text-muted-foreground/60">
          Session Stats
        </div>
        <div className="grid grid-cols-2 gap-2">
          {[
            { label: "Messages", value: stats.messageCount },
            {
              label: "Total Tokens",
              value: stats.totalTokens.toLocaleString(),
            },
            {
              label: "Avg Latency",
              value: stats.avgLatency > 0 ? `${stats.avgLatency}ms` : "—",
            },
            { label: "User Turns", value: stats.userTurns },
          ].map((s) => (
            <div
              key={s.label}
              className="rounded-md border border-border bg-muted/50 px-2.5 py-2"
            >
              <div className="mb-0.5 font-display text-[8px] uppercase tracking-wider text-muted-foreground/60">
                {s.label}
              </div>
              <div className="text-sm font-bold text-foreground">{s.value}</div>
            </div>
          ))}
        </div>
      </div>

      {/* Quick Actions */}
      <div className="mt-4">
        <div className="mb-2 font-display text-[9px] uppercase tracking-widest text-muted-foreground/60">
          Quick Actions
        </div>
        <div className="flex flex-col gap-1">
          {QUICK_ACTIONS.map(({ label, icon: Icon }) => (
            <Button
              key={label}
              variant="outline"
              size="sm"
              className="justify-start gap-1.5 text-[10px] text-muted-foreground"
            >
              <Icon className="size-3" />
              {label}
            </Button>
          ))}
        </div>
      </div>
    </div>
  );
}
