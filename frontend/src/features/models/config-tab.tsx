// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: AGPL-3.0-only
//
import { useState } from "react";
import { Button } from "@/components/ui/button";
import {
  Select,
  SelectTrigger,
  SelectValue,
  SelectContent,
  SelectItem,
} from "@/components/ui/select";
import { useAppStore } from "@/stores/app-store";
import { cn } from "@/utils/cn";
import type { Model } from "./models-data";

const COMPUTE_TARGETS = [
  { icon: "⎈", label: "Local Cluster", desc: "Your Kubernetes nodes" },
  { icon: "☁", label: "Cloud", desc: "Launch on a cloud provider" },
];

const CLOUD_PROVIDERS = [
  { value: "aws", label: "AWS" },
  { value: "gcp", label: "GCP" },
  { value: "azure", label: "Azure" },
];

const ACCELERATOR_OPTIONS = [
  { value: "A100:1", label: "A100" },
  { value: "A10G:1", label: "A10G" },
  { value: "H100:1", label: "H100" },
  { value: "L4:1", label: "L4" },
  { value: "T4:1", label: "T4" },
];

const ENGINE_OPTIONS = [
  { value: "vllm", label: "vLLM" },
  { value: "llamacpp", label: "llama.cpp" },
];

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
  const updateModel = useAppStore((s) => s.updateModel);

  // Derive initial compute target from model's gpu type
  const hasCloud = model.gpu.type !== "\u2014" && CLOUD_PROVIDERS.some((p) => p.value === model.engine);
  const [computeIdx, setComputeIdx] = useState(hasCloud ? 1 : 0);
  const [cloud, setCloud] = useState("");
  const [accelerators, setAccelerators] = useState(
    model.gpu.type !== "\u2014" ? `${model.gpu.type}:${model.gpu.count || 1}` : "",
  );
  const [gpuCount, setGpuCount] = useState(String(model.gpu.count || 1));
  const [engine, setEngine] = useState(model.engine !== "\u2014" ? model.engine : "");
  const [saving, setSaving] = useState(false);

  const handleSave = async () => {
    setSaving(true);
    // Build accelerator string: type:count
    let acc = accelerators;
    if (acc && gpuCount !== "1") {
      const type = acc.split(":")[0];
      acc = `${type}:${gpuCount}`;
    }
    await updateModel(model.id, {
      engine: engine || undefined,
      accelerators: acc || undefined,
      cloud: computeIdx === 1 ? (cloud || undefined) : undefined,
    });
    setSaving(false);
  };

  return (
    <div className="animate-in fade-in duration-150 space-y-4">
      {/* Deployment Configuration */}
      <section className="bg-muted/40 border border-border rounded-lg overflow-hidden">
        <div className="px-4 py-3 border-b border-border flex items-center justify-between">
          <span className="text-[13px] font-semibold text-foreground font-display">
            Deployment Configuration
          </span>
          <Button variant="outline" size="xs" onClick={handleSave} disabled={saving}>
            {saving ? "Saving\u2026" : "Save"}
          </Button>
        </div>
        <div className="p-4 space-y-4">
          {/* Compute target cards */}
          <div>
            <div className="text-[10px] text-muted-foreground/50 uppercase tracking-wide mb-2 font-display">
              Compute
            </div>
            <div className="grid grid-cols-2 gap-2">
              {COMPUTE_TARGETS.map((t, i) => (
                <button
                  key={t.label}
                  onClick={() => setComputeIdx(i)}
                  className={cn(
                    "p-3 rounded-lg border text-center transition-colors cursor-pointer",
                    i === computeIdx
                      ? "border-blue-500 bg-muted/60"
                      : "border-border bg-muted/40 hover:border-blue-500/20 hover:bg-muted/60",
                  )}
                >
                  <div className="text-lg mb-1">{t.icon}</div>
                  <div className="text-[11px] font-semibold text-foreground font-display">
                    {t.label}
                  </div>
                  <div className="text-[9px] text-muted-foreground mt-0.5">
                    {t.desc}
                  </div>
                </button>
              ))}
            </div>
            {computeIdx === 1 && (
              <div className="mt-3">
                <Select value={cloud} onValueChange={setCloud}>
                  <SelectTrigger size="sm" className="w-full text-xs">
                    <SelectValue placeholder="Select cloud provider" />
                  </SelectTrigger>
                  <SelectContent>
                    {CLOUD_PROVIDERS.map((p) => (
                      <SelectItem key={p.value} value={p.value}>{p.label}</SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
            )}
          </div>

          {/* GPU + Engine */}
          <div className={cn("grid gap-3", computeIdx === 1 ? "grid-cols-3" : "grid-cols-2")}>
            {computeIdx === 1 && (
              <div>
                <div className="text-[9px] text-muted-foreground/50 mb-1 font-display uppercase tracking-wide">
                  GPU Type
                </div>
                <Select value={accelerators} onValueChange={setAccelerators}>
                  <SelectTrigger size="sm" className="w-full text-xs">
                    <SelectValue placeholder="Select GPU" />
                  </SelectTrigger>
                  <SelectContent>
                    {ACCELERATOR_OPTIONS.map((opt) => (
                      <SelectItem key={opt.value} value={opt.value}>{opt.label}</SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
            )}
            <div>
              <div className="text-[9px] text-muted-foreground/50 mb-1 font-display uppercase tracking-wide">
                GPU Count
              </div>
              <Select value={gpuCount} onValueChange={setGpuCount}>
                <SelectTrigger size="sm" className="w-full text-xs">
                  <SelectValue placeholder="1" />
                </SelectTrigger>
                <SelectContent>
                  {Array.from({ length: 8 }, (_, i) => (
                    <SelectItem key={i + 1} value={String(i + 1)}>{i + 1}</SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
            <div>
              <div className="text-[9px] text-muted-foreground/50 mb-1 font-display uppercase tracking-wide">
                Engine
              </div>
              <Select value={engine} onValueChange={setEngine}>
                <SelectTrigger size="sm" className="w-full text-xs">
                  <SelectValue placeholder="Select engine" />
                </SelectTrigger>
                <SelectContent>
                  {ENGINE_OPTIONS.map((opt) => (
                    <SelectItem key={opt.value} value={opt.value}>{opt.label}</SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
          </div>
        </div>
      </section>

      {/* Serving Parameters */}
      {model.servingConfig && (
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
      )}

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
