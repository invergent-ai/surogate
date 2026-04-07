// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: AGPL-3.0-only
//
import { useEffect, useState } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import { Checkbox } from "@/components/ui/checkbox";
import { Input } from "@/components/ui/input";
import {
  Select,
  SelectTrigger,
  SelectValue,
  SelectContent,
  SelectItem,
} from "@/components/ui/select";
import { GpuOfferPicker } from "@/components/gpu-offer-picker";
import { useAppStore } from "@/stores/app-store";
import { cn } from "@/utils/cn";
import type { Model } from "./models-data";
import type { InstanceOffer } from "@/api/compute";
import { SUPPORTED_PROVIDERS } from "@/features/compute/compute-data";

const COMPUTE_TARGETS = [
  { icon: "⎈", label: "Local Cluster", desc: "Your Kubernetes nodes" },
  { icon: "☁", label: "Cloud", desc: "Launch on a cloud provider" },
];

const ENGINE_OPTIONS = [
  { value: "vllm", label: "vLLM" },
  { value: "llamacpp", label: "llama.cpp" },
];

const SERVING_DEFAULTS: Record<string, Record<string, unknown>> = {
  vllm: {
    max_model_len: 4096,
    tensor_parallel_size: 1,
    gpu_memory_utilization: 0.9,
    quantization: "",
    enforce_eager: false,
    attention_backend: "",
    reasoning_parser: "",
    tool_call_parser: "",
  },
  llamacpp: {
    ctx_size: 4096,
    threads: 4,
    reasoning_format: "",
  },
  openrouter: {
    api_key: "",
  },
  openai_compat: {
    endpoint: "",
    api_key: "",
  },
};

const GENERATION_PRESETS = [
  { id: "default",  label: "Default",  temp: 0.7, topP: 0.9,  topK: 40, repPenalty: 1.0 },
  { id: "creative", label: "Creative", temp: 1.0, topP: 0.95, topK: 80, repPenalty: 1.1 },
  { id: "precise",  label: "Precise",  temp: 0.1, topP: 0.5,  topK: 10, repPenalty: 1.0 },
  { id: "code",     label: "Code",     temp: 0.2, topP: 0.9,  topK: 40, repPenalty: 1.0 },
] as const;

function presetToDefaults(p: (typeof GENERATION_PRESETS)[number]): Record<string, unknown> {
  return {
    temperature: p.temp,
    top_p: p.topP,
    top_k: p.topK,
    repetition_penalty: p.repPenalty,
    stop_sequences: [],
  };
}

function ConfigKeyValue({
  entries,
  keyColor,
  editable,
  onChange,
}: {
  entries: [string, unknown][];
  keyColor: string;
  editable?: boolean;
  onChange?: (key: string, value: unknown) => void;
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
          {editable && onChange ? (
            typeof v === "boolean" ? (
              <Checkbox
                checked={!!v}
                onCheckedChange={(checked) => onChange(k, !!checked)}
              />
            ) : (
              <Input
                type="text"
                value={Array.isArray(v) ? JSON.stringify(v) : String(v ?? "")}
                onChange={(e) => {
                  const raw = e.target.value;
                  const num = Number(raw);
                  onChange(k, raw === "" ? "" : typeof v === "number" && !isNaN(num) ? num : raw);
                }}
                className="h-6 text-xs"
              />
            )
          ) : (
            typeof v === "boolean" ? (
              <Checkbox checked={v} disabled />
            ) : (
              <span className="text-foreground/70">
                {Array.isArray(v) ? JSON.stringify(v) : String(v)}
              </span>
            )
          )}
        </div>
      ))}
    </div>
  );
}

export function ConfigTab({ model }: { model: Model }) {
  const updateModel = useAppStore((s) => s.updateModel);
  const allBackends = useAppStore((s) => s.cloudBackends);
  const fetchBackends = useAppStore((s) => s.fetchCloudBackends);
  const activeProjectId = useAppStore((s) => s.activeProjectId);

  useEffect(() => { fetchBackends(); }, [activeProjectId, fetchBackends]);

  const backends = allBackends.filter((b) => b.type !== "kubernetes");

  const needsCompute = !model.source || model.source === "local_hub" || model.source === "huggingface";

  // Derive initial compute target from model's infra
  const isCloud = !!model.infra && model.infra !== "kubernetes";
  const [computeIdx, setComputeIdx] = useState(isCloud ? 1 : 0);
  const [cloud, setCloud] = useState(isCloud ? model.infra! : "");
  const [selectedOffer, setSelectedOffer] = useState<InstanceOffer | null>(null);
  const [localGpuCount, setLocalGpuCount] = useState(
    !isCloud ? String(model.gpu.count) : "1"
  );
  const [engine, setEngine] = useState(model.engine !== "\u2014" ? model.engine : "");

  const [saving, setSaving] = useState(false);

  const activeEngine = engine || (model.engine !== "\u2014" ? model.engine : "");
  const servingDefaults = SERVING_DEFAULTS[activeEngine] ?? {};

  const [editedServing, setEditedServing] = useState<Record<string, unknown>>(
    model.servingConfig ? { ...model.servingConfig } : { ...servingDefaults },
  );
  const [editedGen, setEditedGen] = useState<Record<string, unknown>>(
    model.generationDefaults ? { ...model.generationDefaults } : {},
  );

  const handleSave = async () => {
    setSaving(true);
    const acc = computeIdx === 1 && selectedOffer && selectedOffer.gpu_name
      ? `${selectedOffer.gpu_name}:${selectedOffer.gpu_count}`
      : computeIdx === 0 && localGpuCount !== "0"
        ? `GPU:${localGpuCount}`
        : undefined;
    await updateModel(model.id, {
      engine: engine || undefined,
      accelerators: acc,
      infra: computeIdx === 1 ? (cloud || undefined) : "kubernetes",
      instance_type: computeIdx === 1 && selectedOffer ? selectedOffer.instance : undefined,
      region: computeIdx === 1 && selectedOffer ? selectedOffer.region : undefined,
      use_spot: selectedOffer?.spot ?? false,
      serving_config: Object.keys(editedServing).length > 0 ? editedServing : undefined,
      generation_defaults: Object.keys(editedGen).length > 0 ? editedGen : undefined,
    });
    setSaving(false);
  };

  const handleApplyPreset = (preset: (typeof GENERATION_PRESETS)[number]) => {
    setEditedGen(presetToDefaults(preset));
  };

  return (
    <div className="animate-in fade-in duration-150 space-y-4">
      {/* Save controls */}
      <div className="flex justify-end">
        <Button variant="outline" size="sm" onClick={handleSave} disabled={saving}>
          {saving ? "Saving\u2026" : "Save Configuration"}
        </Button>
      </div>

      {/* Deployment Configuration */}
      {needsCompute && (
        <Card size="sm">
          <CardHeader>
            <CardTitle>Deployment Configuration</CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            {/* Compute target cards */}
            <div>
              <div className="text-muted-foreground/50 uppercase tracking-wide mb-2 ">
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
                    <div className="font-semibold text-foreground ">
                      {t.label}
                    </div>
                    <div className="text-muted-foreground mt-0.5">
                      {t.desc}
                    </div>
                  </button>
                ))}
              </div>
              {computeIdx === 0 && (
                <div className="mt-3">
                  <Select value={localGpuCount} onValueChange={setLocalGpuCount}>
                    <SelectTrigger className="w-full">
                      <SelectValue placeholder="Number of GPUs" />
                    </SelectTrigger>
                    <SelectContent>
                      {["0", "1", "2", "4", "8"].map((n) => (
                        <SelectItem key={n} value={n}>{n === "0" ? "CPU only" : `${n} GPU${n !== "1" ? "s" : ""}`}</SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>
              )}
              {computeIdx === 1 && (
                <div className="mt-3">
                  {backends.length === 0 ? (
                    <div className="text-sm text-muted-foreground/60 py-2">
                      No cloud backends connected. Go to Compute &rarr; Cloud to add one.
                    </div>
                  ) : (
                    <Select value={cloud} onValueChange={setCloud}>
                      <SelectTrigger className="w-full">
                        <SelectValue placeholder="Select cloud backend" />
                      </SelectTrigger>
                      <SelectContent>
                        {backends.map((b) => {
                          const info = SUPPORTED_PROVIDERS.find((p) => p.key === b.type);
                          return (
                            <SelectItem key={b.id} value={b.type}>
                              {info?.name ?? b.type.toUpperCase()}
                            </SelectItem>
                          );
                        })}
                      </SelectContent>
                    </Select>
                  )}
                </div>
              )}
            </div>

            {/* Engine */}
            <div>
              <div className="text-muted-foreground/50 mb-1  uppercase tracking-wide">
                Engine
              </div>
              <Select value={engine} onValueChange={(v) => {
                setEngine(v);
                setEditedServing(SERVING_DEFAULTS[v] ? { ...SERVING_DEFAULTS[v] } : {});
              }}>
                <SelectTrigger className="w-full">
                  <SelectValue placeholder="Select engine" />
                </SelectTrigger>
                <SelectContent>
                  {ENGINE_OPTIONS.map((opt) => (
                    <SelectItem key={opt.value} value={opt.value}>{opt.label}</SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>

            {/* Instance offers */}
            {computeIdx === 1 && cloud && (
              <GpuOfferPicker
                backend={cloud}
                selectedOffer={selectedOffer}
                onSelect={setSelectedOffer}
              />
            )}

          </CardContent>
        </Card>
      )}

      {/* Serving Parameters */}
      <Card size="sm">
        <CardHeader>
          <CardTitle>Serving Parameters</CardTitle>
        </CardHeader>
        <CardContent>
          <ConfigKeyValue
            entries={Object.entries(editedServing)}
            keyColor="text-blue-500"
            editable
            onChange={(key, value) => setEditedServing((prev) => ({ ...prev, [key]: value }))}
          />
        </CardContent>
      </Card>

      {/* Generation Defaults */}
      <Card size="sm">
        <CardHeader>
          <CardTitle>Generation Defaults</CardTitle>
        </CardHeader>
        <CardContent className="space-y-3">
          <div className="flex flex-wrap gap-1.5">
            {GENERATION_PRESETS.map((p) => (
              <Button
                key={p.id}
                variant="outline"
                size="xs"
                onClick={() => handleApplyPreset(p)}
              >
                {p.label}
              </Button>
            ))}
          </div>
          {Object.keys(editedGen).length > 0 && (
            <ConfigKeyValue
              entries={Object.entries(editedGen)}
              keyColor="text-violet-500"
              editable
              onChange={(key, value) => setEditedGen((prev) => ({ ...prev, [key]: value }))}
            />
          )}
        </CardContent>
      </Card>
    </div>
  );
}
