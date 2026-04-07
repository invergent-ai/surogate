import { useState } from "react";
import { Database, Link, Route } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { HfSearchInput } from "@/components/ui/hf-search-input";
import { HubRepoSelector, type HubRefSelection } from "@/components/ui/hub-repo-selector";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
  DialogFooter,
} from "@/components/ui/dialog";
import { useAppStore } from "@/stores/app-store";
import { cn } from "@/utils/cn";

const SERVE_SOURCES = [
  { icon: <Database className="w-5 h-5 mx-auto" />, label: "Local Hub", desc: "From your registry" },
  { icon: <span className="text-lg">🤗</span>, label: "Hugging Face", desc: "From HF Hub" },
  { icon: <Route className="w-5 h-5 mx-auto" />, label: "OpenRouter", desc: "From OpenRouter" },
  { icon: <Link className="w-5 h-5 mx-auto" />, label: "URL", desc: "From OpenAI-compatible provider" },
];

export function ServeModelDialog({
  open,
  onOpenChange,
}: {
  open: boolean;
  onOpenChange: (open: boolean) => void;
}) {
  const createModel = useAppStore((s) => s.createModel);
  const activeProjectId = useAppStore((s) => s.activeProjectId);
  const [baseModelField, setBaseModelField] = useState("");
  const [saving, setSaving] = useState(false);
  const [sourceIdx, setSourceIdx] = useState(0);
  const [hubSelection, setHubSelection] = useState<HubRefSelection | null>(null);
  const [openRouterApiKey, setOpenRouterApiKey] = useState("");
  const [openRouterModel, setOpenRouterModel] = useState("");
  const [urlEndpoint, setUrlEndpoint] = useState("");
  const [urlModel, setUrlModel] = useState("");

  const isHub = sourceIdx === 0;
  const isHuggingFace = sourceIdx === 1;
  const isOpenRouter = sourceIdx === 2;
  const isUrl = sourceIdx === 3;

  let baseModel: string;
  if (isHub) {
    baseModel = hubSelection?.repo ?? "";
  } else if (isOpenRouter) {
    baseModel = openRouterModel.trim();
  } else if (isUrl) {
    baseModel = urlModel.trim();
  } else {
    baseModel = baseModelField.trim();
  }

  let canSave = baseModel.length > 0;
  if (isOpenRouter) canSave = canSave && openRouterApiKey.trim().length > 0;
  if (isUrl) canSave = canSave && urlEndpoint.trim().length > 0;

  const handleSave = async () => {
    if (!canSave) return;

    const slug = baseModel.split("/").pop()?.toLowerCase().replace(/[^a-z0-9-]/g, "-") ?? "model";
    const displayName = baseModel.split("/").pop() ?? baseModel;

    const servingConfig: Record<string, unknown> = {};
    if (isOpenRouter) servingConfig.api_key = openRouterApiKey.trim();
    if (isUrl) servingConfig.endpoint = urlEndpoint.trim();

    setSaving(true);
    const source = isHub ? "local_hub" : isHuggingFace ? "huggingface" : isOpenRouter ? "openrouter" : "url";

    const result = await createModel({
      name: slug,
      display_name: displayName,
      base_model: baseModel,
      project_id: activeProjectId ?? "",
      hub_ref: hubSelection ? `${hubSelection.repo}@${hubSelection.ref}` : undefined,
      engine: isOpenRouter ? "openrouter" : isUrl ? "openai_compat" : undefined,
      source,
      serving_config: Object.keys(servingConfig).length > 0 ? servingConfig : undefined,
      generation_defaults: {
        temperature: 0.7,
        top_p: 0.9,
        top_k: 40,
        max_tokens: 2048,
        repetition_penalty: 1.0,
        stop_sequences: [],
      },
    });
    setSaving(false);

    if (result) {
      setBaseModelField("");
      setHubSelection(null);
      setOpenRouterApiKey("");
      setOpenRouterModel("");
      setUrlEndpoint("");
      setUrlModel("");
      onOpenChange(false);
    }
  };

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent>
        <DialogHeader>
          <DialogTitle>Add Model</DialogTitle>
          <DialogDescription>
            Add a model to your registry, then configure serving options
          </DialogDescription>
        </DialogHeader>

        <div className="text-muted-foreground/50 uppercase font-medium">
          Source
        </div>
        <div className="grid grid-cols-2 gap-2">
          {SERVE_SOURCES.map((s, i) => (
            <button
              key={s.label}
              onClick={() => setSourceIdx(i)}
              className={cn(
                "p-3 rounded-lg border text-center transition-colors cursor-pointer",
                i === sourceIdx
                  ? "border-blue-500 bg-muted/60"
                  : "border-border bg-muted/40 hover:border-blue-500/20 hover:bg-muted/60",
              )}
            >
              <div className="text-lg mb-1">{s.icon}</div>
              <div className="font-semibold text-foreground font-display">
                {s.label}
              </div>
              <div className="text-xs text-muted-foreground mt-0.5">
                {s.desc}
              </div>
            </button>
          ))}
        </div>

        <div className="mt-1 flex flex-col gap-2">
          <div className="text-muted-foreground/50 uppercase font-medium">
            Model
          </div>
          {isHub ? (
            <HubRepoSelector
              value={hubSelection}
              onSelect={setHubSelection}
            />
          ) : isOpenRouter ? (
            <>
              <Input
                value={openRouterModel}
                onChange={(e) => setOpenRouterModel(e.target.value)}
                placeholder="openai/gpt-4o"
                className="h-8 text-xs"
              />
              <Input
                value={openRouterApiKey}
                onChange={(e) => setOpenRouterApiKey(e.target.value)}
                placeholder="OpenRouter API Key"
                type="password"
                className="h-8 text-xs"
              />
            </>
          ) : isUrl ? (
            <>
              <Input
                value={urlEndpoint}
                onChange={(e) => setUrlEndpoint(e.target.value)}
                placeholder="https://api.example.com/v1"
                className="h-8 text-xs"
              />
              <Input
                value={urlModel}
                onChange={(e) => setUrlModel(e.target.value)}
                placeholder="meta-llama/Llama-3.1-8B-Instruct"
                className="h-8 text-xs"
              />
            </>
          ) : (
            <HfSearchInput
              value={baseModelField}
              onChange={setBaseModelField}
              kind="models"
              placeholder="meta-llama/Llama-3.1-8B-Instruct"
              className="h-8 text-xs"
            />
          )}
        </div>

        <DialogFooter>
          <Button variant="outline" onClick={() => onOpenChange(false)}>
            Cancel
          </Button>
          <Button onClick={handleSave} disabled={saving || !canSave}>
            {saving ? "Saving\u2026" : "Add Model"}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
