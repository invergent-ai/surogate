import { useState } from "react";
import { Button } from "@/components/ui/button";
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
  { icon: "\u2295", label: "Local Hub", desc: "From your registry" },
  { icon: "\uD83E\uDD17", label: "Hugging Face", desc: "From HF Hub" },
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

  const isHub = sourceIdx === 0;
  const baseModel = isHub
    ? hubSelection?.repo ?? ""
    : baseModelField.trim();
  const canSave = baseModel.length > 0;

  const handleSave = async () => {
    if (!canSave) return;

    const slug = baseModel.split("/").pop()?.toLowerCase().replace(/[^a-z0-9-]/g, "-") ?? "model";
    const displayName = baseModel.split("/").pop() ?? baseModel;

    setSaving(true);
    const result = await createModel({
      name: slug,
      display_name: displayName,
      base_model: baseModel,
      project_id: activeProjectId ?? "",
      hub_ref: hubSelection ? `${hubSelection.repo}@${hubSelection.ref}` : undefined,
    });
    setSaving(false);

    if (result) {
      setBaseModelField("");
      setHubSelection(null);
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

        <div className="text-[10px] text-muted-foreground/50 uppercase tracking-wide mb-2 font-display">
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
              <div className="text-[11px] font-semibold text-foreground font-display">
                {s.label}
              </div>
              <div className="text-[9px] text-muted-foreground mt-0.5">
                {s.desc}
              </div>
            </button>
          ))}
        </div>

        <div className="mt-1">
          <div className="text-[9px] text-muted-foreground/50 mb-1 font-display uppercase tracking-wide">
            Model
          </div>
          {isHub ? (
            <HubRepoSelector
              value={hubSelection}
              onSelect={setHubSelection}
            />
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
