// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: AGPL-3.0-only
//
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
  DialogFooter,
} from "@/components/ui/dialog";

const CREATE_TEMPLATES = [
  { icon: "\u2B21", label: "From Conversations", desc: "Export from agent conversations" },
  { icon: "\u2295", label: "Upload File", desc: "JSONL, CSV, Parquet" },
  { icon: "\u25EC", label: "Synthetic Generation", desc: "Generate with NeMo Data Designer" },
  { icon: "\u29C9", label: "From Hub", desc: "Clone from model registry" },
  { icon: "\u25A4", label: "From Dataset", desc: "Fork & transform existing dataset" },
  { icon: "\u2605", label: "Manual Curation", desc: "Build sample by sample" },
];

export function NewDatasetDialog({
  open,
  onOpenChange,
}: {
  open: boolean;
  onOpenChange: (open: boolean) => void;
}) {
  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent>
        <DialogHeader>
          <DialogTitle>New Dataset</DialogTitle>
          <DialogDescription>
            Create a dataset from conversations, files, or synthetic generation
          </DialogDescription>
        </DialogHeader>
        <div className="grid grid-cols-2 gap-2.5">
          {CREATE_TEMPLATES.map((t) => (
            <button
              key={t.label}
              className="p-3.5 rounded-lg border border-border bg-muted/40 hover:border-amber-500/30 hover:bg-muted/60 transition-colors cursor-pointer text-left flex items-start gap-2.5"
            >
              <span className="text-lg text-amber-500 shrink-0">{t.icon}</span>
              <div>
                <div className="text-xs font-semibold text-foreground font-display">
                  {t.label}
                </div>
                <div className="text-[10px] text-muted-foreground mt-0.5">
                  {t.desc}
                </div>
              </div>
            </button>
          ))}
        </div>
        <DialogFooter>
          <Button variant="outline" onClick={() => onOpenChange(false)}>
            Cancel
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
