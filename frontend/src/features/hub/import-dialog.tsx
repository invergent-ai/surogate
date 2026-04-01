// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: AGPL-3.0-only
//
import { useState } from "react";
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogFooter } from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { Tabs, TabsList, TabsTrigger, TabsContent } from "@/components/ui/tabs";
import { HfSearchInput } from "@/components/ui/hf-search-input";
import { cn } from "@/utils/cn";

type ImportSource = "huggingface" | "modelscope";
type ImportType = "model" | "dataset";

interface ImportDialogProps {
  open: boolean;
  onClose: () => void;
  onImport: (params: {
    source: ImportSource;
    type: ImportType;
    repoId: string;
    ggufFile?: string;
    subset?: string;
    token?: string;
  }) => void;
}

function ImportForm({ source, onImport, onClose }: { source: ImportSource; onImport: ImportDialogProps["onImport"]; onClose: () => void }) {
  const [type, setType] = useState<ImportType>("model");
  const [repoId, setRepoId] = useState("");
  const [subset, setSubset] = useState("");
  const [token, setToken] = useState("");

  const isHf = source === "huggingface";
  const label = isHf ? "HuggingFace" : "ModelScope";
  const placeholder = isHf ? "meta-llama/Llama-3.1-8B" : "ZhipuAI/glm-4-9b";
  const tokenPlaceholder = isHf ? "hf_..." : "ms_...";

  return (
    <>
      <div className="space-y-4">
        <div>
          <label className="block mb-1 text-sm text-muted-foreground font-display">Type</label>
          <div className="flex gap-2">
            {(["model", "dataset"] as const).map((t) => (
              <button
                key={t}
                type="button"
                onClick={() => setType(t)}
                className={cn(
                  "flex-1 px-3 py-1.5 rounded-md border cursor-pointer font-display capitalize transition-all",
                  type === t
                    ? "border-success/30 bg-success/10 text-success font-semibold"
                    : "border-border text-muted-foreground",
                )}
              >
                {t}
              </button>
            ))}
          </div>
        </div>

        <div>
          <label className="block mb-1 text-sm text-muted-foreground font-display">
            {label} Repository ID
          </label>
          {isHf ? (
            <HfSearchInput
              value={repoId}
              onChange={setRepoId}
              kind={type === "model" ? "models" : "datasets"}
              token={token || undefined}
              placeholder={placeholder}
            />
          ) : (
            <Input
              value={repoId}
              onChange={(e) => setRepoId(e.target.value)}
              placeholder={placeholder}
              className="font-mono"
            />
          )}
        </div>

        {type === "dataset" && (
          <div>
            <label className="block mb-1 text-sm text-muted-foreground font-display">
              Subset <span className="text-faint">(optional)</span>
            </label>
            <Input
              value={subset}
              onChange={(e) => setSubset(e.target.value)}
              placeholder="default"
              className="font-mono"
            />
          </div>
        )}

        <div>
          <label className="block mb-1 text-sm text-muted-foreground font-display">
            Access Token <span className="text-faint">(optional)</span>
          </label>
          <Input
            value={token}
            onChange={(e) => setToken(e.target.value)}
            type="password"
            placeholder={tokenPlaceholder}
            className="font-mono"
          />
        </div>
      </div>

      <DialogFooter>
        <Button variant="outline" onClick={onClose}>
          Cancel
        </Button>
        <Button
          disabled={!repoId.trim()}
          onClick={() => {
            const raw = repoId.trim();
            const parts = raw.split("/");
            // HF repo IDs are org/name (2 segments); anything after is a gguf filename
            const repo = parts.slice(0, 2).join("/");
            const gguf = parts.length > 2 ? parts.slice(2).join("/") : undefined;
            onImport({
              source,
              type,
              repoId: repo,
              ggufFile: gguf,
              subset: subset.trim() || undefined,
              token: token.trim() || undefined,
            });
          }}
        >
          Import
        </Button>
      </DialogFooter>
    </>
  );
}

export function ImportDialog({ open, onClose, onImport }: ImportDialogProps) {
  return (
    <Dialog open={open} onOpenChange={(v) => { if (!v) onClose(); }}>
      <DialogContent className="sm:max-w-lg">
        <DialogHeader>
          <DialogTitle>Import Repository</DialogTitle>
        </DialogHeader>

        <Tabs defaultValue="huggingface">
          <TabsList variant="line" className="w-full">
            <TabsTrigger value="huggingface">HuggingFace</TabsTrigger>
            <TabsTrigger value="modelscope">ModelScope</TabsTrigger>
          </TabsList>
          <TabsContent value="huggingface">
            <ImportForm source="huggingface" onImport={onImport} onClose={onClose} />
          </TabsContent>
          <TabsContent value="modelscope">
            <ImportForm source="modelscope" onImport={onImport} onClose={onClose} />
          </TabsContent>
        </Tabs>
      </DialogContent>
    </Dialog>
  );
}
