import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { cn } from "@/utils/cn";
import { ChevronDownIcon, ChevronUpIcon } from "lucide-react";
// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: AGPL-3.0-only
//
import { useState } from "react";
import { PRESETS } from "./playground-data";

interface SystemPromptSectionProps {
  value: string;
  onChange: (value: string) => void;
  activePreset: string | null;
  onApplyPreset: (presetId: string) => void;
}

export function SystemPromptSection({
  value,
  onChange,
  activePreset,
  onApplyPreset,
}: SystemPromptSectionProps) {
  const [isOpen, setIsOpen] = useState(true);

  if (!isOpen) {
    return (
      <button
        type="button"
        className="flex w-full shrink-0 items-center gap-1.5 border-b border-border bg-card/50 px-5 py-1.5 text-left text-[9px] text-muted-foreground transition-colors hover:text-foreground"
        onClick={() => setIsOpen(true)}
      >
        <ChevronDownIcon className="size-3" />
        System prompt ({value.length} chars)
      </button>
    );
  }

  return (
    <div className="shrink-0 border-b border-border bg-card/50 px-5 py-2.5">
      <div className="mb-2 flex items-center justify-between">
        <div className="flex items-center gap-2">
          <span className="font-display text-[9px] font-semibold uppercase tracking-widest text-primary">
            System Prompt
          </span>
          <div className="flex gap-0.5">
            {PRESETS.map((p) => (
              <Button
                key={p.id}
                variant="ghost"
                size="xs"
                className={cn(
                  "h-[18px] rounded px-1.5 text-[8px]",
                  activePreset === p.id
                    ? "border border-primary/30 bg-primary/10 text-primary"
                    : "border border-transparent text-muted-foreground",
                )}
                onClick={() => onApplyPreset(p.id)}
              >
                {p.name}
              </Button>
            ))}
          </div>
        </div>
        <Button
          variant="ghost"
          size="icon-xs"
          className="text-muted-foreground"
          onClick={() => setIsOpen(false)}
        >
          <ChevronUpIcon className="size-3" />
        </Button>
      </div>
      <Textarea
        value={value}
        onChange={(e) => onChange(e.target.value)}
        rows={3}
        className="min-h-0 resize-y border-border bg-muted/50 font-mono text-[11px]"
      />
    </div>
  );
}
