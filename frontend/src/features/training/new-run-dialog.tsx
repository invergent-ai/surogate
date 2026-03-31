// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: AGPL-3.0-only
//
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
  DialogFooter,
} from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Select, SelectTrigger, SelectValue, SelectContent, SelectItem } from "@/components/ui/select";
import { cn } from "@/utils/cn";

const METHODS = [
  { method: "SFT", desc: "Supervised Fine-tuning", icon: "\u25A4", color: "green" },
  { method: "DPO", desc: "Direct Preference Optimization", icon: "\u25C8", color: "violet" },
  { method: "GRPO", desc: "Group Relative Policy Optimization", icon: "\u25EC", color: "blue" },
  { method: "PPO", desc: "Proximal Policy Optimization", icon: "\u2B21", color: "amber" },
] as const;

const COLOR_MAP: Record<string, { bg: string; border: string; fg: string; hoverBorder: string }> = {
  green: { bg: "bg-green-500/[0.07]", border: "border-green-500/20", fg: "text-green-500", hoverBorder: "hover:border-green-500" },
  violet: { bg: "bg-violet-500/[0.07]", border: "border-violet-500/20", fg: "text-violet-500", hoverBorder: "hover:border-violet-500" },
  blue: { bg: "bg-blue-500/[0.07]", border: "border-blue-500/20", fg: "text-blue-500", hoverBorder: "hover:border-blue-500" },
  amber: { bg: "bg-amber-500/[0.07]", border: "border-amber-500/20", fg: "text-amber-500", hoverBorder: "hover:border-amber-500" },
};

const RUN_FIELDS = [
  { label: "Base Model", placeholder: "llama-3.1-8b-cx-v4" },
  { label: "Dataset", placeholder: "cx-convos-v5" },
  { label: "Experiment", placeholder: "CX Quality Improvement" },
  { label: "Run Name", placeholder: "CX SFT Round 5" },
];

export function NewRunDialog({ open, onOpenChange }: { open: boolean; onOpenChange: (open: boolean) => void }) {
  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent>
        <DialogHeader>
          <DialogTitle>New Training Run</DialogTitle>
          <DialogDescription>Configure SFT, DPO, or RL training</DialogDescription>
        </DialogHeader>

        <div className="text-[9px] text-muted-foreground/50 uppercase tracking-wide mb-2 font-display">
          Training Method
        </div>
        <div className="grid grid-cols-4 gap-2 mb-5">
          {METHODS.map(m => {
            const c = COLOR_MAP[m.color];
            return (
              <button
                key={m.method}
                className={cn(
                  "p-3 rounded-lg border text-center transition-colors cursor-pointer",
                  c.bg, c.border, c.hoverBorder,
                )}
              >
                <div className={cn("text-lg mb-1", c.fg)}>{m.icon}</div>
                <div className="text-xs font-semibold text-foreground font-display">{m.method}</div>
                <div className="text-[8px] text-muted-foreground/60 mt-0.5 leading-tight">{m.desc}</div>
              </button>
            );
          })}
        </div>

        <div className="grid grid-cols-2 gap-3 mb-4">
          {RUN_FIELDS.map(f => (
            <div key={f.label}>
              <div className="text-[9px] text-muted-foreground/50 mb-1 font-display uppercase tracking-wide">
                {f.label}
              </div>
              <Input placeholder={f.placeholder} className="h-8 text-xs" />
            </div>
          ))}
        </div>

        <div className="grid grid-cols-2 gap-3">
          <div>
            <div className="text-[9px] text-muted-foreground/50 mb-1 font-display uppercase tracking-wide">
              Compute
            </div>
            <Select defaultValue="local">
              <SelectTrigger size="sm"><SelectValue /></SelectTrigger>
              <SelectContent>
                <SelectItem value="local">Local Cluster</SelectItem>
                <SelectItem value="aws">AWS (SkyPilot)</SelectItem>
                <SelectItem value="gcp">GCP (SkyPilot)</SelectItem>
              </SelectContent>
            </Select>
          </div>
          <div>
            <div className="text-[9px] text-muted-foreground/50 mb-1 font-display uppercase tracking-wide">
              GPU
            </div>
            <Select defaultValue="4xh100">
              <SelectTrigger size="sm"><SelectValue /></SelectTrigger>
              <SelectContent>
                <SelectItem value="4xh100">4&times; H100 80GB</SelectItem>
                <SelectItem value="4xa100">4&times; A100 80GB</SelectItem>
                <SelectItem value="8xa100">8&times; A100 80GB</SelectItem>
                <SelectItem value="1xa100">1&times; A100 80GB</SelectItem>
              </SelectContent>
            </Select>
          </div>
        </div>

        <DialogFooter>
          <Button variant="outline" onClick={() => onOpenChange(false)}>Cancel</Button>
          <Button onClick={() => onOpenChange(false)}>Start Training</Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
