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
import { Textarea } from "@/components/ui/textarea";

export function NewExperimentDialog({ open, onOpenChange }: { open: boolean; onOpenChange: (open: boolean) => void }) {
  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent>
        <DialogHeader>
          <DialogTitle>New Experiment</DialogTitle>
          <DialogDescription>Group related training runs under a hypothesis</DialogDescription>
        </DialogHeader>

        <div className="space-y-3">
          <div>
            <div className="text-[9px] text-muted-foreground/50 mb-1 font-display uppercase tracking-wide">
              Experiment Name
            </div>
            <Input placeholder="CX Quality Improvement" className="h-8 text-xs" />
          </div>
          <div>
            <div className="text-[9px] text-muted-foreground/50 mb-1 font-display uppercase tracking-wide">
              Base Model
            </div>
            <Input placeholder="meta-llama/Llama-3.1-8B-Instruct" className="h-8 text-xs" />
          </div>
          <div>
            <div className="text-[9px] text-muted-foreground/50 mb-1 font-display uppercase tracking-wide">
              Hypothesis
            </div>
            <Textarea
              placeholder="DPO on curated preference pairs will improve..."
              rows={3}
              className="text-xs"
            />
          </div>
        </div>

        <DialogFooter>
          <Button variant="outline" onClick={() => onOpenChange(false)}>Cancel</Button>
          <Button onClick={() => onOpenChange(false)}>Create</Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
