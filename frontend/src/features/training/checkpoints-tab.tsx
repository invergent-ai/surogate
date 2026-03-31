// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: AGPL-3.0-only
//
import { Button } from "@/components/ui/button";
import { cn } from "@/utils/cn";

interface Checkpoint {
  step: number;
  loss: number;
  reward?: number;
  path: string;
  best?: boolean;
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
export function CheckpointsTab({ run }: { run: any }) {
  const checkpoints: Checkpoint[] = run.checkpoints;

  return (
    <div className="animate-in fade-in duration-200">
      <section className="bg-card rounded-lg ring-1 ring-foreground/10 overflow-hidden">
        <div className="px-4 py-3 border-b border-border flex justify-between items-center">
          <span className="text-[13px] font-semibold text-foreground font-display">
            Checkpoints
          </span>
          <span className="text-[10px] text-muted-foreground/60">
            {checkpoints.length} saved
          </span>
        </div>
        {checkpoints.length === 0 ? (
          <div className="py-8 text-center text-muted-foreground/30 text-[11px]">
            No checkpoints yet
          </div>
        ) : (
          checkpoints.map((ckpt, i) => (
            <div
              key={i}
              className="px-4 py-3 border-b border-border/30 flex items-center justify-between hover:bg-muted/30 transition-colors"
            >
              <div className="flex items-center gap-2.5">
                <div
                  className={cn(
                    "w-6 h-6 rounded-[5px] flex items-center justify-center text-[9px] font-bold border",
                    ckpt.best
                      ? "bg-green-500/[0.07] border-green-500/20 text-green-500"
                      : "bg-muted border-border text-muted-foreground",
                  )}
                >
                  {ckpt.best ? "\u2605" : i + 1}
                </div>
                <div>
                  <div className="text-xs text-foreground font-medium">
                    Step {ckpt.step.toLocaleString()}
                  </div>
                  <div className="text-[9px] text-muted-foreground/40">
                    loss: <span className="text-muted-foreground/60">{ckpt.loss}</span>
                    {ckpt.reward !== undefined && (
                      <> &middot; reward: <span className="text-green-500">{ckpt.reward}</span></>
                    )}
                  </div>
                </div>
              </div>
              <div className="flex gap-1.5">
                <Button variant="outline" size="xs" className="text-green-500 border-green-500/20 hover:bg-green-500/10">
                  Publish
                </Button>
                <Button variant="outline" size="xs">Evaluate</Button>
                <Button variant="outline" size="xs">Resume</Button>
              </div>
            </div>
          ))
        )}
      </section>
    </div>
  );
}
