// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: AGPL-3.0-only
//
import type { Model } from "@/types/model";
import { DEFAULT_COLOR } from "./playground-data";

interface ModelInfoBarProps {
  model: Model;
  messageCount: number;
  totalTokens: number;
  avgLatency: number;
  isStreaming: boolean;
}

export function ModelInfoBar({
  model,
  messageCount,
  totalTokens,
  avgLatency,
  isStreaming,
}: ModelInfoBarProps) {
  const color = model.projectColor || DEFAULT_COLOR;

  return (
    <div className="flex shrink-0 items-center justify-between border-b border-border bg-card px-5 py-1.5">
      <div className="flex items-center gap-2 text-sm">
        <span
          className="inline-block size-1.5 shrink-0 rounded-full"
          style={{
            backgroundColor: color,
            boxShadow: `0 0 6px ${color}`,
          }}
        />
        <span className="font-display font-semibold" style={{ color }}>
          {model.displayName}
        </span>
        <span className="text-muted-foreground/40">·</span>
        <span className="text-muted-foreground">{model.tps} tok/s</span>
      </div>
      <div className="flex items-center gap-2.5 text-sm text-muted-foreground/60">
        <span>{messageCount} messages</span>
        <span>·</span>
        <span>{totalTokens.toLocaleString()} tokens</span>
        {avgLatency > 0 && (
          <>
            <span>·</span>
            <span>avg {avgLatency}ms</span>
          </>
        )}
        {isStreaming && (
          <span className="flex items-center gap-1 text-primary">
            <span className="size-[5px] animate-pulse rounded-full bg-primary" />
            streaming
          </span>
        )}
      </div>
    </div>
  );
}
