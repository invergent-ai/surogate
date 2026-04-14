// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: AGPL-3.0-only
//
import type { Model } from "@/types/model";
import { DEFAULT_COLOR } from "./playground-data";

export function ModelInfoBar({ model }: { model: Model }) {
  const color = model.projectColor || DEFAULT_COLOR;

  return (
    <div className="flex shrink-0 items-center border-b border-border bg-card px-5 py-1.5">
      <span
        className="inline-block size-1.5 shrink-0 rounded-full"
        style={{
          backgroundColor: color,
          boxShadow: `0 0 6px ${color}`,
        }}
      />
      <span className="ml-2 font-display text-sm font-semibold" style={{ color }}>
        {model.displayName}
      </span>
    </div>
  );
}
