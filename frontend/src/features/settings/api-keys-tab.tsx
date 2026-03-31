// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: AGPL-3.0-only
//
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { cn } from "@/utils/cn";
import { API_KEYS } from "./settings-data";

export function ApiKeysTab() {
  return (
    <div>
      <div className="flex justify-between items-center mb-5">
        <h2 className="font-display text-[15px] font-bold text-foreground">
          API Keys
        </h2>
        <Button size="sm">+ Create Key</Button>
      </div>
      <p className="text-[11px] text-muted-foreground mb-4">
        API keys authenticate external clients against the Studio API. Keys are
        scoped to specific permissions.
      </p>

      <div className="bg-card border border-line rounded-lg overflow-hidden">
        {API_KEYS.map((k) => (
          <div
            key={k.id}
            className={cn(
              "px-4 py-3 border-b border-line last:border-b-0",
              k.status === "revoked" && "opacity-40",
            )}
          >
            <div className="flex items-center justify-between mb-1">
              <div className="flex items-center gap-2">
                <span className="font-display text-xs font-semibold text-foreground">
                  {k.name}
                </span>
                {k.status === "revoked" && (
                  <Badge variant="danger">REVOKED</Badge>
                )}
              </div>
              {k.status === "active" && (
                <Button variant="destructive" size="xs">
                  Revoke
                </Button>
              )}
            </div>
            <div className="flex items-center gap-3 text-[10px] text-faint">
              <code className="text-muted-foreground bg-accent px-1.5 py-px rounded text-[10px]">
                {k.prefix}
              </code>
              <span>created {k.created}</span>
              <span>last used {k.lastUsed}</span>
              <span className="flex-1" />
              <div className="flex gap-1">
                {k.scopes.map((s) => (
                  <span
                    key={s}
                    className="text-[8px] px-1.5 py-px rounded bg-accent text-muted-foreground"
                  >
                    {s}
                  </span>
                ))}
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
