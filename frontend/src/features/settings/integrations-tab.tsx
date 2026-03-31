// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: AGPL-3.0-only
//
import { Button } from "@/components/ui/button";
import { INTEGRATIONS } from "./settings-data";

export function IntegrationsTab() {
  return (
    <div>
      <h2 className="font-display text-[15px] font-bold text-foreground mb-5">
        Integrations
      </h2>
      <div className="flex flex-col gap-2">
        {INTEGRATIONS.map((intg) => (
          <div
            key={intg.id}
            className="bg-card border border-line rounded-lg px-4 py-4"
          >
            <div className="flex items-center justify-between mb-1.5">
              <div className="flex items-center gap-2.5">
                <span className="w-9 h-9 rounded-lg bg-accent flex items-center justify-center text-[11px] font-bold text-muted-foreground shrink-0">
                  {intg.icon}
                </span>
                <div>
                  <div className="flex items-center gap-1.5">
                    <span className="font-display text-[13px] font-semibold text-foreground">
                      {intg.name}
                    </span>
                    <span
                      className="inline-block w-1.5 h-1.5 rounded-full"
                      style={{
                        backgroundColor:
                          intg.status === "connected"
                            ? "#22C55E"
                            : "var(--muted-foreground)",
                        boxShadow:
                          intg.status === "connected"
                            ? "0 0 6px #22C55E"
                            : "none",
                      }}
                    />
                    <span
                      className="text-[9px]"
                      style={{
                        color:
                          intg.status === "connected"
                            ? "#22C55E"
                            : "var(--muted-foreground)",
                      }}
                    >
                      {intg.status}
                    </span>
                  </div>
                  <div className="text-[10px] text-muted-foreground mt-px">
                    {intg.description}
                  </div>
                </div>
              </div>
              <Button
                variant={
                  intg.status === "connected" ? "destructive" : "outline"
                }
                size="xs"
              >
                {intg.status === "connected" ? "Disconnect" : "Connect"}
              </Button>
            </div>

            {intg.status === "connected" &&
              Object.keys(intg.config).length > 0 && (
                <div className="flex gap-3 text-[9px] text-faint mt-1 pl-[46px]">
                  {Object.entries(intg.config).map(([k, v]) => (
                    <span key={k}>
                      {k}:{" "}
                      <code className="text-muted-foreground">{v}</code>
                    </span>
                  ))}
                </div>
              )}
          </div>
        ))}
      </div>
    </div>
  );
}
