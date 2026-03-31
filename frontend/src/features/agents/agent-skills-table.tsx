// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: AGPL-3.0-only
//
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { StatusDot } from "@/components/ui/status-dot";
import { cn } from "@/utils/cn";
import { SKILL_TYPE_STYLES, toStatus } from "./agents-data";
import type { AgentSkill } from "./agents-data";

interface AgentSkillsTableProps {
  skills: AgentSkill[];
}

export function AgentSkillsTable({ skills }: AgentSkillsTableProps) {
  return (
    <section className="bg-muted/40 border border-border rounded-lg overflow-hidden">
      <div className="px-4 py-3 border-b border-border flex items-center justify-between">
        <div className="flex items-center gap-2">
          <span className="text-amber-500">&#x26A1;</span>
          <span className="text-[13px] font-semibold text-foreground font-display">
            Skills & Tools
          </span>
          <span className="text-[9px] px-1.5 py-px bg-muted rounded text-muted-foreground">
            {skills.length}
          </span>
        </div>
        <Button variant="outline" size="xs">
          + Add Skill
        </Button>
      </div>

      <table className="w-full">
        <thead>
          <tr className="border-b border-border">
            {["Name", "Version", "Type", "Status", ""].map((h) => (
              <th
                key={h}
                className="px-4 py-2 text-left text-[9px] font-medium text-muted-foreground/50 uppercase tracking-wider font-display"
              >
                {h}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {skills.map((s) => {
            const style = SKILL_TYPE_STYLES[s.type];
            return (
              <tr
                key={s.name}
                className="border-b border-border/50 last:border-b-0 hover:bg-muted/30 transition-colors cursor-pointer"
              >
                <td className="px-4 py-2.5 text-xs font-medium text-foreground">
                  {s.name}
                </td>
                <td className="px-4 py-2.5">
                  <Badge>v{s.version}</Badge>
                </td>
                <td className="px-4 py-2.5">
                  <span
                    className={cn(
                      "text-[9px] px-1.5 py-0.5 rounded font-medium uppercase tracking-wide",
                      style?.bg,
                      style?.fg,
                    )}
                  >
                    {s.type}
                  </span>
                </td>
                <td className="px-4 py-2.5">
                  <span className="flex items-center gap-1.5 text-[11px]">
                    <StatusDot
                      status={toStatus(s.status)}
                    />
                    <span
                      className={cn(
                        s.status === "active"
                          ? "text-muted-foreground"
                          : "text-destructive",
                      )}
                    >
                      {s.status}
                    </span>
                  </span>
                </td>
                <td className="px-4 py-2.5 text-right">
                  <Button variant="ghost" size="icon-xs">
                    &#x22EF;
                  </Button>
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </section>
  );
}
