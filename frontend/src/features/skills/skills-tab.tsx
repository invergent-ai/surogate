// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: AGPL-3.0-only
//
import { useState } from "react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { StatusDot } from "@/components/ui/status-dot";
import { Tabs, TabsList, TabsTrigger, TabsContent } from "@/components/ui/tabs";
import { cn } from "@/utils/cn";
import { SKILLS, toStatus } from "./skills-data";
import type { Skill } from "./skills-data";

function SkillListItem({
  skill,
  selected,
  onSelect,
}: {
  skill: Skill;
  selected: boolean;
  onSelect: () => void;
}) {
  return (
    <button
      onClick={onSelect}
      className={cn(
        "w-full text-left px-4 py-3 border-l-2 transition-colors cursor-pointer",
        selected
          ? "bg-muted/60 border-l-amber-500"
          : "border-l-transparent hover:bg-muted/30",
      )}
    >
      <div className="flex items-center justify-between mb-1">
        <div className="flex items-center gap-2">
          <span className="text-amber-500 text-sm">&#x1F4C4;</span>
          <span className="text-xs font-semibold text-foreground font-display">
            {skill.displayName}
          </span>
          <Badge>v{skill.version}</Badge>
        </div>
        <StatusDot status={toStatus(skill.status)} />
      </div>
      <p className="text-[10px] text-muted-foreground leading-snug line-clamp-1 mb-1.5">
        {skill.description}
      </p>
      <div className="flex items-center gap-2 text-[9px] text-muted-foreground/50">
        <span>&#x2B21; {skill.agent}</span>
        <span>&middot;</span>
        <span>{skill.author}</span>
        <span>&middot;</span>
        <span>{skill.updatedAt}</span>
      </div>
    </button>
  );
}

function SkillDetail({ skill, onClose }: { skill: Skill; onClose: () => void }) {
  return (
    <div className="flex-1 flex flex-col overflow-hidden animate-in fade-in duration-150">
      {/* Header */}
      <div className="px-6 py-4 border-b border-border shrink-0">
        <div className="flex items-center justify-between mb-1.5">
          <div className="flex items-center gap-2">
            <span className="text-xl text-amber-500">&#x1F4C4;</span>
            <div>
              <div className="flex items-center gap-2">
                <span className="text-base font-bold text-foreground font-display">
                  {skill.displayName}
                </span>
                <Badge>v{skill.version}</Badge>
                <StatusDot status={toStatus(skill.status)} />
              </div>
              <div className="text-[10px] text-muted-foreground mt-0.5">
                Agent: <span className="text-foreground/70">{skill.agent}</span> &middot;{" "}
                {skill.author} &middot; {skill.updatedAt}
              </div>
            </div>
          </div>
          <div className="flex gap-1.5">
            <Button variant="outline" size="xs">Edit</Button>
            <Button variant="outline" size="xs">Publish</Button>
            <Button variant="ghost" size="icon-xs" onClick={onClose}>
              &#x2715;
            </Button>
          </div>
        </div>
        <div className="flex flex-wrap gap-1 mt-1.5">
          {skill.tags.map((t) => (
            <Badge key={t}>{t}</Badge>
          ))}
        </div>
      </div>

      {/* Sub-tabs */}
      <Tabs defaultValue="content" className="flex-1 flex flex-col overflow-hidden">
        <div className="px-6 border-b border-border shrink-0">
          <TabsList variant="line">
            <TabsTrigger value="content">Content</TabsTrigger>
            <TabsTrigger value="versions">Versions</TabsTrigger>
          </TabsList>
        </div>

        <TabsContent value="content" className="flex-1 overflow-y-auto p-6">
          <pre className="bg-muted/40 border border-border rounded-lg p-5 text-xs text-foreground/80 leading-relaxed whitespace-pre-wrap font-mono">
            {skill.content}
          </pre>
        </TabsContent>

        <TabsContent value="versions" className="flex-1 overflow-y-auto p-6">
          {skill.versions.map((v, i) => (
            <div key={v.version} className="flex gap-3 mb-1">
              <div className="flex flex-col items-center w-5">
                <div
                  className={cn(
                    "w-2.5 h-2.5 rounded-full shrink-0",
                    v.status === "active"
                      ? "bg-green-500"
                      : "bg-muted border border-muted-foreground/30",
                  )}
                />
                {i < skill.versions.length - 1 && (
                  <div className="w-px flex-1 bg-border" />
                )}
              </div>
              <div className="pb-4 flex-1">
                <div className="flex items-center gap-1.5 mb-0.5">
                  <code
                    className={cn(
                      "text-[11px] font-semibold",
                      v.status === "active" ? "text-foreground" : "text-muted-foreground",
                    )}
                  >
                    v{v.version}
                  </code>
                  {v.status === "active" && (
                    <Badge variant="active">LIVE</Badge>
                  )}
                </div>
                <div className="text-[10px] text-muted-foreground mb-0.5">
                  {v.change}
                </div>
                <div className="text-[9px] text-muted-foreground/50">
                  {v.author} &middot; {v.date}
                </div>
              </div>
            </div>
          ))}
        </TabsContent>
      </Tabs>
    </div>
  );
}

function SkillEmptyState() {
  return (
    <div className="flex-1 flex items-center justify-center text-muted-foreground/40">
      <div className="text-center">
        <div className="text-3xl mb-2">&#x1F4C4;</div>
        <div className="font-display text-sm">Select a skill to view its content</div>
        <div className="text-[10px] mt-1 max-w-[300px] leading-relaxed text-muted-foreground/30">
          Skills are markdown files that define agent capabilities, workflows,
          escalation rules, and behavioral guidelines.
        </div>
      </div>
    </div>
  );
}

export function SkillsTab() {
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const selected = selectedId ? SKILLS.find((s) => s.id === selectedId) ?? null : null;

  return (
    <div className="flex-1 flex overflow-hidden">
      {/* List */}
      <div className="w-[420px] min-w-[420px] border-r border-border flex flex-col">
        <div className="px-4 py-3 border-b border-border flex justify-between items-center">
          <span className="text-[10px] text-muted-foreground">
            {SKILLS.length} agent skill files
          </span>
          <Button size="xs">+ New Skill</Button>
        </div>
        <div className="flex-1 overflow-y-auto">
          {SKILLS.map((s) => (
            <SkillListItem
              key={s.id}
              skill={s}
              selected={selectedId === s.id}
              onSelect={() => setSelectedId(s.id)}
            />
          ))}
        </div>
      </div>

      {/* Detail / Empty */}
      {selected ? (
        <SkillDetail skill={selected} onClose={() => setSelectedId(null)} />
      ) : (
        <SkillEmptyState />
      )}
    </div>
  );
}
