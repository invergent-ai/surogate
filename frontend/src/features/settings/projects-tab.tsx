// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: AGPL-3.0-only
//
import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { cn } from "@/utils/cn";
import { PROJECTS, type Project } from "./settings-data";

function MemberRow({ member }: { member: Project["members"][number] }) {
  const roleVariant =
    member.role === "Owner"
      ? "active"
      : member.role === "Editor"
        ? "default"
        : "default";

  return (
    <div className="grid grid-cols-[1fr_1fr_80px_80px_50px] items-center px-4 py-2 border-t border-line hover:bg-accent/50 transition-colors cursor-pointer">
      <div className="flex items-center gap-2">
        <div
          className="w-6 h-6 rounded-full flex items-center justify-center text-[8px] font-bold text-white shrink-0"
          style={{ backgroundColor: member.color }}
        >
          {member.avatar}
        </div>
        <span className="text-[11px] text-foreground font-medium">
          {member.name}
        </span>
      </div>
      <span className="text-[10px] text-muted-foreground">{member.email}</span>
      <Badge
        variant={roleVariant}
        className={cn(
          "text-center justify-center",
          member.role === "Owner" && "bg-amber-500/10 text-amber-500",
          member.role === "Editor" && "bg-blue-500/10 text-blue-500",
        )}
      >
        {member.role}
      </Badge>
      <div className="flex items-center gap-1 text-[9px]">
        <span
          className="inline-block w-1 h-1 rounded-full shrink-0"
          style={{
            backgroundColor:
              member.status === "online" ? "#22C55E" : "var(--muted-foreground)",
            boxShadow:
              member.status === "online" ? "0 0 6px #22C55E" : "none",
          }}
        />
        <span
          style={{
            color:
              member.status === "online" ? "#22C55E" : "var(--muted-foreground)",
          }}
        >
          {member.lastActive}
        </span>
      </div>
      <Button variant="outline" size="xs" className="justify-self-end">
        Edit
      </Button>
    </div>
  );
}

function ProjectCard({ project }: { project: Project }) {
  const [expanded, setExpanded] = useState(project.id === "prod-cx");

  return (
    <div
      className="bg-card border border-line rounded-lg overflow-hidden"
      style={{ borderLeftWidth: 3, borderLeftColor: project.color }}
    >
      {/* Header */}
      <div
        onClick={() => setExpanded(!expanded)}
        className="px-4 py-3.5 cursor-pointer hover:bg-accent/50 transition-colors"
      >
        <div className="flex items-center justify-between mb-1">
          <div className="flex items-center gap-2">
            <span
              className={cn(
                "text-[10px] transition-transform",
                expanded ? "text-amber-500" : "text-muted-foreground",
              )}
            >
              {expanded ? "▾" : "▸"}
            </span>
            <span className="font-display text-sm font-bold text-foreground">
              {project.name}
            </span>
            <span
              className="inline-block w-1.5 h-1.5 rounded-full"
              style={{
                backgroundColor: "#22C55E",
                boxShadow: "0 0 6px #22C55E",
              }}
            />
          </div>
          <Button
            variant="outline"
            size="icon-xs"
            onClick={(e) => e.stopPropagation()}
          >
            <span className="text-[10px]">&#x22EF;</span>
          </Button>
        </div>
        <div className="flex gap-4 text-[10px] text-faint pl-[18px]">
          <span>
            ns: <span className="text-muted-foreground">{project.namespace}</span>
          </span>
          <span>{project.members.length} members</span>
          <span>{project.agents} agents</span>
          <span>{project.models} models</span>
        </div>
      </div>

      {/* Members */}
      {expanded && (
        <div className="border-t border-line">
          <div className="px-4 py-2 flex justify-between items-center bg-card">
            <span className="text-[9px] font-semibold text-faint tracking-[0.1em] uppercase font-display">
              Team Members
            </span>
            <button
              type="button"
              className="bg-transparent border-none cursor-pointer text-[9px] font-display font-semibold"
              style={{ color: project.color }}
            >
              + Invite
            </button>
          </div>
          {project.members.map((m) => (
            <MemberRow key={m.email} member={m} />
          ))}
        </div>
      )}
    </div>
  );
}

export function ProjectsTab() {
  return (
    <div>
      <div className="flex justify-between items-center mb-5">
        <h2 className="font-display text-[15px] font-bold text-foreground">
          Projects & Teams
        </h2>
        <Button size="sm">+ New Project</Button>
      </div>
      <p className="text-[11px] text-muted-foreground mb-4">
        Each project maps to a Kubernetes namespace for resource isolation. Team
        members are managed per project with role-based access (Owner, Editor,
        Viewer).
      </p>
      <div className="flex flex-col gap-2.5">
        {PROJECTS.map((p) => (
          <ProjectCard key={p.id} project={p} />
        ))}
      </div>
    </div>
  );
}
