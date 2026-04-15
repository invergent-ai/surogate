// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: AGPL-3.0-only
//
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { StatusDot } from "@/components/ui/status-dot";
import { Tabs, TabsList, TabsTrigger, TabsContent } from "@/components/ui/tabs";
import { cn } from "@/utils/cn";
import { toStatus } from "./agents-data";
import { AgentSkillsTable } from "./agent-skills-table";
import { AgentMcpServers } from "./agent-mcp-servers";
import type { Agent } from "./agents-data";
import { OverviewTab } from "./overview-tab";
import { MetricsTab } from "./metrics-tab";
import { VersionsTab } from "./versions-tab";
import { ConfigTab } from "./config-tab";
import { UsersTab } from "./users-tab";
import { useAppStore } from "@/stores/app-store";

export function AgentDetail({
  agent,
  onScale,
  onDelete,
}: {
  agent: Agent;
  onScale: () => void;
  onDelete: () => void;
}) {
  const projectColor =
    useAppStore((s) => s.projects.find((p) => p.id === agent.projectId)?.color) ?? "#F59E0B";
  const startAgent = useAppStore((s) => s.startAgent);
  const stopAgent = useAppStore((s) => s.stopAgent);
  const canStart = agent.status !== "running" && agent.status !== "deploying";
  const canStop = agent.status === "running";
  return (
    <div className="flex-1 flex flex-col overflow-hidden">
      {/* Header */}
      <div className="px-6 py-4 border-b border-border shrink-0">
        <div className="flex items-start justify-between mb-3">
          <div className="flex items-start gap-3.5">
            <div
              className="w-11 h-11 rounded-[10px] shrink-0 flex items-center justify-center text-xl border"
              style={{
                backgroundColor: `${projectColor}12`,
                borderColor: `${projectColor}30`,
                color: projectColor,
              }}
            >
              &#x2B21;
            </div>
            <div>
              <div className="flex items-center gap-2 mb-0.5">
                <h2 className="text-base font-bold text-foreground font-display tracking-tight">
                  {agent.displayName}
                </h2>
                <Badge>v{agent.version}</Badge>
                <span className="flex items-center gap-1 text-[10px]">
                  <StatusDot status={toStatus(agent.status)} />
                  <span
                    className={cn(
                      "font-medium",
                      agent.status === "error"
                        ? "text-destructive"
                        : agent.status === "deploying"
                          ? "text-amber-500"
                          : "text-green-500",
                    )}
                  >
                    {agent.status}
                  </span>
                </span>
              </div>
              <p className="text-[11px] text-muted-foreground max-w-[540px] leading-snug">
                {agent.description}
              </p>
              <div className="flex gap-3 mt-1.5 text-[10px] text-muted-foreground/40">
                {agent.endpoint && agent.endpoint !== "\u2014" && (
                  <span>
                    url:{" "}
                    <a
                      href={agent.endpoint}
                      target="_blank"
                      rel="noreferrer"
                      className="text-primary hover:underline"
                    >
                      {agent.endpoint.replace(/^https?:\/\//, "")}
                    </a>
                  </span>
                )}
                <span>
                  model:{" "}
                  <span className="text-muted-foreground">{agent.model}</span>
                </span>
                <span>
                  by:{" "}
                  <span className="text-muted-foreground">
                    {agent.createdBy}
                  </span>
                </span>
                <span>
                  deployed:{" "}
                  <span className="text-muted-foreground">
                    {agent.lastDeployed}
                  </span>
                </span>
              </div>
            </div>
          </div>
          <div className="flex gap-1.5 shrink-0">
            {canStart && (
              <Button size="xs" onClick={() => startAgent(agent.id)}>
                Start
              </Button>
            )}
            {canStop && (
              <Button variant="outline" size="xs" onClick={() => stopAgent(agent.id)}>
                Stop
              </Button>
            )}
            {agent.status === "running" && (
              <Button variant="outline" size="xs" onClick={onScale}>
                Scale
              </Button>
            )}
            {agent.status === "error" && (
              <Button variant="destructive" size="xs">
                Restart
              </Button>
            )}
            <Button variant="outline" size="xs">
              Logs
            </Button>
            <Button variant="destructive" size="xs" onClick={onDelete}>
              Delete
            </Button>
          </div>
        </div>
      </div>

      {/* Tabs */}
      <Tabs
        defaultValue="overview"
        className="flex-1 flex flex-col overflow-hidden"
      >
        <div className="px-6 border-b border-border shrink-0">
          <TabsList variant="line">
            <TabsTrigger value="overview">Overview</TabsTrigger>
            <TabsTrigger value="skills">Skills & MCP</TabsTrigger>
            <TabsTrigger value="users">Users</TabsTrigger>
            <TabsTrigger value="config">Config</TabsTrigger>
            <TabsTrigger value="versions">Versions</TabsTrigger>
            <TabsTrigger value="metrics">Metrics</TabsTrigger>
          </TabsList>
        </div>

        <div className="flex-1 overflow-y-auto px-6 py-5">
          <TabsContent value="overview" className="mt-0">
            <OverviewTab agent={agent} />
          </TabsContent>
          <TabsContent value="skills" className="mt-0 space-y-4">
            <AgentSkillsTable skills={agent.skills} />
            <AgentMcpServers servers={agent.mcpServers} />
          </TabsContent>
          <TabsContent value="users" className="mt-0">
            <UsersTab agentId={agent.id} />
          </TabsContent>
          <TabsContent value="config" className="mt-0">
            <ConfigTab agent={agent} />
          </TabsContent>
          <TabsContent value="versions" className="mt-0">
            <VersionsTab agent={agent} />
          </TabsContent>
          <TabsContent value="metrics" className="mt-0">
            <MetricsTab agent={agent} />
          </TabsContent>
        </div>
      </Tabs>
    </div>
  );
}

// ── Empty state ────────────────────────────────────────────────

export function AgentEmptyState() {
  return (
    <div className="flex-1 flex items-center justify-center text-muted-foreground/40">
      <div className="text-center">
        <div className="text-3xl mb-2">&#x2B21;</div>
        <div className="font-display text-sm">
          Select an agent to view details
        </div>
        <div className="text-[10px] mt-1 max-w-[300px] leading-relaxed text-muted-foreground/30">
          Agents are deployed AI services with skills, tools, and MCP server
          connections.
        </div>
      </div>
    </div>
  );
}
