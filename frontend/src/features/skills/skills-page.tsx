// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: AGPL-3.0-only
//
import { Outlet, useLocation, useNavigate } from "@tanstack/react-router";
import { PageHeader } from "@/components/page-header";
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { SKILLS, TOOLS, MCP_SERVERS } from "./skills-data";

const TAB_ROUTES: Record<string, string> = {
  skills: "/studio/skills",
  tools: "/studio/skills/tools",
  mcp: "/studio/skills/mcp",
};

const ROUTE_TO_TAB = Object.fromEntries(
  Object.entries(TAB_ROUTES).map(([k, v]) => [v, k]),
);

const connectedMcp = MCP_SERVERS.filter((m) => m.status === "connected").length;

export function SkillsPage() {
  const { pathname } = useLocation();
  const navigate = useNavigate();
  const activeTab = ROUTE_TO_TAB[pathname.replace(/\/$/, "")] ?? "skills";

  return (
    <div className="flex-1 flex flex-col overflow-hidden bg-background">
      <PageHeader
        title="Skills & Tools"
        subtitle={
          <>
            {SKILLS.length} skills &middot; {TOOLS.length} tools &middot;{" "}
            {connectedMcp} MCP servers connected
          </>
        }
      />
      <Tabs
        value={activeTab}
        onValueChange={(v) => navigate({ to: TAB_ROUTES[v] })}
        className="flex-1 flex flex-col overflow-hidden"
      >
        <div className="px-7 border-b border-border shrink-0">
          <TabsList variant="line">
            <TabsTrigger value="skills">
              Skills
              <span className="text-[9px] ml-1 opacity-50">{SKILLS.length}</span>
            </TabsTrigger>
            <TabsTrigger value="tools">
              Tools
              <span className="text-[9px] ml-1 opacity-50">{TOOLS.length}</span>
            </TabsTrigger>
            <TabsTrigger value="mcp">
              MCP Servers
              <span className="text-[9px] ml-1 opacity-50">{MCP_SERVERS.length}</span>
            </TabsTrigger>
          </TabsList>
        </div>

        <div className="flex-1 overflow-hidden flex">
          <Outlet />
        </div>
      </Tabs>
    </div>
  );
}
