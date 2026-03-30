// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: AGPL-3.0-only
//
import { Outlet, useLocation, useNavigate } from "@tanstack/react-router";
import { PageHeader } from "@/components/page-header";
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Button } from "@/components/ui/button";
import { Cloud, Plus } from "lucide-react";
import { LOCAL_NODES, CLOUD_INSTANCES } from "./compute-data";

const TAB_ROUTES: Record<string, string> = {
  overview: "/studio/compute",
  nodes: "/studio/compute/cluster-nodes",
  cloud: "/studio/compute/cloud",
  queue: "/studio/compute/workload-queue",
  costs: "/studio/compute/costs",
  policies: "/studio/compute/policies",
};

const ROUTE_TO_TAB = Object.fromEntries(
  Object.entries(TAB_ROUTES).map(([k, v]) => [v, k]),
);

export function ComputePage() {
  const { pathname } = useLocation();
  const navigate = useNavigate();
  const activeTab = ROUTE_TO_TAB[pathname.replace(/\/$/, "")] ?? "overview";

  const usedGpu = LOCAL_NODES.reduce((s, n) => s + (n.gpu?.used || 0), 0);
  const totalGpu = LOCAL_NODES.reduce((s, n) => s + (n.gpu?.count || 0), 0);
  const runningCloud = CLOUD_INSTANCES.filter((c) => c.status === "running").length;
  const cloudCost = CLOUD_INSTANCES.filter((c) => c.status === "running").reduce((s, c) => s + c.costPerHour, 0);

  return (
    <div className="flex-1 flex flex-col overflow-hidden bg-background">
      <PageHeader
        title="Compute"
        subtitle={`${usedGpu}/${totalGpu} local GPUs · ${runningCloud} cloud instances · $${cloudCost.toFixed(0)}/hr cloud`}
      />

      <Tabs value={activeTab} onValueChange={(v) => navigate({ to: TAB_ROUTES[v] })} className="flex-1 flex flex-col overflow-hidden">
        <div className="px-7 border-b border-line bg-card shrink-0 flex items-center justify-between">
          <TabsList variant="line">
            <TabsTrigger value="overview">Overview</TabsTrigger>
            <TabsTrigger value="nodes">Cluster Nodes</TabsTrigger>
            <TabsTrigger value="cloud">Cloud / SkyPilot</TabsTrigger>
            <TabsTrigger value="queue">Workload Queue</TabsTrigger>
            <TabsTrigger value="costs">Costs</TabsTrigger>
            <TabsTrigger value="policies">Policies</TabsTrigger>
          </TabsList>
          <div className="flex items-center gap-2">
            <Button variant="outline" size="sm">
              <Plus size={14} />
              Connect Cloud
            </Button>
            <Button size="sm">
              <Cloud size={14} />
              Launch Instance
            </Button>
          </div>
        </div>

        <div className="flex-1 overflow-y-auto px-7 py-5 pb-10">
          <Outlet />
        </div>
      </Tabs>
    </div>
  );
}
