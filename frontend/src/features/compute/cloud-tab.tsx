// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: AGPL-3.0-only
//
import { useEffect, useState } from "react";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { StatusDot } from "@/components/ui/status-dot";
import { ConfirmDialog } from "@/components/ui/confirm-dialog";
import { PROVIDER_COLORS, SUPPORTED_PROVIDERS } from "./compute-data";
import { AddCloudCard } from "./add-cloud-card";
import { useAppStore } from "@/stores/app-store";
import { useNavigate } from "@tanstack/react-router";
import { Trash2 } from "lucide-react";

export function CloudTab() {
  const backends = useAppStore((s) => s.cloudBackends);
  const fetchBackends = useAppStore((s) => s.fetchCloudBackends);
  const cloudInstances = useAppStore((s) => s.cloudInstances);
  const fetchCloudInstances = useAppStore((s) => s.fetchCloudInstances);
  const terminateInstance = useAppStore((s) => s.terminateCloudInstance);
  const deleteBackend = useAppStore((s) => s.deleteCloudBackend);
  const activeProjectId = useAppStore((s) => s.activeProjectId);
  const navigate = useNavigate();
  const [deleteTarget, setDeleteTarget] = useState<string | null>(null);

  useEffect(() => { fetchBackends(); fetchCloudInstances(); }, [activeProjectId, fetchBackends, fetchCloudInstances]);

  const cloudOnly = cloudInstances.filter(i => i.provider !== "kubernetes");
  const cloudHourlyCost = cloudOnly.reduce((s, c) => s + c.cost_per_hour, 0);

  // Exclude kubernetes — that's the local cluster, not a cloud backend
  const cloudBackends = backends.filter(b => b.type !== "kubernetes");

  return (
    <div className="space-y-5 animate-in fade-in duration-200">
      {/* Registered backends */}
      {cloudBackends.length > 0 && (
        <div className="grid grid-cols-3 gap-3">
          {cloudBackends.map(b => {
            const info = SUPPORTED_PROVIDERS.find(p => p.key === b.type);
            const color = PROVIDER_COLORS[b.type] ?? "#888";
            return (
              <Card key={b.id} size="sm" className="p-4" style={{ borderColor: color + "30" }}>
                <div className="flex items-center justify-between mb-3">
                  <div className="flex items-center gap-2">
                    <div>
                      <div className="text-sm font-bold text-foreground font-display">
                        {info?.name ?? b.type}
                      </div>
                      <div className="text-[11px] text-faint">{info?.description ?? "Cloud backend"}</div>
                    </div>
                  </div>
                  <div className="flex items-center gap-2">
                    <StatusDot status="running" />
                    <Button
                      variant="ghost"
                      size="icon-xs"
                      className="text-faint hover:text-destructive"
                      onClick={() => setDeleteTarget(b.type)}
                    >
                      <Trash2 size={12} />
                    </Button>
                  </div>
                </div>
                <div className="grid grid-cols-2 gap-2 mb-3">
                  <div>
                    <div className="text-[10px] text-faint uppercase mb-0.5">Instances</div>
                    <div className="text-base font-bold text-foreground">{b.active_instances}</div>
                  </div>
                  <div>
                    <div className="text-[10px] text-faint uppercase mb-0.5">Cost</div>
                    <div className="text-base font-bold" style={{ color: b.hourly_cost > 0 ? "#F59E0B" : undefined }}>
                      ${b.hourly_cost.toFixed(2)}/hr
                    </div>
                  </div>
                </div>
                <Button
                  variant="outline"
                  size="xs"
                  className="w-full"
                  onClick={() => navigate({ to: "/studio/backend-offers", search: { backend: b.type } })}
                >
                  Available instances
                </Button>
              </Card>
            );
          })}
        </div>
      )}

      {cloudBackends.length === 0 && (
        <Card size="sm" className="p-6 text-center text-sm text-faint">
          No cloud backends connected. Add one below to start launching cloud GPU instances.
        </Card>
      )}

      {/* Active Cloud instances */}
      <Card size="sm" className="overflow-hidden">
        <div className="flex items-center justify-between px-4 py-2.5 border-b border-line">
          <span className="text-sm font-semibold text-foreground font-display">Active Cloud Instances</span>
          <span className="text-sm" style={{ color: "#F59E0B" }}>Total: ${cloudHourlyCost.toFixed(2)}/hr</span>
        </div>
        {cloudOnly.length === 0 && (
          <div className="px-4 py-6 text-center text-sm text-faint">No active cloud instances</div>
        )}
        {cloudOnly.map(inst => {
          const isRunning = inst.status === "idle" || inst.status === "busy";
          return (
            <div key={inst.id} className="px-4 py-3.5 border-b border-line last:border-0 flex items-start justify-between">
              <div>
                <div className="flex items-center gap-2 mb-1">
                  <StatusDot status={isRunning ? "running" : "deploying"} />
                  <span className="text-sm font-semibold text-foreground">{inst.workload || inst.name}</span>
                  {inst.status === "provisioning" && (
                    <span className="text-[10px] px-1.5 py-px rounded bg-[#06B6D412] text-[#06B6D4] animate-pulse">PROVISIONING</span>
                  )}
                  {inst.status === "pending" && (
                    <span className="text-[10px] px-1.5 py-px rounded bg-[#F59E0B12] text-[#F59E0B] animate-pulse">PENDING</span>
                  )}
                </div>
                <div className="flex gap-3 text-[11px] text-faint">
                  <span style={{ color: PROVIDER_COLORS[inst.provider] }}>{inst.provider.toUpperCase()} · {inst.region}</span>
                  {inst.instance_type && <span>{inst.instance_type}</span>}
                  {inst.gpu && <span>{inst.gpu}</span>}
                  {inst.spot_instance && <span className="text-success">Spot</span>}
                  {inst.started_at && <span>started {new Date(inst.started_at).toLocaleString()}</span>}
                </div>
              </div>
              <div className="text-right shrink-0">
                <div className="text-lg font-bold" style={{ color: "#F59E0B" }}>${inst.cost_per_hour}/hr</div>
                {inst.estimated_total > 0 && <div className="text-[11px] text-faint">est: ${inst.estimated_total.toFixed(0)} total</div>}
                <Button
                  variant="outline"
                  size="xs"
                  className="mt-1.5 text-destructive border-destructive/30"
                  onClick={() => terminateInstance(inst.id, inst.project_name)}
                >
                  Terminate
                </Button>
              </div>
            </div>
          );
        })}
      </Card>

      {/* Supported cloud providers */}
      <AddCloudCard />

      <ConfirmDialog
        open={deleteTarget !== null}
        title="Remove cloud backend"
        description={`This will disconnect the ${deleteTarget?.toUpperCase()} backend from this project and terminate all running instances on it.`}
        confirmLabel="Remove"
        variant="destructive"
        onConfirm={async () => {
          if (deleteTarget) await deleteBackend(deleteTarget);
          setDeleteTarget(null);
        }}
        onCancel={() => setDeleteTarget(null)}
      />

    </div>
  );
}
