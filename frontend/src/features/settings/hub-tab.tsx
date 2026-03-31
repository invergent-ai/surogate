// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: AGPL-3.0-only
//
import { ProgressBar } from "@/components/ui/progress-bar";
import { HUB_CONFIG } from "./settings-data";

const STATS = [
  { label: "Repositories", value: HUB_CONFIG.repositories, color: "#22C55E" },
  { label: "Storage Used", value: HUB_CONFIG.storageUsed, color: "#3B82F6" },
  { label: "Storage Limit", value: HUB_CONFIG.storageLimit, color: undefined },
];

const ENDPOINTS = [
  {
    label: "Git Server",
    value: HUB_CONFIG.gitUrl,
    desc: "Clone, push, pull repositories",
  },
  {
    label: "S3 Endpoint",
    value: HUB_CONFIG.s3Endpoint,
    desc: "Large file storage (model weights, datasets)",
  },
  {
    label: "API Server",
    value: HUB_CONFIG.apiUrl,
    desc: "Metadata, search, repository management",
  },
];

export function HubTab() {
  return (
    <div>
      <h2 className="font-display text-[15px] font-bold text-foreground mb-5">
        Data Hub Configuration
      </h2>
      <p className="text-[11px] text-muted-foreground mb-4">
        The Data Hub is a platform service running in the K8s cluster. It
        provides a Git server for version control, an S3-compatible server for
        large file access, and an API server for metadata and search.
      </p>

      {/* Stat cards */}
      <div className="grid grid-cols-3 gap-2.5 mb-5">
        {STATS.map((m) => (
          <div
            key={m.label}
            className="bg-card border border-line rounded-lg px-3.5 py-3"
          >
            <div className="text-[8px] text-muted-foreground uppercase tracking-[0.06em] font-display mb-0.5">
              {m.label}
            </div>
            <div className="text-lg font-bold text-foreground">{m.value}</div>
          </div>
        ))}
      </div>

      {/* Endpoints */}
      <div className="font-display text-[11px] font-semibold text-foreground mb-2">
        Endpoints
      </div>
      <div className="bg-card border border-line rounded-lg overflow-hidden mb-5">
        {ENDPOINTS.map((e) => (
          <div
            key={e.label}
            className="px-4 py-3 border-b border-line last:border-b-0"
          >
            <div className="flex justify-between items-center mb-0.5">
              <span className="text-[11px] font-medium text-foreground/70">
                {e.label}
              </span>
              <code className="text-[10px] text-green-500 bg-accent px-1.5 py-px rounded">
                {e.value}
              </code>
            </div>
            <div className="text-[9px] text-faint">{e.desc}</div>
          </div>
        ))}
      </div>

      {/* Storage */}
      <div className="font-display text-[11px] font-semibold text-foreground mb-2">
        Storage
      </div>
      <div className="bg-card border border-line rounded-lg px-4 py-3.5">
        <div className="flex justify-between text-[10px] text-muted-foreground mb-1.5">
          <span>{HUB_CONFIG.storageUsed} used</span>
          <span>{HUB_CONFIG.storageLimit} limit</span>
        </div>
        <ProgressBar value={24} color="#22C55E" />
        <div className="flex justify-between text-[9px] text-faint mt-1.5">
          <span>Git LFS: {HUB_CONFIG.lfsEnabled ? "enabled" : "disabled"}</span>
          <span>24% used</span>
        </div>
      </div>
    </div>
  );
}
