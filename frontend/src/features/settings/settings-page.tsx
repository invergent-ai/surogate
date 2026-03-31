// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: AGPL-3.0-only
//
import { Outlet, useLocation, useNavigate } from "@tanstack/react-router";
import { PageHeader } from "@/components/page-header";
import { cn } from "@/utils/cn";

const TABS = [
  { id: "profile", path: "/studio/settings", label: "Profile", icon: "◉" },
  { id: "projects", path: "/studio/settings/projects", label: "Projects & Teams", icon: "⊞" },
  { id: "api-keys", path: "/studio/settings/api-keys", label: "API Keys", icon: "⚿" },
  { id: "hub", path: "/studio/settings/hub", label: "Data Hub", icon: "⊕" },
  { id: "integrations", path: "/studio/settings/integrations", label: "Integrations", icon: "⧉" },
  { id: "notifications", path: "/studio/settings/notifications", label: "Notifications", icon: "◈" },
] as const;

export function SettingsPage() {
  const { pathname } = useLocation();
  const navigate = useNavigate();
  const normalized = pathname.replace(/\/$/, "");

  return (
    <div className="flex flex-col h-screen overflow-hidden">
      <PageHeader
        title="Settings"
        subtitle="Platform configuration, team, and integrations"
      />

      <div className="flex flex-1 overflow-hidden">
        {/* Left: Tab navigation */}
        <div className="w-55 min-w-55 border-r border-line bg-card py-3.5 px-3">
          {TABS.map((t) => {
            const isActive =
              t.path === "/studio/settings"
                ? normalized === "/studio/settings"
                : normalized.startsWith(t.path);

            return (
              <button
                key={t.id}
                type="button"
                onClick={() => navigate({ to: t.path })}
                className={cn(
                  "flex items-center gap-2 w-full border-none cursor-pointer px-2.5 py-2 my-px rounded-md font-display text-[11px] transition-all duration-150",
                  isActive
                    ? "bg-line text-foreground font-semibold"
                    : "bg-transparent text-muted-foreground hover:bg-accent hover:text-foreground",
                )}
              >
                <span className="w-4.5 text-center text-xs shrink-0">
                  {t.icon}
                </span>
                <span>{t.label}</span>
              </button>
            );
          })}
        </div>

        {/* Right: Content */}
        <div className="flex-1 overflow-y-auto px-8 py-6">
          <div className="max-w-[760px]">
            <Outlet />
          </div>
        </div>
      </div>
    </div>
  );
}
