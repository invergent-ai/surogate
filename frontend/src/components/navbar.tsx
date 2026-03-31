// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: AGPL-3.0-only
//
import { useState } from "react";
import { Link, useRouterState } from "@tanstack/react-router";
import { useAppStore } from "@/stores/app-store";
import { cn } from "@/utils/cn";

const NAV_SECTIONS = [
  {
    label: "OPERATE",
    items: [
      { path: "/studio", icon: "◫", label: "Dashboard", badge: null },
      { path: "/studio/agents", icon: "⬡", label: "Agents", badge: "12" },
      { path: "/studio/models", icon: "◇", label: "Models", badge: "4" },
      { path: "/studio/skills", icon: "⚡", label: "Skills & Tools", badge: "38" },
      { path: "/studio/compute", icon: "☁", label: "Compute", badge: null },
    ],
  },
  {
    label: "OBSERVE",
    items: [
      { path: "/studio/monitoring", icon: "◉", label: "Monitoring", badge: null },
      { path: "/studio/conversations", icon: "⊡", label: "Conversations", badge: "2.4k" },
      { path: "/studio/evaluations", icon: "◈", label: "Evaluations", badge: null },
      { path: "/studio/playground", icon: "▷", label: "Playground", badge: null },
    ],
  },
  {
    label: "TRAIN",
    items: [
      { path: "/studio/datasets", icon: "▤", label: "Datasets", badge: "9" },
      { path: "/studio/training", icon: "◬", label: "Training", badge: "2" },
    ],
  },
  {
    label: "REGISTRY",
    items: [{ path: "/studio/hub", icon: "⊕", label: "Data Hub", badge: null }],
  },
] as const;

export function Navbar() {
  const [collapsed, setCollapsed] = useState(false);
  const pathname = useRouterState({ select: (s) => s.location.pathname });
  const projects = useAppStore((s) => s.projects);
  const activeProjectId = useAppStore((s) => s.activeProjectId);
  const project = projects.find((p) => p.id === activeProjectId);

  return (
    <nav
      className={cn(
        "bg-card border-r border-line flex flex-col overflow-hidden z-10 transition-all duration-200",
        collapsed ? "w-14 min-w-14" : "w-[232px] min-w-[232px]",
      )}
    >
      {/* logo */}
      <div
        className={cn(
          "flex items-center gap-2.5 border-b border-line min-h-14",
          collapsed ? "justify-center py-4" : "px-4 py-4",
        )}
      >
        <img
          src="/login.svg"
          alt="Surogate"
          className="w-7 h-7 rounded-md shrink-0"
        />
        {!collapsed && (
          <div>
            <div className="font-display font-bold text-foreground tracking-tight">
              Studio
            </div>
            <div className="text-[9px] text-muted-foreground tracking-[0.08em] uppercase">
              Agent Platform
            </div>
          </div>
        )}
      </div>

      {/* project selector */}
      {!collapsed && (
        <div className="px-3 py-2.5">
          <div className="bg-input border border-border rounded-md px-2.5 py-1 text-sm flex items-center gap-2 cursor-pointer">
            <div
              className="w-2 h-2 rounded-full shrink-0"
              style={{ background: project?.color }}
            />
            <span className="text-subtle flex-1 truncate font-semibold">
              {project?.name}
            </span>
          </div>
        </div>
      )}

      {/* nav groups */}
      <div
        className={cn(
          "flex-1 overflow-y-auto",
          collapsed ? "py-2" : "py-1",
        )}
      >
        {NAV_SECTIONS.map((section) => (
          <div key={section.label} className="mb-1">
            {!collapsed && (
              <div className="px-4 pt-2.5 pb-1 text-[9px] font-semibold text-faint tracking-[0.12em] uppercase">
                {section.label}
              </div>
            )}
            {collapsed && <div className="border-b border-line mx-3 my-1" />}
            {section.items.map((item) => {
              const isActive =
                item.path === "/studio"
                  ? pathname === "/studio" || pathname === "/studio/"
                  : pathname.startsWith(item.path);
              return (
                <Link
                  key={item.path}
                  to={item.path}
                  className={cn(
                    "flex items-center gap-2 w-full no-underline cursor-pointer font-display transition-all duration-150",
                    collapsed
                      ? "justify-center py-2"
                      : "px-3 py-1.5 my-px",
                    isActive
                      ? "bg-line text-primary"
                      : "bg-transparent text-subtle hover:bg-input hover:text-foreground",
                    !collapsed &&
                      (isActive
                        ? "border-l-2 border-l-primary"
                        : "border-l-2 border-l-transparent"),
                  )}
                >
                  <span className="w-5 text-center shrink-0">
                    {item.icon}
                  </span>
                  {!collapsed && (
                    <>
                      <span className="flex-1 text-left">{item.label}</span>
                      {item.badge && (
                        <span
                          className={cn(
                            "text-[9px] px-[5px] py-px rounded",
                            isActive
                              ? "bg-primary/15 text-primary"
                              : "bg-accent text-muted-foreground",
                          )}
                        >
                          {item.badge}
                        </span>
                      )}
                    </>
                  )}
                </Link>
              );
            })}
          </div>
        ))}
      </div>

      {/* bottom */}
      <div
        className={cn(
          "border-t border-line",
          collapsed ? "py-2" : "p-3",
        )}
      >
        {!collapsed && (
          <>
            <button
              type="button"
              className="flex items-center gap-2 w-full border-none cursor-pointer px-2.5 py-[7px] bg-input rounded-md text-muted-foreground font-display mb-1.5"
            >
              <span>⌕</span>
              <span className="flex-1 text-left">Search</span>
              <kbd className="text-[9px] bg-accent px-1 py-px rounded">
                ⌘K
              </kbd>
            </button>
            <Link
              to="/studio/settings"
              className={cn(
                "flex items-center gap-2 px-1 py-1.5 cursor-pointer no-underline rounded-md transition-colors",
                pathname.startsWith("/studio/settings")
                  ? "bg-line"
                  : "hover:bg-input",
              )}
            >
              <div className="w-6 h-6 rounded-full bg-linear-to-br from-blue-500 to-violet-500 flex items-center justify-center text-[10px] font-bold text-white shrink-0">
                AK
              </div>
              <div className="flex-1 min-w-0">
                <div className="text-subtle font-medium font-display">
                  A. Kovács
                </div>
                <div className="text-[9px] text-faint">Skill Engineer</div>
              </div>
            </Link>
          </>
        )}
        <button
          type="button"
          onClick={() => setCollapsed(!collapsed)}
          className="flex items-center justify-center w-full border-none cursor-pointer py-1.5 bg-transparent text-faint mt-1 hover:text-subtle"
        >
          {collapsed ? "▸" : "◂"}
        </button>
      </div>
    </nav>
  );
}
