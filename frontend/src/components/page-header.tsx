// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: AGPL-3.0-only
//
import type { ReactNode } from "react";
import { Settings, Sun, Moon, LogOut } from "lucide-react";
import { useNavigate } from "@tanstack/react-router";
import { useTheme } from "@/hooks/use-theme";
import { logout } from "@/api/auth";
import { ProjectSelector } from "@/components/project-selector";
import { useAppStore } from "@/stores/app-store";

interface PageHeaderProps {
  title: string;
  subtitle?: ReactNode;
  action?: ReactNode;
}

export function PageHeader({ title, subtitle, action }: PageHeaderProps) {
  const { isDark, toggle } = useTheme();
  const navigate = useNavigate();
  const hasActiveTasks = useAppStore((s) =>
    s.tasks.some((t) => t.status === "pending" || t.status === "running"),
  );

  return (
    <header className="px-7 py-3 border-b border-line flex items-center justify-between bg-card sticky top-0 z-5">
      <div>
        <h1 className="font-display text-[17px] font-bold text-foreground tracking-tight">
          {title}
        </h1>
        {subtitle && (
          <p className="text-muted-foreground mt-px">{subtitle}</p>
        )}
      </div>
      <div className="flex items-center gap-3">
        {hasActiveTasks && (
          <button
            type="button"
            onClick={() => navigate({ to: "/studio/compute/workload-queue" })}
            className="relative flex size-2.5 cursor-pointer bg-transparent border-none p-0"
          >
            <span className="absolute inline-flex size-full animate-ping rounded-full bg-green-500 opacity-75" />
            <span className="relative inline-flex size-2.5 rounded-full bg-green-500" />
          </button>
        )}
        {action}
        <ProjectSelector />
        <div className="w-px h-5 bg-line" />
        <button
          type="button"
          className="bg-transparent border-none text-muted-foreground cursor-pointer hover:text-foreground"
        >
          <Settings size={16} />
        </button>
        <button
          type="button"
          onClick={toggle}
          className="bg-transparent border-none text-muted-foreground cursor-pointer hover:text-foreground"
        >
          {isDark ? <Sun size={16} /> : <Moon size={16} />}
        </button>
        <button
          type="button"
          onClick={() => { logout(); window.location.href = "/login"; }}
          className="bg-transparent border-none text-muted-foreground cursor-pointer hover:text-foreground"
        >
          <LogOut size={16} />
        </button>
      </div>
    </header>
  );
}
