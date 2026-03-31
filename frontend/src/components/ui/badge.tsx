// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: AGPL-3.0-only
//
import { cn } from "@/utils/cn";
import type { ReactNode } from "react";

interface BadgeProps {
  children: ReactNode;
  variant?: "default" | "danger" | "active";
  className?: string;
}

export function Badge({ children, variant = "default", className }: BadgeProps) {
  return (
    <span
      className={cn(
        "text-sm px-1.5 py-[1px] rounded font-display",
        variant === "default" && "bg-accent text-muted-foreground",
        variant === "danger" && "bg-destructive/10 text-destructive",
        variant === "active" && "bg-primary/15 text-primary",
        className,
      )}
    >
      {children}
    </span>
  );
}
