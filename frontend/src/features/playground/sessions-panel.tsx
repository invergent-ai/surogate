// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: AGPL-3.0-only
//

import { Button } from "@/components/ui/button";
import { ScrollArea } from "@/components/ui/scroll-area";
import { cn } from "@/utils/cn";
import { ColumnsIcon, Trash2Icon } from "lucide-react";
import { db, useLiveQuery } from "./db";
import type { PlaygroundView, ThreadRecord } from "./types";

interface SidebarItem {
  type: "single" | "compare";
  id: string;
  title: string;
  createdAt: number;
}

function groupThreads(threads: ThreadRecord[]): SidebarItem[] {
  const items: SidebarItem[] = [];
  const seenPairs = new Set<string>();

  for (const t of threads) {
    if (t.archived) continue;
    if (t.pairId) {
      if (seenPairs.has(t.pairId)) continue;
      seenPairs.add(t.pairId);
      items.push({
        type: "compare",
        id: t.pairId,
        title: t.title,
        createdAt: t.createdAt,
      });
    } else {
      items.push({
        type: "single",
        id: t.id,
        title: t.title,
        createdAt: t.createdAt,
      });
    }
  }

  return items.sort((a, b) => b.createdAt - a.createdAt);
}

function viewForItem(item: SidebarItem): PlaygroundView {
  return item.type === "single"
    ? { mode: "single", threadId: item.id }
    : { mode: "compare", pairId: item.id };
}

interface SessionsPanelProps {
  view: PlaygroundView;
  onSelect: (view: PlaygroundView) => void;
  onNewSession: () => void;
  onNewCompare: () => void;
}

export function SessionsPanel({
  view,
  onSelect,
  onNewSession,
  onNewCompare,
}: SessionsPanelProps) {
  const allThreads = useLiveQuery(
    () => db.threads.orderBy("createdAt").reverse().toArray(),
    [],
  );
  const items = groupThreads(allThreads ?? []);
  const activeId =
    view.mode === "single" ? view.threadId : view.pairId;

  async function handleDelete(item: SidebarItem) {
    if (item.type === "single") {
      await db.messages.where("threadId").equals(item.id).delete();
      await db.threads.delete(item.id);
    } else {
      const paired = await db.threads
        .where("pairId")
        .equals(item.id)
        .toArray();
      for (const t of paired) {
        await db.messages.where("threadId").equals(t.id).delete();
        await db.threads.delete(t.id);
      }
    }
    if (activeId === item.id) {
      onSelect({ mode: "single" });
    }
  }

  return (
    <div className="flex w-55 min-w-55 animate-in fade-in flex-col border-r border-border bg-card duration-150">
      <div className="flex items-center justify-between border-b border-border px-3 py-2.5">
        <span className="font-display text-[11px] font-semibold text-foreground">
          Sessions
        </span>
        <div className="flex gap-1">
          <Button variant="outline" size="xs" onClick={onNewSession}>
            + New
          </Button>
          <Button variant="outline" size="xs" onClick={onNewCompare}>
            <ColumnsIcon className="size-3" />
          </Button>
        </div>
      </div>
      <ScrollArea className="flex-1">
        {items.map((item) => (
          <div
            key={item.id}
            className={cn(
              "group flex w-full items-center gap-1.5 border-b border-border/50 px-3 py-2.5 transition-colors hover:bg-muted",
              activeId === item.id && "bg-muted",
            )}
          >
            <button
              type="button"
              className="min-w-0 flex-1 text-left"
              onClick={() => onSelect(viewForItem(item))}
            >
              <div className="flex items-center gap-1.5">
                {item.type === "compare" && (
                  <ColumnsIcon className="size-3 shrink-0 text-muted-foreground/60" />
                )}
                <span className="truncate text-[11px] font-medium text-foreground">
                  {item.title}
                </span>
              </div>
              <div className="mt-0.5 text-[9px] text-muted-foreground">
                {new Date(item.createdAt).toLocaleDateString()}
              </div>
            </button>
            <button
              type="button"
              className="shrink-0 rounded p-0.5 text-muted-foreground/40 opacity-0 transition-opacity hover:text-destructive group-hover:opacity-100"
              onClick={() => handleDelete(item)}
              aria-label="Delete"
            >
              <Trash2Icon className="size-3" />
            </button>
          </div>
        ))}
        {items.length === 0 && (
          <p className="px-3 py-6 text-center text-[10px] text-muted-foreground">
            No threads yet
          </p>
        )}
      </ScrollArea>
    </div>
  );
}
