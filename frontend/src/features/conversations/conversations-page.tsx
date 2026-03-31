// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: AGPL-3.0-only
//
import { useState } from "react";
import { PageHeader } from "@/components/page-header";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
import { Input } from "@/components/ui/input";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
  DialogFooter,
} from "@/components/ui/dialog";
import {
  Select,
  SelectTrigger,
  SelectValue,
  SelectContent,
  SelectItem,
} from "@/components/ui/select";
import { cn } from "@/utils/cn";
import { CONVERSATIONS, SENTIMENT_COLORS } from "./conversations-data";
import { ConversationDetail, ConversationEmptyState } from "./conversation-detail";
import type { Conversation } from "./conversations-data";

// ── Sentiment filter buttons ──────────────────────────────────

const SENTIMENT_FILTERS = [
  { id: "all", label: "All" },
  { id: "positive", label: "Positive", color: "#22C55E" },
  { id: "negative", label: "Negative", color: "#EF4444" },
  { id: "neutral", label: "Neutral", color: "#6B7585" },
] as const;

// ── Conversation list item ────────────────────────────────────

function ConversationListItem({
  convo,
  selected,
  checked,
  onSelect,
  onToggleCheck,
}: {
  convo: Conversation;
  selected: boolean;
  checked: boolean;
  onSelect: () => void;
  onToggleCheck: () => void;
}) {
  return (
    <div
      className={cn(
        "w-full text-left px-3.5 py-3 border-l-2 border-b border-b-border/50 transition-colors cursor-pointer",
        selected
          ? "bg-muted/60"
          : "border-l-transparent hover:bg-muted/30",
      )}
      style={selected ? { borderLeftColor: convo.agentColor } : undefined}
    >
      <div className="flex items-start gap-2">
        {/* checkbox */}
        <div className="pt-0.5 shrink-0" onClick={(e) => e.stopPropagation()}>
          <Checkbox
            checked={checked}
            onCheckedChange={onToggleCheck}
            className="size-3.5"
          />
        </div>

        <div className="flex-1 min-w-0" onClick={onSelect}>
          {/* top line: sentiment dot + agent → user + time */}
          <div className="flex items-center gap-1.5 mb-0.5">
            <span
              className="inline-block w-1.5 h-1.5 rounded-full shrink-0"
              style={{ backgroundColor: SENTIMENT_COLORS[convo.sentiment] }}
            />
            <span className="text-[11px] font-semibold text-foreground font-display">
              {convo.agent}
            </span>
            <span className="text-[9px] text-muted-foreground/30">&rarr;</span>
            <span className="text-[10px] text-muted-foreground">
              {convo.user}
            </span>
            <span className="flex-1" />
            <span className="text-[9px] text-muted-foreground/30">
              {convo.time}
            </span>
          </div>

          {/* preview */}
          <div className="text-[11px] text-muted-foreground/60 truncate mb-1 pl-3">
            {convo.preview}
          </div>

          {/* meta line: badges */}
          <div className="flex items-center gap-1 pl-3 flex-wrap">
            {convo.flagged && (
              <Badge variant="danger">FLAGGED</Badge>
            )}
            {convo.status === "active" && (
              <span className="text-[7px] px-1 py-px rounded font-semibold font-display border flex items-center gap-1"
                style={{
                  background: "#22C55E12",
                  color: "#22C55E",
                  borderColor: "#22C55E20",
                }}
              >
                <span className="w-1 h-1 rounded-full bg-green-500 animate-pulse" />
                LIVE
              </span>
            )}
            {convo.escalated && (
              <span
                className="text-[7px] px-1 py-px rounded font-semibold font-display"
                style={{ background: "#F59E0B12", color: "#F59E0B" }}
              >
                ESCALATED
              </span>
            )}
            {convo.dataset && (
              <span
                className="text-[7px] px-1 py-px rounded font-semibold font-display"
                style={{ background: "#8B5CF612", color: "#8B5CF6" }}
              >
                IN DATASET
              </span>
            )}
            {convo.starred && (
              <span className="text-[9px] text-amber-500">&starf;</span>
            )}
            <span className="flex-1" />
            <span className="text-[9px] text-muted-foreground/30">
              {convo.turns}t &middot; {convo.tokens.in + convo.tokens.out} tok
            </span>
          </div>
        </div>
      </div>
    </div>
  );
}

// ── Export dialog ──────────────────────────────────────────────

function ExportDialog({
  open,
  onOpenChange,
  selectedCount,
}: {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  selectedCount: number;
}) {
  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent>
        <DialogHeader>
          <DialogTitle>Export to Dataset</DialogTitle>
          <DialogDescription>
            {selectedCount > 0
              ? `${selectedCount} CONVERSATIONS selected`
              : "Configure export filters"}
          </DialogDescription>
        </DialogHeader>

        <div className="grid grid-cols-2 gap-3 mb-4">
          <div>
            <div className="text-[9px] text-muted-foreground/50 mb-1 font-display uppercase tracking-wide">
              Target Dataset
            </div>
            <Select defaultValue="cx-convos-v5">
              <SelectTrigger size="sm" className="w-full">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="cx-convos-v5">cx-convos-v5 (new)</SelectItem>
                <SelectItem value="cx-convos-v4">cx-convos-v4</SelectItem>
                <SelectItem value="code-trajectories-v2">
                  code-trajectories-v2
                </SelectItem>
              </SelectContent>
            </Select>
          </div>
          <div>
            <div className="text-[9px] text-muted-foreground/50 mb-1 font-display uppercase tracking-wide">
              Format
            </div>
            <Select defaultValue="sft">
              <SelectTrigger size="sm" className="w-full">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="sft">SFT (instruction/response)</SelectItem>
                <SelectItem value="dpo">DPO (chosen/rejected)</SelectItem>
                <SelectItem value="rlhf">RLHF (preference pairs)</SelectItem>
                <SelectItem value="raw">Raw (full conversation)</SelectItem>
              </SelectContent>
            </Select>
          </div>
        </div>

        <div className="mb-4">
          <div className="text-[9px] text-muted-foreground/50 mb-2 font-display uppercase tracking-wide">
            Include
          </div>
          <div className="flex flex-wrap gap-1.5">
            {[
              "Tool call traces",
              "Annotations",
              "Metadata",
              "Sentiment scores",
              "Latency data",
            ].map((opt) => (
              <label
                key={opt}
                className="text-[10px] px-2.5 py-1 rounded border border-border bg-muted/40 text-muted-foreground cursor-pointer flex items-center gap-1.5 font-display"
              >
                <Checkbox defaultChecked className="size-3" />
                {opt}
              </label>
            ))}
          </div>
        </div>

        <div className="bg-muted/40 border border-border rounded-md px-3.5 py-2.5 text-[10px] text-muted-foreground">
          Pipeline: Export &rarr;{" "}
          <span className="text-purple-500">NeMo Data Designer</span> &rarr;
          Transform &rarr; Publish to Hub
        </div>

        <DialogFooter>
          <Button variant="outline" onClick={() => onOpenChange(false)}>
            Cancel
          </Button>
          <Button onClick={() => onOpenChange(false)}>
            Export{selectedCount > 0 ? ` (${selectedCount})` : ""}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}

// ── Main page ─────────────────────────────────────────────────

export function ConversationsPage() {
  const [selectedId, setSelectedId] = useState<string | null>("c-9820");
  const [filterAgent, setFilterAgent] = useState("all");
  const [filterSentiment, setFilterSentiment] = useState("all");
  const [filterFlagged, setFilterFlagged] = useState(false);
  const [searchQuery, setSearchQuery] = useState("");
  const [showExportModal, setShowExportModal] = useState(false);
  const [selectedConvos, setSelectedConvos] = useState<Set<string>>(new Set());

  const convo = selectedId
    ? CONVERSATIONS.find((c) => c.id === selectedId) ?? null
    : null;

  const filtered = CONVERSATIONS.filter((c) => {
    if (filterAgent !== "all" && c.agent !== filterAgent) return false;
    if (filterSentiment !== "all" && c.sentiment !== filterSentiment) return false;
    if (filterFlagged && !c.flagged) return false;
    if (searchQuery) {
      const q = searchQuery.toLowerCase();
      if (
        !c.preview.toLowerCase().includes(q) &&
        !c.user.toLowerCase().includes(q) &&
        !c.tags.some((t) => t.includes(q))
      )
        return false;
    }
    return true;
  });

  const toggleSelect = (id: string) => {
    setSelectedConvos((prev) => {
      const next = new Set(prev);
      next.has(id) ? next.delete(id) : next.add(id);
      return next;
    });
  };

  const agents = [...new Set(CONVERSATIONS.map((c) => c.agent))];
  const flaggedCount = CONVERSATIONS.filter((c) => c.flagged).length;
  const activeCount = CONVERSATIONS.filter((c) => c.status === "active").length;

  return (
    <div className="flex-1 flex flex-col overflow-hidden bg-background">
      <PageHeader
        title="Conversations"
        subtitle={
          <>
            {CONVERSATIONS.length} CONVERSATIONS &middot; {flaggedCount} flagged
            &middot; {activeCount} active now
          </>
        }
        action={
          <div className="flex items-center gap-2">
            {selectedConvos.size > 0 && (
              <div className="flex items-center gap-1.5 mr-2">
                <span className="text-[10px] text-amber-500">
                  {selectedConvos.size} selected
                </span>
                <Button variant="outline" size="xs">
                  Tag
                </Button>
                <Button size="xs" onClick={() => setShowExportModal(true)}>
                  &rarr; Export to Dataset
                </Button>
                <button
                  type="button"
                  onClick={() => setSelectedConvos(new Set())}
                  className="bg-transparent border-none text-muted-foreground/40 cursor-pointer text-[11px] hover:text-muted-foreground"
                >
                  &times;
                </button>
              </div>
            )}
            <Button
              variant="outline"
              size="sm"
              onClick={() => setShowExportModal(true)}
            >
              Export to Dataset
            </Button>
          </div>
        }
      />

      <div className="flex-1 flex overflow-hidden">
        {/* Conversation list (left) */}
        <div className="w-[420px] min-w-[420px] border-r border-border flex flex-col">
          {/* Search + filters */}
          <div className="px-3.5 py-3 border-b border-border space-y-2.5">
            <Input
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              placeholder="Search CONVERSATIONS, users, tags..."
              className="h-8 text-xs"
            />
            <div className="flex gap-1 flex-wrap">
              <Select
                value={filterAgent}
                onValueChange={setFilterAgent}
              >
                <SelectTrigger
                  size="sm"
                  className={cn(
                    "h-6 text-[10px] px-1.5 font-display",
                    filterAgent !== "all"
                      ? "text-amber-500 border-amber-500/20"
                      : "text-muted-foreground",
                  )}
                >
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="all">All Agents</SelectItem>
                  {agents.map((a) => (
                    <SelectItem key={a} value={a}>
                      {a}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>

              {SENTIMENT_FILTERS.map((f) => {
                const isActive = filterSentiment === f.id;
                return (
                  <button
                    key={f.id}
                    type="button"
                    onClick={() => setFilterSentiment(f.id)}
                    className={cn(
                      "px-2 py-1 rounded text-[9px] font-medium font-display border transition-colors cursor-pointer",
                      isActive
                        ? "border-current/20 bg-current/10"
                        : "border-transparent text-muted-foreground hover:bg-muted/50",
                    )}
                    style={
                      isActive && "color" in f
                        ? { color: f.color }
                        : isActive
                          ? { color: "#22C55E" }
                          : undefined
                    }
                  >
                    {f.label}
                  </button>
                );
              })}

              <button
                type="button"
                onClick={() => setFilterFlagged(!filterFlagged)}
                className={cn(
                  "px-2 py-1 rounded text-[9px] font-medium font-display border transition-colors cursor-pointer",
                  filterFlagged
                    ? "border-red-500/20 bg-red-500/10 text-red-500"
                    : "border-transparent text-muted-foreground hover:bg-muted/50",
                )}
              >
                &#x2691; Flagged {flaggedCount}
              </button>
            </div>
          </div>

          {/* List */}
          <div className="flex-1 overflow-y-auto">
            {filtered.map((c) => (
              <ConversationListItem
                key={c.id}
                convo={c}
                selected={selectedId === c.id}
                checked={selectedConvos.has(c.id)}
                onSelect={() => setSelectedId(c.id)}
                onToggleCheck={() => toggleSelect(c.id)}
              />
            ))}
            {filtered.length === 0 && (
              <div className="py-8 text-center text-muted-foreground/30 text-xs">
                No CONVERSATIONS match filters
              </div>
            )}
          </div>
        </div>

        {/* Detail (right) */}
        {convo ? (
          <ConversationDetail
            convo={convo}
            onExport={() => setShowExportModal(true)}
          />
        ) : (
          <ConversationEmptyState />
        )}
      </div>

      {/* Export modal */}
      <ExportDialog
        open={showExportModal}
        onOpenChange={setShowExportModal}
        selectedCount={selectedConvos.size}
      />
    </div>
  );
}
