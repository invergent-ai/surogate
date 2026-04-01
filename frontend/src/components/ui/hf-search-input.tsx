// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: AGPL-3.0-only
//
import { useState, useEffect, useRef } from "react";
import { hfSearch, type HFItem } from "@/utils/hf";
import { cn } from "@/utils/cn";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import {
  ChevronLeft,
  ChevronRight,
  ChevronsUpDown,
  Download,
  FileBox,
  Loader2,
  Search,
} from "lucide-react";

type HfSearchKind = "models" | "datasets";

interface HfSearchInputProps {
  value: string;
  onChange: (value: string) => void;
  kind: HfSearchKind;
  token?: string;
  placeholder?: string;
  className?: string;
  disabled?: boolean;
}

function ggufFiles(item: HFItem): string[] {
  if (!item.siblings) return [];
  return item.siblings
    .map((s) => s.rfilename)
    .filter((f) => f.endsWith(".gguf"));
}

export function HfSearchInput({
  value,
  onChange,
  kind,
  token,
  placeholder = "qwen/Qwen3-4B",
  className,
  disabled,
}: HfSearchInputProps) {
  const [results, setResults] = useState<HFItem[]>([]);
  const [loading, setLoading] = useState(false);
  const [open, setOpen] = useState(false);
  const [query, setQuery] = useState("");
  const [selectedItem, setSelectedItem] = useState<HFItem | null>(null);
  const [filter, setFilter] = useState("");
  const containerRef = useRef<HTMLDivElement>(null);
  const searchRef = useRef<HTMLInputElement>(null);

  // Debounced HF search
  useEffect(() => {
    if (query.length < 3) {
      setResults([]);
      return;
    }

    setLoading(true);
    const timer = setTimeout(async () => {
      try {
        const items = await hfSearch(kind, query, 10, token || undefined);
        setResults(items);
      } catch {
        setResults([]);
      } finally {
        setLoading(false);
      }
    }, 300);

    return () => { clearTimeout(timer); setLoading(false); };
  }, [query, kind, token]);

  // Close on outside click
  useEffect(() => {
    if (!open) return;
    const handler = (e: MouseEvent) => {
      if (containerRef.current && !containerRef.current.contains(e.target as Node)) {
        setOpen(false);
        setSelectedItem(null);
        setFilter("");
      }
    };
    document.addEventListener("mousedown", handler);
    return () => document.removeEventListener("mousedown", handler);
  }, [open]);

  // Focus search input when panel opens or navigates
  useEffect(() => {
    if (open) searchRef.current?.focus();
  }, [open, selectedItem]);

  const handleRepoClick = (item: HFItem) => {
    const ggufs = ggufFiles(item);
    if (ggufs.length > 0) {
      setSelectedItem(item);
      setFilter("");
    } else {
      onChange(item.id);
      setOpen(false);
      setSelectedItem(null);
      setQuery("");
    }
  };

  const handleGgufSelect = (repo: string, filename: string) => {
    onChange(`${repo}/${filename}`);
    setOpen(false);
    setSelectedItem(null);
    setQuery("");
  };

  const handleBack = () => {
    setSelectedItem(null);
    setFilter("");
  };

  const ggufs = selectedItem ? ggufFiles(selectedItem) : [];
  const filteredGgufs = filter
    ? ggufs.filter((f) => f.toLowerCase().includes(filter.toLowerCase()))
    : ggufs;

  return (
    <TooltipProvider>
    <div ref={containerRef} className={cn("relative min-w-0 w-full", className)}>
      {/* Trigger */}
      <Tooltip>
        <TooltipTrigger asChild>
          <button
            type="button"
            disabled={disabled}
            onClick={() => setOpen((v) => !v)}
            className={cn(
              "flex w-full items-center justify-between h-8 rounded-lg border border-input bg-transparent px-2.5 text-xs transition-colors cursor-pointer",
              "hover:border-ring/50 focus-visible:border-ring focus-visible:ring-3 focus-visible:ring-ring/50 outline-none",
              value ? "text-foreground font-mono" : "text-muted-foreground",
            )}
          >
            <span className="flex-1 w-0 truncate text-left">{value || placeholder}</span>
            <ChevronsUpDown size={12} className="shrink-0 ml-2 text-muted-foreground/50" />
          </button>
        </TooltipTrigger>
        {value && (
          <TooltipContent side="top" className="max-w-sm break-all font-mono">
            {value}
          </TooltipContent>
        )}
      </Tooltip>

      {/* Dropdown panel */}
      {open && (
        <div className="absolute z-50 mt-1 w-full rounded-lg border border-border bg-popover shadow-md overflow-hidden">
          {/* Header / breadcrumb */}
          <div className="flex items-center gap-1.5 border-b border-border bg-muted/30 px-2 h-7">
            {selectedItem ? (
              <>
                <button
                  onClick={handleBack}
                  className="flex items-center gap-0.5 text-muted-foreground hover:text-foreground transition-colors cursor-pointer bg-transparent border-none p-0"
                >
                  <ChevronLeft size={14} />
                  <span className="text-[10px] font-display">Results</span>
                </button>
                <span className="text-muted-foreground/40 text-[10px]">/</span>
                <span className="text-[11px] font-mono text-foreground truncate">
                  {selectedItem.id}
                </span>
              </>
            ) : (
              <span className="text-[10px] text-muted-foreground/60 uppercase tracking-wide font-display">
                HuggingFace {kind === "models" ? "Models" : "Datasets"}
              </span>
            )}
          </div>

          {/* Search / filter */}
          <div className="relative border-b border-border">
            <Search
              size={12}
              className="absolute left-2.5 top-1/2 -translate-y-1/2 text-muted-foreground/50"
            />
            <input
              ref={searchRef}
              value={selectedItem ? filter : query}
              onChange={(e) =>
                selectedItem ? setFilter(e.target.value) : setQuery(e.target.value)
              }
              placeholder={
                selectedItem ? "Filter GGUF files…" : "Search HuggingFace…"
              }
              className="w-full h-7 pl-7 pr-2 text-xs bg-transparent border-none outline-none placeholder:text-muted-foreground/40"
            />
            {loading && !selectedItem && (
              <Loader2
                size={12}
                className="absolute right-2.5 top-1/2 -translate-y-1/2 animate-spin text-muted-foreground"
              />
            )}
          </div>

          {/* Sliding panels */}
          <div className="relative h-52 overflow-hidden">
            <div
              className={cn(
                "absolute inset-0 flex transition-transform duration-200 ease-out",
                selectedItem ? "-translate-x-full" : "translate-x-0",
              )}
            >
              {/* Panel 1 — Search results */}
              <div className="min-w-full h-full overflow-y-auto">
                {results.length === 0 ? (
                  <div className="flex items-center justify-center h-full text-xs text-muted-foreground/50">
                    {query.length < 3 ? "Type to search…" : loading ? "" : "No results"}
                  </div>
                ) : (
                  results.map((item) => {
                    const hasGgufs = ggufFiles(item).length > 0;
                    return (
                      <Tooltip key={item.id}>
                        <TooltipTrigger asChild>
                          <button
                            type="button"
                            onClick={() => handleRepoClick(item)}
                            className={cn(
                              "flex w-full items-center gap-2 px-3 py-1.5 text-left text-xs transition-colors cursor-pointer border-none bg-transparent",
                              "hover:bg-accent/60",
                            )}
                          >
                            <div className="flex-1 min-w-0">
                              <span className="font-mono text-foreground truncate block">
                                {item.id}
                              </span>
                              {item.downloads != null && item.downloads > 0 && (
                                <span className="flex items-center gap-1 text-[10px] text-muted-foreground mt-0.5">
                                  <Download size={9} />
                                  {item.downloads.toLocaleString()}
                                </span>
                              )}
                            </div>
                            {hasGgufs && (
                              <ChevronRight
                                size={12}
                                className="text-muted-foreground/40 shrink-0"
                              />
                            )}
                          </button>
                        </TooltipTrigger>
                        <TooltipContent side="right" className="max-w-sm break-all font-mono">
                          {item.id}
                        </TooltipContent>
                      </Tooltip>
                    );
                  })
                )}
              </div>

              {/* Panel 2 — GGUF file list */}
              <div className="min-w-full h-full overflow-y-auto">
                {filteredGgufs.length === 0 ? (
                  <div className="flex items-center justify-center h-full text-xs text-muted-foreground/50">
                    No GGUF files match
                  </div>
                ) : (
                  <>
                    <div className="px-3 py-1 text-[9px] uppercase tracking-wide text-muted-foreground/50 font-display sticky top-0 bg-popover z-10">
                      GGUF Files
                    </div>
                    {filteredGgufs.map((filename) => (
                      <Tooltip key={filename}>
                        <TooltipTrigger asChild>
                          <button
                            type="button"
                            onClick={() => handleGgufSelect(selectedItem!.id, filename)}
                            className={cn(
                              "flex w-full items-center gap-2 px-3 py-1.5 text-left text-xs transition-colors cursor-pointer border-none bg-transparent",
                              "hover:bg-accent/60",
                            )}
                          >
                            <FileBox size={12} className="shrink-0 text-muted-foreground" />
                            <span className="font-mono truncate">{filename}</span>
                          </button>
                        </TooltipTrigger>
                        <TooltipContent side="right" className="max-w-sm break-all font-mono">
                          {filename}
                        </TooltipContent>
                      </Tooltip>
                    ))}
                  </>
                )}
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
    </TooltipProvider>
  );
}
