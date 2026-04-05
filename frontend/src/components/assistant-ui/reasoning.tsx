// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: AGPL-3.0-only
//

import { MarkdownText } from "@/components/assistant-ui/markdown-text";
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/ui/collapsible";
import { cn } from "@/utils/cn";
import {
  type ReasoningGroupComponent,
  type ReasoningMessagePartComponent,
  useAuiState,
} from "@assistant-ui/react";
import { ChevronDownIcon, LightbulbIcon } from "lucide-react";
import {
  type ComponentProps,
  memo,
  useCallback,
  useEffect,
  useRef,
  useState,
} from "react";

const AUTO_SCROLL_THRESHOLD_PX = 24;

const ReasoningImpl: ReasoningMessagePartComponent = () => <MarkdownText />;

function ReasoningText({
  className,
  streaming,
  children,
  ...props
}: ComponentProps<"div"> & { streaming?: boolean }) {
  const scrollRef = useRef<HTMLDivElement>(null);
  const shouldAutoScrollRef = useRef(true);
  const detachedFromBottomRef = useRef(false);
  const lastScrollTopRef = useRef(0);

  useEffect(() => {
    if (!(streaming && scrollRef.current)) return;
    const el = scrollRef.current;
    const updateAutoScroll = () => {
      const currentScrollTop = el.scrollTop;
      if (currentScrollTop < lastScrollTopRef.current) {
        detachedFromBottomRef.current = true;
      }
      const distanceFromBottom =
        el.scrollHeight - el.scrollTop - el.clientHeight;
      if (
        detachedFromBottomRef.current &&
        distanceFromBottom <= AUTO_SCROLL_THRESHOLD_PX
      ) {
        detachedFromBottomRef.current = false;
      }
      shouldAutoScrollRef.current = !detachedFromBottomRef.current;
      lastScrollTopRef.current = currentScrollTop;
    };
    const handleWheel = (event: WheelEvent) => {
      if (event.deltaY < 0) {
        detachedFromBottomRef.current = true;
        shouldAutoScrollRef.current = false;
      }
    };
    const observer = new MutationObserver(() => {
      if (shouldAutoScrollRef.current) {
        el.scrollTop = el.scrollHeight;
      }
    });
    el.addEventListener("scroll", updateAutoScroll);
    el.addEventListener("wheel", handleWheel, { passive: true });
    observer.observe(el, {
      childList: true,
      subtree: true,
      characterData: true,
    });
    lastScrollTopRef.current = el.scrollTop;
    detachedFromBottomRef.current = false;
    updateAutoScroll();
    return () => {
      observer.disconnect();
      el.removeEventListener("scroll", updateAutoScroll);
      el.removeEventListener("wheel", handleWheel);
    };
  }, [streaming]);

  return (
    <div
      ref={scrollRef}
      className={cn(
        "relative z-0 overflow-y-auto pt-2 pb-2 leading-relaxed",
        streaming ? "max-h-32" : "max-h-64",
        className,
      )}
      {...props}
    >
      {children}
    </div>
  );
}

const ReasoningGroupImpl: ReasoningGroupComponent = ({
  children,
  startIndex,
  endIndex,
}) => {
  const isReasoningStreaming = useAuiState(({ message }) => {
    if (message.status?.type !== "running") return false;
    const lastIndex = message.parts.length - 1;
    if (lastIndex < 0) return false;
    const lastType = message.parts[lastIndex]?.type;
    if (lastType !== "reasoning") return false;
    return lastIndex >= startIndex && lastIndex <= endIndex;
  });

  const persistedDuration = useAuiState(({ message }) => {
    const d = (message.metadata?.custom as Record<string, unknown>)
      ?.reasoningDuration;
    return typeof d === "number" ? d : 0;
  });

  const [manualOpen, setManualOpen] = useState(false);
  const [dismissedWhileStreaming, setDismissedWhileStreaming] = useState(false);
  const [duration, setDuration] = useState<number>(0);
  const startTimeRef = useRef<number | null>(null);

  useEffect(() => {
    if (isReasoningStreaming) {
      if (startTimeRef.current === null) {
        startTimeRef.current = Date.now();
      }
    } else if (startTimeRef.current !== null) {
      const elapsed = Math.round((Date.now() - startTimeRef.current) / 1000);
      setDuration(elapsed);
      startTimeRef.current = null;
    }
  }, [isReasoningStreaming]);

  useEffect(() => {
    if (isReasoningStreaming) setDismissedWhileStreaming(false);
  }, [isReasoningStreaming]);

  const isOpen =
    (isReasoningStreaming && !dismissedWhileStreaming) || manualOpen;

  const handleOpenChange = useCallback(
    (open: boolean) => {
      if (isReasoningStreaming) {
        setDismissedWhileStreaming(!open);
      } else {
        setManualOpen(open);
      }
    },
    [isReasoningStreaming],
  );

  const displayDuration = duration || persistedDuration;

  return (
    <Collapsible
      open={isOpen}
      onOpenChange={handleOpenChange}
      className={cn(
        "mb-4 w-full rounded-lg border px-3 py-2",
        isOpen ? "border-border" : "border-transparent",
      )}
    >
      <CollapsibleTrigger className="flex w-full min-w-0 items-center gap-2 py-1 text-muted-foreground text-sm transition-colors hover:text-foreground">
        <LightbulbIcon className="size-4 shrink-0" />
        <span className="flex-1 text-left leading-none">
          {isReasoningStreaming ? (
            <span className="animate-pulse">Thinking...</span>
          ) : (
            <span>Thought for {displayDuration} seconds</span>
          )}
        </span>
        <ChevronDownIcon
          className={cn(
            "size-4 shrink-0 transition-transform duration-200",
            isOpen ? "rotate-0" : "-rotate-90",
          )}
        />
      </CollapsibleTrigger>
      <CollapsibleContent className="overflow-hidden text-muted-foreground text-sm">
        <ReasoningText streaming={isReasoningStreaming}>
          {children}
        </ReasoningText>
      </CollapsibleContent>
    </Collapsible>
  );
};

export const Reasoning = memo(
  ReasoningImpl,
) as unknown as ReasoningMessagePartComponent;
Reasoning.displayName = "Reasoning";

export const ReasoningGroup = memo(ReasoningGroupImpl);
ReasoningGroup.displayName = "ReasoningGroup";
