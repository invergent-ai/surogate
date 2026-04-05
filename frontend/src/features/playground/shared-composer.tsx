// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: AGPL-3.0-only
//

import { TooltipIconButton } from "@/components/assistant-ui/tooltip-icon-button";
import { Button } from "@/components/ui/button";
import { useAui } from "@assistant-ui/react";
import { ArrowUpIcon, PlusIcon, SquareIcon, XIcon } from "lucide-react";
import {
  type KeyboardEvent,
  type MutableRefObject,
  type ReactElement,
  type ReactNode,
  createContext,
  useCallback,
  useContext,
  useEffect,
  useRef,
  useState,
} from "react";
import { usePlaygroundStore } from "./stores/playground-store";

export type CompareMessagePart =
  | { type: "text"; text: string }
  | { type: "image"; image: string };

export interface CompareHandle {
  append: (content: CompareMessagePart[]) => void;
  appendMessage: (content: CompareMessagePart[]) => void;
  startRun: () => void;
  cancel: () => void;
  isRunning: () => boolean;
  waitForRunEnd: () => Promise<void>;
}

export type CompareHandles = MutableRefObject<Record<string, CompareHandle>>;

const IMAGE_ACCEPT = "image/jpeg,image/png,image/webp,image/gif";
const MAX_IMAGE_SIZE = 20 * 1024 * 1024;

function fileToBase64DataURL(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result as string);
    reader.onerror = () => reject(new Error("Failed to read image file"));
    reader.readAsDataURL(file);
  });
}

// ── Context ──────────────────────────────────────────────────

const CompareHandlesContext = createContext<CompareHandles | null>(null);

export function CompareHandlesProvider({
  handlesRef,
  children,
}: {
  handlesRef: CompareHandles;
  children: ReactNode;
}): ReactElement {
  return (
    <CompareHandlesContext.Provider value={handlesRef}>
      {children}
    </CompareHandlesContext.Provider>
  );
}

export function RegisterCompareHandle({
  name,
}: {
  name: string;
}): ReactElement | null {
  const handlesRef = useContext(CompareHandlesContext);
  const aui = useAui();

  useEffect(() => {
    if (!handlesRef) return;
    const currentHandles = handlesRef.current;
    currentHandles[name] = {
      append: (content) =>
        aui
          .thread()
          .append({
            role: "user",
            content,
            createdAt: new Date(),
          } as never),
      appendMessage: (content) =>
        aui
          .thread()
          .append({
            role: "user",
            content,
            createdAt: new Date(),
            startRun: false,
          } as never),
      startRun: () => {
        const msgs = aui.thread().getState().messages;
        const lastId = msgs.length > 0 ? msgs[msgs.length - 1].id : null;
        aui.thread().startRun({ parentId: lastId });
      },
      cancel: () => aui.thread().cancelRun(),
      isRunning: () => aui.thread().getState().isRunning,
      waitForRunEnd: () =>
        new Promise<void>((resolve) => {
          let wasRunning = false;
          const unsub = usePlaygroundStore.subscribe((state) => {
            const anyRunning =
              Object.keys(state.runningByThreadId).length > 0;
            if (anyRunning) wasRunning = true;
            if (wasRunning && !anyRunning) {
              unsub();
              resolve();
            }
          });
        }),
    };
    return () => {
      delete currentHandles[name];
    };
  }, [handlesRef, name, aui]);

  return null;
}

// ── Pending Image Thumbnail ──────────────────────────────────

type PendingImage = { id: string; file: File };

function PendingImageThumb({
  file,
  onRemove,
}: {
  file: File;
  onRemove: () => void;
}): ReactElement {
  const [src, setSrc] = useState<string | null>(null);
  useEffect(() => {
    const url = URL.createObjectURL(file);
    setSrc(url);
    return () => URL.revokeObjectURL(url);
  }, [file]);
  if (!src)
    return (
      <div className="size-14 animate-pulse rounded-[14px] bg-muted" />
    );
  return (
    <div className="relative size-14 shrink-0 overflow-hidden rounded-[14px] border border-foreground/20 bg-muted">
      <img
        src={src}
        alt={file.name}
        className="h-full w-full object-cover"
      />
      <button
        type="button"
        onClick={onRemove}
        className="absolute top-1 right-1 flex size-5 items-center justify-center rounded-full bg-white text-muted-foreground shadow-sm hover:bg-destructive hover:text-destructive-foreground"
        aria-label="Remove attachment"
      >
        <XIcon className="size-3" />
      </button>
    </div>
  );
}

// ── Shared Composer ──────────────────────────────────────────

export function SharedComposer({
  handlesRef,
}: {
  handlesRef: CompareHandles;
}): ReactElement {
  const [text, setText] = useState("");
  const [running, setRunning] = useState(false);
  const [pendingImages, setPendingImages] = useState<PendingImage[]>([]);
  const [dragging, setDragging] = useState(false);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    const id = setInterval(() => {
      const handles = handlesRef.current;
      const any = Object.values(handles).some((h) => h.isRunning());
      setRunning(any);
    }, 200);
    return () => clearInterval(id);
  }, [handlesRef]);

  const addFiles = useCallback((files: FileList | null) => {
    if (!files?.length) return;
    const next: PendingImage[] = [];
    for (let i = 0; i < files.length; i++) {
      const file = files[i];
      if (!file) continue;
      if (!file.type.match(/^image\/(jpeg|png|webp|gif)$/i)) continue;
      if (file.size > MAX_IMAGE_SIZE) continue;
      next.push({ id: crypto.randomUUID(), file });
    }
    setPendingImages((prev) => [...prev, ...next]);
  }, []);

  const removePendingImage = useCallback((id: string) => {
    setPendingImages((prev) => prev.filter((p) => p.id !== id));
  }, []);

  async function send() {
    const msg = text.trim();
    if (!msg && pendingImages.length === 0) return;

    const content: CompareMessagePart[] = [];
    for (const { file } of pendingImages) {
      try {
        const image = await fileToBase64DataURL(file);
        content.push({ type: "image", image });
      } catch {
        // skip failed image
      }
    }
    if (msg) {
      content.push({ type: "text", text: msg });
    }
    if (content.length === 0) return;

    setText("");
    setPendingImages([]);
    textareaRef.current?.focus();

    // Fire all handles simultaneously
    for (const handle of Object.values(handlesRef.current)) {
      handle.append(content);
    }
  }

  function stop() {
    for (const handle of Object.values(handlesRef.current)) {
      handle.cancel();
    }
  }

  function onKeyDown(e: KeyboardEvent) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      if (!running) {
        send();
      }
    }
  }

  const canSend =
    (text.trim().length > 0 || pendingImages.length > 0) && !running;

  return (
    <div
      className={`relative flex w-full flex-col rounded-2xl border bg-background px-1 pt-2 shadow-sm transition-shadow outline-none ${dragging ? "border-ring bg-accent/50" : "border-border"}`}
      onDragOver={(e) => {
        e.preventDefault();
        setDragging(true);
      }}
      onDragLeave={() => setDragging(false)}
      onDrop={(e) => {
        e.preventDefault();
        setDragging(false);
        addFiles(e.dataTransfer.files);
      }}
    >
      {pendingImages.length > 0 && (
        <div className="mb-2 flex w-full flex-row flex-wrap items-center gap-2 px-1.5 pt-0.5 pb-1">
          {pendingImages.map(({ id, file }) => (
            <PendingImageThumb
              key={id}
              file={file}
              onRemove={() => removePendingImage(id)}
            />
          ))}
        </div>
      )}
      <textarea
        ref={textareaRef}
        value={text}
        onChange={(e) => setText(e.target.value)}
        onKeyDown={onKeyDown}
        placeholder="Send to both models..."
        className="mb-1 max-h-32 min-h-14 w-full resize-none bg-transparent px-4 pt-2 pb-3 text-sm outline-none placeholder:text-muted-foreground"
        rows={1}
      />
      <div className="relative mx-2 mb-2 flex items-center justify-between">
        <div className="flex items-center gap-1">
          <input
            ref={fileInputRef}
            type="file"
            accept={IMAGE_ACCEPT}
            multiple
            className="hidden"
            onChange={(e) => {
              addFiles(e.target.files);
              e.target.value = "";
            }}
          />
          <TooltipIconButton
            tooltip="Add attachment"
            side="bottom"
            variant="ghost"
            size="icon"
            className="size-8 rounded-full text-muted-foreground hover:bg-muted-foreground/15"
            onClick={() => fileInputRef.current?.click()}
          >
            <PlusIcon className="size-5 stroke-[1.5px]" />
          </TooltipIconButton>
        </div>
        <div className="flex items-center gap-1">
          {running ? (
            <Button
              type="button"
              variant="default"
              size="icon"
              className="size-8 rounded-full"
              onClick={stop}
            >
              <SquareIcon className="size-3 fill-current" />
            </Button>
          ) : (
            <TooltipIconButton
              tooltip="Send message"
              side="bottom"
              variant="default"
              size="icon"
              className="size-8 rounded-full"
              onClick={send}
              disabled={!canSend}
            >
              <ArrowUpIcon className="size-4" />
            </TooltipIconButton>
          )}
        </div>
      </div>
    </div>
  );
}
