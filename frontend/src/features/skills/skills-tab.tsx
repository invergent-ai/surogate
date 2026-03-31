// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: AGPL-3.0-only
//
import { useEffect, useState } from "react";
import { Loader2, Trash2 } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { ConfirmDialog } from "@/components/ui/confirm-dialog";
import { Input } from "@/components/ui/input";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
  DialogFooter,
} from "@/components/ui/dialog";
import { StatusDot } from "@/components/ui/status-dot";
import { Tabs, TabsList, TabsTrigger, TabsContent } from "@/components/ui/tabs";
import { cn } from "@/utils/cn";
import { toStatus } from "./skills-data";
import type { Skill } from "@/types/skill";
import { SkillForm } from "./skill-form";
import type { SkillFormData } from "./skill-form";
import { useAppStore } from "@/stores/app-store";
import { RepoExplorer } from "@/features/hub/repo-explorer";
import { MarkdownPreview } from "@/components/markdown/markdown-preview";

function SkillListItem({
  skill,
  selected,
  onSelect,
}: {
  skill: Skill;
  selected: boolean;
  onSelect: () => void;
}) {
  return (
    <button
      onClick={onSelect}
      className={cn(
        "w-full text-left px-4 py-3 border-l-2 transition-colors cursor-pointer",
        selected
          ? "bg-muted/60 border-l-amber-500"
          : "border-l-transparent hover:bg-muted/30",
      )}
    >
      <div className="flex items-center justify-between mb-1">
        <div className="flex items-center gap-2">
          <span className="text-amber-500 text-sm">&#x1F4C4;</span>
          <span className="text-sm font-semibold text-foreground font-display">
            {skill.displayName}
          </span>
          <Badge>{skill.version}</Badge>
        </div>
        <StatusDot status={toStatus(skill.status)} />
      </div>
      <div className="flex items-center gap-2 text-xs text-muted-foreground/50">
        { skill.agent && <span>&#x2B21; {skill.agent}</span> }
        { skill.agent && <span>&middot;</span> }
        <span>{skill.author}</span>
        <span>&middot;</span>
        <span>{skill.updatedAt}</span>
      </div>
    </button>
  );
}

function SkillDetail({ skill, onClose, onEdit, onDelete, onPublish }: { skill: Skill; onClose: () => void; onEdit: () => void; onDelete: () => Promise<void>; onPublish: (tag: string) => Promise<boolean> }) {
  const [showDeleteConfirm, setShowDeleteConfirm] = useState(false);
  const [showPublish, setShowPublish] = useState(false);
  const [publishTag, setPublishTag] = useState("");
  const [publishing, setPublishing] = useState(false);
  const [publishError, setPublishError] = useState<string | null>(null);

  const handlePublish = async () => {
    if (!publishTag.trim()) return;
    setPublishing(true);
    setPublishError(null);
    const ok = await onPublish(publishTag.trim());
    setPublishing(false);
    if (ok) {
      setShowPublish(false);
      setPublishTag("");
    } else {
      setPublishError("Failed to publish. The tag may already exist.");
    }
  };

  return (
    <div className="flex-1 flex flex-col overflow-hidden animate-in fade-in duration-150">
      {/* Header */}
      <div className="px-6 py-4 border-b border-border shrink-0">
        <div className="flex items-center justify-between mb-1.5">
          <div className="flex items-center gap-2">
            <span className="text-xl text-amber-500">&#x1F4C4;</span>
            <div>
              <div className="flex items-center gap-2">
                <span className="text-base font-bold text-foreground font-display">
                  {skill.displayName}
                </span>
                <Badge>{skill.version}</Badge>
                <StatusDot status={toStatus(skill.status)} />
              </div>
              <div className="text-sm text-muted-foreground mt-0.5">
                Agent: <span className="text-foreground/70">{skill.agent || 'none'}</span> &middot;{" "}
                {skill.author} &middot; {skill.updatedAt}
              </div>
            </div>
          </div>
          <div className="flex gap-1.5">
            <Button variant="outline" onClick={onEdit}>Edit</Button>
            <Button onClick={() => { setPublishTag(""); setPublishError(null); setShowPublish(true); }}>Publish</Button>
            <Button variant="destructive" onClick={() => setShowDeleteConfirm(true)}>
              <Trash2 size={12} className="mr-1" />Delete
            </Button>
            <Button variant="ghost" size="icon-xs" onClick={onClose}>
              &#x2715;
            </Button>
          </div>
        </div>
        <div className="flex flex-wrap gap-1 mt-1.5">
          {skill.tags.map((t) => (
            <Badge key={t}>{t}</Badge>
          ))}
        </div>
      </div>

      {/* Sub-tabs */}
      <Tabs defaultValue="content" className="flex-1 flex flex-col overflow-hidden">
        <div className="px-6 border-b border-border shrink-0">
          <TabsList variant="line">
            <TabsTrigger value="content">Content</TabsTrigger>
            {skill.hubRef && <TabsTrigger value="repository">Repository</TabsTrigger>}
          </TabsList>
        </div>

        <TabsContent value="content" className="flex-1 overflow-y-auto">
          <pre className="m-5 rounded-md border border-border/60 bg-muted/20 text-foreground/80 leading-relaxed whitespace-pre-wrap font-mono">
            <p className="px-5 py-3 font-bold">DESCRIPTION</p>
            <p className="px-5">{skill.description}</p>
          </pre>
          <pre className="px-5 text-foreground/80 leading-relaxed whitespace-pre-wrap font-mono">
            <MarkdownPreview
              markdown={skill.content}
              className="!max-h-none !text-base !p-5 !leading-normal"
            />
          </pre>
        </TabsContent>

        {skill.hubRef && (
          <TabsContent value="repository" className="flex-1 flex flex-col overflow-hidden">
            <RepoExplorer repoId={skill.hubRef} className="flex-1" />
          </TabsContent>
        )}
      </Tabs>

      <ConfirmDialog
        open={showDeleteConfirm}
        title="Delete Skill"
        description={<>Are you sure you want to delete <span className="font-semibold text-foreground">{skill.displayName}</span>? This action cannot be undone.</>}
        confirmLabel="Delete"
        confirmIcon={<Trash2 size={14} className="mr-1.5" />}
        onConfirm={async () => { await onDelete(); setShowDeleteConfirm(false); }}
        onCancel={() => setShowDeleteConfirm(false)}
      />

      <Dialog open={showPublish} onOpenChange={(o) => { if (!o && !publishing) setShowPublish(false); }}>
        <DialogContent className="sm:max-w-md">
          <DialogHeader>
            <DialogTitle>Publish Skill</DialogTitle>
            <DialogDescription>
              Tag the current <code className="text-foreground/70">main</code> branch as a new version.
            </DialogDescription>
          </DialogHeader>
          <div className="py-2">
            <label className="block mb-1 text-sm text-muted-foreground font-display">Tag name</label>
            <Input
              value={publishTag}
              onChange={(e) => setPublishTag(e.target.value)}
              placeholder="v1.0.0"
              className="font-mono"
              autoFocus
              onKeyDown={(e) => { if (e.key === "Enter") handlePublish(); }}
            />
            {publishError && (
              <div className="text-destructive text-xs mt-1">{publishError}</div>
            )}
          </div>
          <DialogFooter>
            <Button variant="outline" disabled={publishing} onClick={() => setShowPublish(false)}>
              Cancel
            </Button>
            <Button disabled={publishing || !publishTag.trim()} onClick={handlePublish}>
              {publishing && <Loader2 className="animate-spin mr-1.5" size={14} />}
              Publish
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
}

function SkillEmptyState() {
  return (
    <div className="flex-1 flex items-center justify-center text-muted-foreground/40">
      <div className="text-center">
        <div className="text-3xl mb-2">&#x1F4C4;</div>
        <div className="font-display text-sm">Select a skill to view its content</div>
        <div className="text-[10px] mt-1 max-w-75 leading-relaxed text-muted-foreground/30">
          Skills are markdown files that define agent capabilities, workflows,
          escalation rules, and behavioral guidelines.
        </div>
      </div>
    </div>
  );
}

type RightPanel =
  | { kind: "empty" }
  | { kind: "detail"; skill: Skill }
  | { kind: "create" }
  | { kind: "edit"; skill: Skill };

export function SkillsTab() {
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [panel, setPanel] = useState<RightPanel>({ kind: "empty" });

  const skills = useAppStore((s) => s.skills);
  const activeProjectId = useAppStore((s) => s.activeProjectId);
  const fetchSkills = useAppStore((s) => s.fetchSkills);
  const createSkill = useAppStore((s) => s.createSkill);
  const updateSkill = useAppStore((s) => s.updateSkill);
  const deleteSkill = useAppStore((s) => s.deleteSkill);
  const publishSkill = useAppStore((s) => s.publishSkill);

  useEffect(() => {
    fetchSkills(activeProjectId ?? undefined);
  }, [fetchSkills, activeProjectId]);

  const handleSelect = (id: string) => {
    const skill = skills.find((s) => s.id === id);
    if (!skill) return;
    setSelectedId(id);
    setPanel({ kind: "detail", skill });
  };

  const handleCreate = () => {
    setSelectedId(null);
    setPanel({ kind: "create" });
  };

  const handleEdit = (skill: Skill) => {
    setPanel({ kind: "edit", skill });
  };

  const handleClose = () => {
    setSelectedId(null);
    setPanel({ kind: "empty" });
  };

  const handleDelete = async (skillId: string) => {
    const ok = await deleteSkill(skillId);
    if (ok) {
      setSelectedId(null);
      setPanel({ kind: "empty" });
    }
  };

  const handleSave = async (data: SkillFormData) => {
    if (panel.kind === "edit") {
      const result = await updateSkill(panel.skill.id, {
        name: data.name,
        display_name: data.displayName,
        description: data.description,
        content: data.content,
        tags: data.tags,
      });
      if (result) {
        setSelectedId(result.id);
        setPanel({ kind: "detail", skill: result });
        return;
      }
    } else {
      const projectId = activeProjectId;
      if (!projectId) return;
      const result = await createSkill(projectId, {
        name: data.name,
        display_name: data.displayName,
        description: data.description,
        content: data.content,
        version: data.version,
        status: data.status,
        tags: data.tags,
      });
      if (result) {
        setSelectedId(result.id);
        setPanel({ kind: "detail", skill: result });
        return;
      }
    }
    setPanel({ kind: "empty" });
    setSelectedId(null);
  };

  return (
    <div className="flex-1 flex overflow-hidden">
      {/* List */}
      <div className="w-105 min-w-105 border-r border-border flex flex-col">
        <div className="px-4 py-3 border-b border-border flex justify-between items-center">
          <span className="text-sm text-muted-foreground">
            {skills.length} agent skill files
          </span>
          <Button onClick={handleCreate}>+ New Skill</Button>
        </div>
        <div className="flex-1 overflow-y-auto">
          {skills.map((s) => (
            <SkillListItem
              key={s.id}
              skill={s}
              selected={selectedId === s.id}
              onSelect={() => handleSelect(s.id)}
            />
          ))}
        </div>
      </div>

      {/* Right panel */}
      {panel.kind === "detail" && (
        <SkillDetail
          skill={panel.skill}
          onClose={handleClose}
          onEdit={() => handleEdit(panel.skill)}
          onDelete={() => handleDelete(panel.skill.id)}
          onPublish={(tag) => publishSkill(panel.skill.id, tag)}
        />
      )}
      {panel.kind === "create" && (
        <SkillForm onSave={handleSave} onCancel={handleClose} />
      )}
      {panel.kind === "edit" && (
        <SkillForm skill={panel.skill} onSave={handleSave} onCancel={handleClose} />
      )}
      {panel.kind === "empty" && <SkillEmptyState />}
    </div>
  );
}
