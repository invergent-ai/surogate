// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: AGPL-3.0-only
//
import { useEffect, useState } from "react";
import { toast } from "sonner";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { ConfirmDialog } from "@/components/ui/confirm-dialog";
import {
  Dialog,
  DialogContent,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";

import {
  createAgentUser,
  deleteAgentUser,
  listAgentUsers,
  updateAgentUser,
  type AgentUserResponse,
} from "@/api/agents";

// ── Utils ──────────────────────────────────────────────────────

function formatDate(iso: string | null): string {
  if (!iso) return "\u2014";
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return iso;
  return d.toISOString().slice(0, 10);
}

// ── Component ──────────────────────────────────────────────────

export function UsersTab({ agentId }: { agentId: string }) {
  const [users, setUsers] = useState<AgentUserResponse[]>([]);
  const [loading, setLoading] = useState(true);

  const [createOpen, setCreateOpen] = useState(false);
  const [editing, setEditing] = useState<AgentUserResponse | null>(null);
  const [deleting, setDeleting] = useState<AgentUserResponse | null>(null);

  async function refresh() {
    setLoading(true);
    try {
      const res = await listAgentUsers(agentId);
      setUsers(res.users);
    } catch (e) {
      toast.error((e as Error).message);
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    refresh();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [agentId]);

  return (
    <>
      <section className="bg-muted/40 border border-border rounded-lg overflow-hidden animate-in fade-in duration-150">
        <div className="px-4 py-3 border-b border-border flex items-center justify-between">
          <div className="flex items-center gap-2">
            <span className="text-blue-500">&#x1F464;</span>
            <span className="text-[13px] font-semibold text-foreground font-display">
              Users
            </span>
            <span className="text-[9px] px-1.5 py-px bg-muted rounded text-muted-foreground">
              {users.length}
            </span>
          </div>
          <Button
            variant="outline"
            size="xs"
            onClick={() => setCreateOpen(true)}
          >
            + Add User
          </Button>
        </div>

        <Table>
          <TableHeader>
            <TableRow>
              <TableHead>Email</TableHead>
              <TableHead>Display name</TableHead>
              <TableHead>Auth</TableHead>
              <TableHead>Created</TableHead>
              <TableHead />
            </TableRow>
          </TableHeader>
          <TableBody>
            {loading && (
              <TableRow>
                <TableCell
                  colSpan={5}
                  className="text-center text-xs text-muted-foreground py-6"
                >
                  Loading…
                </TableCell>
              </TableRow>
            )}
            {!loading && users.length === 0 && (
              <TableRow>
                <TableCell
                  colSpan={5}
                  className="text-center text-xs text-muted-foreground py-6"
                >
                  No users yet. Add one to let it log into this agent.
                </TableCell>
              </TableRow>
            )}
            {users.map((u) => (
              <TableRow key={u.id}>
                <TableCell className="text-xs font-medium">{u.email}</TableCell>
                <TableCell className="text-xs text-muted-foreground">
                  {u.display_name}
                </TableCell>
                <TableCell>
                  <Badge
                    variant={u.auth_provider === "database" ? "active" : "default"}
                  >
                    {u.auth_provider}
                  </Badge>
                </TableCell>
                <TableCell className="text-xs text-muted-foreground">
                  {formatDate(u.created_at)}
                </TableCell>
                <TableCell className="text-right">
                  <div className="flex justify-end gap-1">
                    <Button
                      variant="ghost"
                      size="xs"
                      onClick={() => setEditing(u)}
                    >
                      Edit
                    </Button>
                    <Button
                      variant="ghost"
                      size="xs"
                      onClick={() => setDeleting(u)}
                    >
                      Delete
                    </Button>
                  </div>
                </TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </section>

      {createOpen && (
        <CreateUserDialog
          agentId={agentId}
          onClose={() => setCreateOpen(false)}
          onCreated={() => {
            setCreateOpen(false);
            void refresh();
          }}
        />
      )}

      {editing && (
        <EditUserDialog
          agentId={agentId}
          user={editing}
          onClose={() => setEditing(null)}
          onSaved={() => {
            setEditing(null);
            void refresh();
          }}
        />
      )}

      <ConfirmDialog
        open={deleting !== null}
        title="Delete user?"
        description={
          deleting
            ? `This will permanently remove ${deleting.email} from the agent's organisation.`
            : ""
        }
        confirmLabel="Delete"
        onConfirm={async () => {
          if (!deleting) return;
          try {
            await deleteAgentUser(agentId, deleting.id);
            toast.success(`Deleted ${deleting.email}`);
            setDeleting(null);
            void refresh();
          } catch (e) {
            toast.error((e as Error).message);
          }
        }}
        onCancel={() => setDeleting(null)}
      />
    </>
  );
}

// ── Dialogs ────────────────────────────────────────────────────

function CreateUserDialog({
  agentId,
  onClose,
  onCreated,
}: {
  agentId: string;
  onClose: () => void;
  onCreated: () => void;
}) {
  const [email, setEmail] = useState("");
  const [displayName, setDisplayName] = useState("");
  const [password, setPassword] = useState("");
  const [saving, setSaving] = useState(false);

  const canSave =
    email.trim().length > 0 && displayName.trim().length > 0 && !saving;

  async function save() {
    setSaving(true);
    try {
      await createAgentUser(agentId, {
        email: email.trim(),
        display_name: displayName.trim(),
        password: password.length > 0 ? password : undefined,
      });
      toast.success(`Created ${email.trim()}`);
      onCreated();
    } catch (e) {
      toast.error((e as Error).message);
    } finally {
      setSaving(false);
    }
  }

  return (
    <Dialog open onOpenChange={(o) => !o && onClose()}>
      <DialogContent>
        <DialogHeader>
          <DialogTitle>Add user</DialogTitle>
        </DialogHeader>
        <div className="flex flex-col gap-3 py-2">
          <div className="flex flex-col gap-1.5">
            <Label htmlFor="new-user-email">Email</Label>
            <Input
              id="new-user-email"
              type="email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              placeholder="user@example.com"
              autoFocus
            />
          </div>
          <div className="flex flex-col gap-1.5">
            <Label htmlFor="new-user-display">Display name</Label>
            <Input
              id="new-user-display"
              value={displayName}
              onChange={(e) => setDisplayName(e.target.value)}
              placeholder="Ada Lovelace"
            />
          </div>
          <div className="flex flex-col gap-1.5">
            <Label htmlFor="new-user-password">
              Password{" "}
              <span className="text-muted-foreground font-normal">
                (optional &mdash; leave empty for SSO)
              </span>
            </Label>
            <Input
              id="new-user-password"
              type="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              placeholder="&bull;&bull;&bull;&bull;&bull;&bull;&bull;&bull;"
            />
          </div>
        </div>
        <DialogFooter>
          <Button variant="outline" size="sm" onClick={onClose}>
            Cancel
          </Button>
          <Button size="sm" disabled={!canSave} onClick={save}>
            {saving ? "Saving…" : "Create"}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}

function EditUserDialog({
  agentId,
  user,
  onClose,
  onSaved,
}: {
  agentId: string;
  user: AgentUserResponse;
  onClose: () => void;
  onSaved: () => void;
}) {
  const [displayName, setDisplayName] = useState(user.display_name);
  const [password, setPassword] = useState("");
  const [saving, setSaving] = useState(false);

  const changed =
    displayName.trim() !== user.display_name || password.length > 0;

  async function save() {
    setSaving(true);
    try {
      await updateAgentUser(agentId, user.id, {
        display_name:
          displayName.trim() !== user.display_name
            ? displayName.trim()
            : undefined,
        password: password.length > 0 ? password : undefined,
      });
      toast.success(`Updated ${user.email}`);
      onSaved();
    } catch (e) {
      toast.error((e as Error).message);
    } finally {
      setSaving(false);
    }
  }

  return (
    <Dialog open onOpenChange={(o) => !o && onClose()}>
      <DialogContent>
        <DialogHeader>
          <DialogTitle>Edit user</DialogTitle>
        </DialogHeader>
        <div className="flex flex-col gap-3 py-2">
          <div className="flex flex-col gap-1.5">
            <Label>Email</Label>
            <Input value={user.email} disabled />
          </div>
          <div className="flex flex-col gap-1.5">
            <Label htmlFor="edit-user-display">Display name</Label>
            <Input
              id="edit-user-display"
              value={displayName}
              onChange={(e) => setDisplayName(e.target.value)}
              autoFocus
            />
          </div>
          <div className="flex flex-col gap-1.5">
            <Label htmlFor="edit-user-password">
              New password{" "}
              <span className="text-muted-foreground font-normal">
                (leave empty to keep current)
              </span>
            </Label>
            <Input
              id="edit-user-password"
              type="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              placeholder="&bull;&bull;&bull;&bull;&bull;&bull;&bull;&bull;"
            />
          </div>
        </div>
        <DialogFooter>
          <Button variant="outline" size="sm" onClick={onClose}>
            Cancel
          </Button>
          <Button size="sm" disabled={!changed || saving} onClick={save}>
            {saving ? "Saving…" : "Save"}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
