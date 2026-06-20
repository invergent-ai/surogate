// A registry of compute providers the user has set up — SSH GPU servers, Modal
// workspaces, and dstack cloud backends (RunPod / Lambda / Vast / AWS / GCP /
// Azure / Nebius / OCI). This is the "add · save · delete · pick" backbone the
// GPUs/Setup screens drive.
//
// IMPORTANT: only NON-SECRET metadata lives here (labels, hostnames, backend
// type, region, preferred GPU). Real credentials are written to each tool's own
// store when you connect — Modal → ~/.modal.toml (`modal token set`), SSH →
// ~/.ssh, dstack → its own encrypted backend config — so API keys are never
// duplicated into a plaintext file we own.
import fs from "node:fs";
import os from "node:os";
import path from "node:path";

export type ProviderKind = "ssh" | "modal" | "dstack";
export type DstackBackend = "runpod" | "lambda" | "vastai" | "aws" | "gcp" | "azure" | "nebius" | "oci";

// What each cloud backend needs to connect (shown in the UI; captured then handed
// to dstack, never stored here). Grounded in dstack's backend credential types.
export const DSTACK_BACKENDS: { id: DstackBackend; label: string; needs: string[] }[] = [
  { id: "runpod", label: "RunPod", needs: ["API key"] },
  { id: "lambda", label: "Lambda", needs: ["API key"] },
  { id: "vastai", label: "Vast.ai", needs: ["API key"] },
  { id: "aws", label: "AWS", needs: ["access key id", "secret access key"] },
  { id: "gcp", label: "GCP", needs: ["project id", "service-account JSON"] },
  { id: "azure", label: "Azure", needs: ["tenant id", "subscription id", "client id", "client secret"] },
  { id: "nebius", label: "Nebius", needs: ["service account id", "public key id", "private key"] },
  { id: "oci", label: "OCI", needs: ["user/tenancy OCID", "private key", "fingerprint", "region"] },
];

export interface ProviderRecord {
  id: string;
  kind: ProviderKind;
  label: string;
  host?: string; // ssh: user@host[:port] or a ~/.ssh/config alias
  workspace?: string; // modal: workspace name (from token verify)
  environment?: string; // modal: environment name
  backend?: DstackBackend; // dstack: which cloud
  region?: string; // dstack: preferred region
  gpu?: string; // preferred GPU type/count, e.g. "H100" or "H100:2"
  addedAt: string; // ISO timestamp
}

const file = (): string => path.join(os.homedir(), ".surogate-watch", "providers.json");

export function loadProviders(): ProviderRecord[] {
  try {
    const v = JSON.parse(fs.readFileSync(file(), "utf8"));
    return Array.isArray(v) ? (v as ProviderRecord[]).filter((p) => p && p.id && p.kind && p.label) : [];
  } catch {
    return [];
  }
}

function writeAll(list: ProviderRecord[]): void {
  try {
    fs.mkdirSync(path.dirname(file()), { recursive: true });
    fs.writeFileSync(file(), JSON.stringify(list, null, 2));
  } catch {
    /* best effort */
  }
}

/** Insert or update a provider (matched by id); newest entries sort last. */
export function saveProvider(p: ProviderRecord): ProviderRecord[] {
  const list = loadProviders().filter((x) => x.id !== p.id);
  list.push(p);
  writeAll(list);
  return list;
}

export function deleteProvider(id: string): ProviderRecord[] {
  const list = loadProviders().filter((x) => x.id !== id);
  writeAll(list);
  return list;
}

export function getProvider(id: string): ProviderRecord | undefined {
  return loadProviders().find((x) => x.id === id);
}

/** Stable id from kind + a key (host/workspace/backend), so re-adding the same
 *  target updates it in place rather than duplicating. The key is kept verbatim
 *  (only trimmed) — slugging it would collapse distinct hosts like `a-1` and
 *  `a_1` to the same id and silently overwrite one. ids are plain JSON strings
 *  (never used as file paths), so arbitrary characters are fine. */
export function providerId(kind: ProviderKind, key: string): string {
  return `${kind}:${key.trim() || "default"}`;
}
