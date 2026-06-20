// HuggingFace Hub API client (plain fetch; Node 20+ has global fetch). Search
// models/datasets and fetch a model's config/safetensors so we can show the
// "trainable by surogate?" badge without downloading weights.
//
// Endpoints: GET https://huggingface.co/api/models | /api/datasets
//   ?search=&author=&sort=downloads&direction=-1&limit=&expand[]=…
// Rate limits (Sep 2025): ~1000 req/5min with a token; 429 carries RateLimit
// headers. We debounce + cache upstream; a token raises the limit.

import fs from "node:fs";
import os from "node:os";
import path from "node:path";

const BASE = "https://huggingface.co/api";

// Resolve the Hub token once: env vars first, then the standard huggingface_hub
// CLI token file (written by `huggingface-cli login`). Without this, a user who
// is logged in via the CLI but has no HF_TOKEN env var still hits the much lower
// anonymous rate limit (429s that look like "search is broken"). `null` = looked
// and found nothing (don't re-read the filesystem on every request).
let cliToken: string | null | undefined;
function hfToken(): string | undefined {
  const env = process.env.HF_TOKEN || process.env.HUGGING_FACE_HUB_TOKEN || process.env.HUGGINGFACE_TOKEN;
  if (env) return env;
  if (cliToken === undefined) {
    cliToken = null;
    const home = os.homedir();
    const candidates = [
      process.env.HF_TOKEN_PATH,
      process.env.HF_HOME ? path.join(process.env.HF_HOME, "token") : undefined,
      path.join(home, ".cache", "huggingface", "token"),
      path.join(home, ".huggingface", "token"),
    ].filter((p): p is string => !!p);
    for (const p of candidates) {
      try {
        const t = fs.readFileSync(p, "utf8").trim();
        if (t) {
          cliToken = t;
          break;
        }
      } catch {
        /* missing — try the next candidate */
      }
    }
  }
  return cliToken ?? undefined;
}

function authHeaders(): Record<string, string> {
  const t = hfToken();
  return t ? { Authorization: `Bearer ${t}` } : {};
}

/** Is a Hub token configured (env var or CLI token file)? */
export function hasHfToken(): boolean {
  return !!hfToken();
}

/** Validate a pasted token (whoami) and, if valid, persist it to the standard
 *  huggingface_hub token file so the CLI/surogate use it too, and refresh our
 *  in-process cache. Returns the account name on success. */
export async function validateAndSaveHfToken(token: string): Promise<{ ok: boolean; user?: string; reason?: string }> {
  const t = token.trim();
  if (!t) return { ok: false, reason: "empty token" };
  try {
    const res = await fetch(`${BASE}/whoami-v2`, { headers: { Authorization: `Bearer ${t}` } });
    if (!res.ok) return { ok: false, reason: res.status === 401 ? "invalid token" : `HF ${res.status}` };
    const who = (await res.json()) as { name?: string };
    const file = process.env.HF_TOKEN_PATH || path.join(os.homedir(), ".cache", "huggingface", "token");
    fs.mkdirSync(path.dirname(file), { recursive: true });
    fs.writeFileSync(file, t, { mode: 0o600 });
    cliToken = t; // refresh the cache so subsequent requests are authed immediately
    return { ok: true, user: who.name };
  } catch (e) {
    return { ok: false, reason: (e as Error).message };
  }
}

export interface HfHit {
  id: string;
  downloads?: number;
  likes?: number;
  pipelineTag?: string;
  gated?: boolean | string;
  tags?: string[];
}

export type Sort = "downloads" | "likes" | "trendingScore" | "lastModified";

async function getJson(url: URL, signal?: AbortSignal): Promise<unknown> {
  const res = await fetch(url, { headers: { Accept: "application/json", ...authHeaders() }, signal });
  if (res.status === 429) throw new Error("rate limited (429) — set HF_TOKEN or slow down");
  if (!res.ok) throw new Error(`HF API ${res.status}`);
  return res.json();
}

function toHit(o: Record<string, unknown>): HfHit {
  return {
    id: String(o["id"] ?? o["modelId"] ?? ""),
    downloads: typeof o["downloads"] === "number" ? (o["downloads"] as number) : undefined,
    likes: typeof o["likes"] === "number" ? (o["likes"] as number) : undefined,
    pipelineTag: typeof o["pipeline_tag"] === "string" ? (o["pipeline_tag"] as string) : undefined,
    gated: o["gated"] as boolean | string | undefined,
    tags: Array.isArray(o["tags"]) ? (o["tags"] as string[]) : undefined,
  };
}

async function search(kind: "models" | "datasets", query: string, opts: { limit?: number; sort?: Sort; signal?: AbortSignal } = {}): Promise<HfHit[]> {
  const u = new URL(`${BASE}/${kind}`);
  if (query) u.searchParams.set("search", query);
  u.searchParams.set("limit", String(opts.limit ?? 25));
  u.searchParams.set("sort", opts.sort ?? "downloads");
  u.searchParams.set("direction", "-1");
  // expand fields differ per endpoint — `pipeline_tag` is model-only; sending it
  // to /datasets returns HTTP 400.
  const expand = kind === "models" ? ["downloads", "likes", "pipeline_tag", "gated", "tags"] : ["downloads", "likes", "gated", "tags"];
  for (const f of expand) u.searchParams.append("expand[]", f);
  const arr = (await getJson(u, opts.signal)) as Record<string, unknown>[];
  return Array.isArray(arr) ? arr.map(toHit) : [];
}

export const searchModels = (q: string, o?: { limit?: number; sort?: Sort; signal?: AbortSignal }) => search("models", q, o);
export const searchDatasets = (q: string, o?: { limit?: number; sort?: Sort; signal?: AbortSignal }) => search("datasets", q, o);

export interface ModelDetail {
  id: string;
  architectures?: string[];
  modelType?: string;
  paramsBytes?: number; // safetensors.total (bytes)
  downloads?: number;
  likes?: number;
  gated?: boolean | string;
  libraryName?: string;
  pipelineTag?: string;
  tags?: string[];
  files?: string[]; // sibling filenames
  lastModified?: string;
}

function strArr(v: unknown): string[] | undefined {
  return Array.isArray(v) ? (v as unknown[]).filter((x) => typeof x === "string") as string[] : undefined;
}
function siblingNames(v: unknown): string[] | undefined {
  if (!Array.isArray(v)) return undefined;
  return (v as Array<Record<string, unknown>>).map((s) => String(s["rfilename"] ?? "")).filter(Boolean);
}

// Fetch a raw repo file (config.json / adapter_config.json) — the source of truth
// when the API's `config` expand is empty (common for fine-tunes / LoRA adapters).
async function getRepoJson(repo: string, file: string, signal?: AbortSignal): Promise<Record<string, unknown> | null> {
  try {
    const res = await fetch(`https://huggingface.co/${repo}/resolve/main/${file}`, { headers: { ...authHeaders() }, signal });
    if (!res.ok) return null;
    return (await res.json()) as Record<string, unknown>;
  } catch {
    return null;
  }
}

function archFromConfig(cfg: Record<string, unknown> | null): { architectures?: string[]; modelType?: string } {
  if (!cfg) return {};
  return {
    architectures: strArr(cfg["architectures"]),
    modelType: typeof cfg["model_type"] === "string" ? (cfg["model_type"] as string) : undefined,
  };
}

/** When the API gave us no architecture, read it from the repo's actual files:
 *  config.json, or follow a LoRA adapter's base model. */
async function detectArchFromFiles(id: string, files: string[] | undefined, signal?: AbortSignal): Promise<{ architectures?: string[]; modelType?: string }> {
  const has = (f: string) => files?.some((x) => x === f || x.endsWith(`/${f}`));
  if (has("config.json")) {
    const got = archFromConfig(await getRepoJson(id, "config.json", signal));
    if (got.architectures?.length || got.modelType) return got;
  }
  if (has("adapter_config.json")) {
    const adapter = await getRepoJson(id, "adapter_config.json", signal);
    const base = adapter && typeof adapter["base_model_name_or_path"] === "string" ? (adapter["base_model_name_or_path"] as string) : null;
    if (base) return archFromConfig(await getRepoJson(base, "config.json", signal));
  }
  return {};
}

/** Fetch a single model's config + safetensors + metadata (no weight download).
 *  `deep` enables the extra config.json/adapter-base requests that recover the
 *  architecture for fine-tunes/adapters — costly, so callers browsing a list pass
 *  false (one shallow request per row) and only resolve deeply on explicit pick. */
export async function modelDetail(id: string, signal?: AbortSignal, deep = false): Promise<ModelDetail> {
  const u = new URL(`${BASE}/models/${id}`);
  for (const f of ["config", "safetensors", "downloads", "likes", "gated", "library_name", "pipeline_tag", "tags", "siblings", "lastModified"]) {
    u.searchParams.append("expand[]", f);
  }
  const o = (await getJson(u, signal)) as Record<string, unknown>;
  const config = (o["config"] ?? {}) as Record<string, unknown>;
  const safetensors = (o["safetensors"] ?? {}) as Record<string, unknown>;
  const files = siblingNames(o["siblings"]);
  let architectures = strArr(config["architectures"]);
  let modelType = typeof config["model_type"] === "string" ? (config["model_type"] as string) : undefined;
  // The API's config expand is often empty for fine-tunes / adapters — read the
  // repo's real config.json (or the adapter's base model) to detect the arch.
  // Only on a deep (explicit-pick) fetch, to keep list browsing to one request/row.
  if (deep && !architectures?.length && !modelType) {
    const detected = await detectArchFromFiles(id, files, signal);
    architectures = detected.architectures;
    modelType = detected.modelType;
  }
  return {
    id,
    architectures,
    modelType,
    paramsBytes: typeof safetensors["total"] === "number" ? (safetensors["total"] as number) : undefined,
    downloads: typeof o["downloads"] === "number" ? (o["downloads"] as number) : undefined,
    likes: typeof o["likes"] === "number" ? (o["likes"] as number) : undefined,
    gated: o["gated"] as boolean | string | undefined,
    libraryName: typeof o["library_name"] === "string" ? (o["library_name"] as string) : undefined,
    pipelineTag: typeof o["pipeline_tag"] === "string" ? (o["pipeline_tag"] as string) : undefined,
    tags: strArr(o["tags"]),
    files,
    lastModified: typeof o["lastModified"] === "string" ? (o["lastModified"] as string) : undefined,
  };
}

export interface DatasetDetail {
  id: string;
  downloads?: number;
  likes?: number;
  gated?: boolean | string;
  tags?: string[];
  files?: string[];
  lastModified?: string;
  description?: string;
}

/** Fetch a single dataset's metadata + file list (no data download). */
export async function datasetDetail(id: string, signal?: AbortSignal): Promise<DatasetDetail> {
  const u = new URL(`${BASE}/datasets/${id}`);
  for (const f of ["downloads", "likes", "gated", "tags", "siblings", "lastModified", "cardData"]) {
    u.searchParams.append("expand[]", f);
  }
  const o = (await getJson(u, signal)) as Record<string, unknown>;
  const card = (o["cardData"] ?? {}) as Record<string, unknown>;
  return {
    id,
    downloads: typeof o["downloads"] === "number" ? (o["downloads"] as number) : undefined,
    likes: typeof o["likes"] === "number" ? (o["likes"] as number) : undefined,
    gated: o["gated"] as boolean | string | undefined,
    tags: strArr(o["tags"]),
    files: siblingNames(o["siblings"]),
    lastModified: typeof o["lastModified"] === "string" ? (o["lastModified"] as string) : undefined,
    description: typeof card["pretty_name"] === "string" ? (card["pretty_name"] as string) : undefined,
  };
}

/** Params in billions from safetensors total bytes (bf16 ≈ 2 bytes/param). */
export function paramsB(detail: ModelDetail): number | null {
  if (!detail.paramsBytes) return null;
  return Math.round((detail.paramsBytes / 2 / 1e9) * 10) / 10;
}
