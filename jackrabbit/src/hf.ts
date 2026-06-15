// HuggingFace Hub API client (plain fetch; Node 20+ has global fetch). Search
// models/datasets and fetch a model's config/safetensors so we can show the
// "trainable by surogate?" badge without downloading weights.
//
// Endpoints: GET https://huggingface.co/api/models | /api/datasets
//   ?search=&author=&sort=downloads&direction=-1&limit=&expand[]=…
// Rate limits (Sep 2025): ~1000 req/5min with a token; 429 carries RateLimit
// headers. We debounce + cache upstream; a token raises the limit.

const BASE = "https://huggingface.co/api";

function authHeaders(): Record<string, string> {
  const t = process.env.HF_TOKEN || process.env.HUGGING_FACE_HUB_TOKEN || process.env.HUGGINGFACE_TOKEN;
  return t ? { Authorization: `Bearer ${t}` } : {};
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
  for (const f of ["downloads", "likes", "pipeline_tag", "gated", "tags"]) u.searchParams.append("expand[]", f);
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
}

/** Fetch a single model's config + safetensors metadata (no weight download). */
export async function modelDetail(id: string, signal?: AbortSignal): Promise<ModelDetail> {
  const u = new URL(`${BASE}/models/${id}`);
  for (const f of ["config", "safetensors", "downloads", "likes", "gated", "library_name", "pipeline_tag", "tags"]) {
    u.searchParams.append("expand[]", f);
  }
  const o = (await getJson(u, signal)) as Record<string, unknown>;
  const config = (o["config"] ?? {}) as Record<string, unknown>;
  const safetensors = (o["safetensors"] ?? {}) as Record<string, unknown>;
  return {
    id,
    architectures: Array.isArray(config["architectures"]) ? (config["architectures"] as string[]) : undefined,
    modelType: typeof config["model_type"] === "string" ? (config["model_type"] as string) : undefined,
    paramsBytes: typeof safetensors["total"] === "number" ? (safetensors["total"] as number) : undefined,
    downloads: typeof o["downloads"] === "number" ? (o["downloads"] as number) : undefined,
    likes: typeof o["likes"] === "number" ? (o["likes"] as number) : undefined,
    gated: o["gated"] as boolean | string | undefined,
    libraryName: typeof o["library_name"] === "string" ? (o["library_name"] as string) : undefined,
    pipelineTag: typeof o["pipeline_tag"] === "string" ? (o["pipeline_tag"] as string) : undefined,
  };
}

/** Params in billions from safetensors total bytes (bf16 ≈ 2 bytes/param). */
export function paramsB(detail: ModelDetail): number | null {
  if (!detail.paramsBytes) return null;
  return Math.round((detail.paramsBytes / 2 / 1e9) * 10) / 10;
}
