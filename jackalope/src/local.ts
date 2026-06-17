// Local model / dataset support. surogate loads a path locally whenever it
// exists on disk (a model folder with config.json; a dataset dir via
// load_from_disk/load_dataset, or a single file keyed by extension) and only
// falls back to the HuggingFace Hub otherwise. These helpers resolve a typed
// path and inspect it so the Browse UI can offer "use this local model/dataset"
// right alongside Hub search.

import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { checkTrainable, type Trainability } from "./supported.ts";

/** A typed string that looks like a filesystem path (vs a Hub "org/name" id). */
export function looksLikePath(s: string): boolean {
  return /^(~|\.\.?\/|\/)/.test(s.trim());
}

/** Expand a leading ~ and resolve to an absolute path. */
export function resolveLocalPath(p: string): string {
  let s = p.trim();
  if (s === "~" || s.startsWith("~/")) s = path.join(os.homedir(), s.slice(1));
  return path.resolve(s);
}

export interface LocalModel {
  path: string;
  exists: boolean;
  isDir: boolean;
  architectures?: string[];
  modelType?: string;
  paramsB: number | null;
  trainability: Trainability;
  error?: string;
}

/** ~params in billions from the on-disk safetensors size (bf16 ≈ 2 bytes/param). */
function localParamsB(dir: string): number | null {
  try {
    let bytes = 0;
    for (const f of fs.readdirSync(dir)) {
      if (f.endsWith(".safetensors")) bytes += fs.statSync(path.join(dir, f)).size;
    }
    return bytes > 0 ? Math.round((bytes / 2 / 1e9) * 10) / 10 : null;
  } catch {
    return null;
  }
}

/** Inspect a local model folder: read config.json → architectures/model_type →
 *  surogate trainability, plus an approximate parameter count from disk. */
export function localModelInfo(input: string): LocalModel {
  const abs = resolveLocalPath(input);
  const untrainable = checkTrainable(undefined, undefined);
  let st: fs.Stats | null = null;
  try {
    st = fs.statSync(abs);
  } catch {
    return { path: abs, exists: false, isDir: false, paramsB: null, trainability: untrainable, error: "path not found" };
  }
  if (!st.isDirectory()) {
    return {
      path: abs,
      exists: true,
      isDir: false,
      paramsB: null,
      trainability: untrainable,
      error: "not a folder — point at a model directory containing config.json",
    };
  }
  let architectures: string[] | undefined;
  let modelType: string | undefined;
  let error: string | undefined;
  try {
    const cfg = JSON.parse(fs.readFileSync(path.join(abs, "config.json"), "utf8")) as Record<string, unknown>;
    architectures = Array.isArray(cfg["architectures"]) ? (cfg["architectures"] as string[]) : undefined;
    modelType = typeof cfg["model_type"] === "string" ? (cfg["model_type"] as string) : undefined;
  } catch {
    error = "no readable config.json — not a HuggingFace model folder";
  }
  return {
    path: abs,
    exists: true,
    isDir: true,
    architectures,
    modelType,
    paramsB: localParamsB(abs),
    trainability: checkTrainable(architectures, modelType),
    error,
  };
}

// Maps a dataset file extension to the HF `datasets` loader type surogate uses.
const EXT_TYPE: Record<string, string> = {
  ".jsonl": "json",
  ".json": "json",
  ".parquet": "parquet",
  ".csv": "csv",
  ".arrow": "arrow",
  ".txt": "text",
};

export interface LocalDataset {
  path: string;
  exists: boolean;
  kind: "dir" | "file" | null;
  dsType: string | null; // inferred loader type for a single file
  sizeBytes: number;
  error?: string;
}

function topLevelSize(dir: string): number {
  try {
    let bytes = 0;
    for (const f of fs.readdirSync(dir, { withFileTypes: true })) {
      if (f.isFile()) {
        try {
          bytes += fs.statSync(path.join(dir, f.name)).size;
        } catch {
          /* skip */
        }
      }
    }
    return bytes;
  } catch {
    return 0;
  }
}

/** Inspect a local dataset path: a directory (load_from_disk / load_dataset) or
 *  a single file whose extension chooses the loader (jsonl/parquet/csv/…). */
export function localDatasetInfo(input: string): LocalDataset {
  const abs = resolveLocalPath(input);
  let st: fs.Stats | null = null;
  try {
    st = fs.statSync(abs);
  } catch {
    return { path: abs, exists: false, kind: null, dsType: null, sizeBytes: 0, error: "path not found" };
  }
  if (st.isDirectory()) {
    return { path: abs, exists: true, kind: "dir", dsType: null, sizeBytes: topLevelSize(abs) };
  }
  const ext = path.extname(abs).toLowerCase();
  return { path: abs, exists: true, kind: "file", dsType: EXT_TYPE[ext] ?? "json", sizeBytes: st.size };
}
