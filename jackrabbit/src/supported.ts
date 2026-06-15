// Which HuggingFace models surogate can train. Mirrors the framework's DSL model
// registry (resolved from config.json `architectures` / `model_type`, exactly how
// vLLM picks a model class). Source: surogate/dsl/models/* + docs/guides.

export interface Family {
  type: string; // HF model_type
  archs: string[]; // HF architectures class names
  label: string;
  moe?: boolean;
  vision?: boolean;
}

export const FAMILIES: Family[] = [
  { type: "llama", archs: ["LlamaForCausalLM"], label: "Llama" },
  { type: "qwen3", archs: ["Qwen3ForCausalLM"], label: "Qwen3" },
  { type: "qwen3_moe", archs: ["Qwen3MoeForCausalLM"], label: "Qwen3-MoE", moe: true },
  { type: "qwen3_5_text", archs: ["Qwen3_5ForCausalLM"], label: "Qwen3.5" },
  { type: "qwen3_5", archs: ["Qwen3_5ForConditionalGeneration"], label: "Qwen3.5-VL", vision: true },
  { type: "qwen3_5_moe_text", archs: ["Qwen3_5MoeForCausalLM"], label: "Qwen3.5-MoE", moe: true },
  { type: "qwen3_5_moe", archs: ["Qwen3_5MoeForConditionalGeneration"], label: "Qwen3.5-MoE-VL", moe: true, vision: true },
  { type: "qwen3_vl", archs: ["Qwen3VLForConditionalGeneration"], label: "Qwen3-VL", vision: true },
  { type: "gemma4_text", archs: ["Gemma4ForCausalLM"], label: "Gemma4" },
  { type: "gemma4", archs: ["Gemma4ForConditionalGeneration"], label: "Gemma4-VL", vision: true },
  { type: "gemma4_unified", archs: ["Gemma4UnifiedForConditionalGeneration"], label: "Gemma4-Unified", vision: true },
  { type: "gpt_oss", archs: ["GptOssForCausalLM"], label: "GPT-OSS", moe: true },
  { type: "lfm2", archs: ["Lfm2ForCausalLM"], label: "LFM2" },
  { type: "nemotron_h", archs: ["NemotronHForCausalLM"], label: "Nemotron-H" },
];

// All recipes apply to every supported family; the GPU generation (not the model)
// gates fp8 (SM89+) and nvfp4 (Blackwell SM100+).
export const ALL_RECIPES = ["bf16", "fp8-hybrid", "nvfp4", "nvfp4_quartet"] as const;

export interface Trainability {
  supported: boolean;
  family: Family | null;
  moe: boolean;
  vision: boolean;
  recipes: readonly string[];
  reason: string;
}

const ARCH_INDEX = new Map<string, Family>();
const TYPE_INDEX = new Map<string, Family>();
for (const f of FAMILIES) {
  TYPE_INDEX.set(f.type.toLowerCase(), f);
  for (const a of f.archs) ARCH_INDEX.set(a.toLowerCase(), f);
}

/** Match a model's architectures / model_type against the supported registry. */
export function checkTrainable(architectures: string[] | undefined, modelType: string | undefined): Trainability {
  let fam: Family | undefined;
  for (const a of architectures ?? []) {
    fam = ARCH_INDEX.get(a.toLowerCase());
    if (fam) break;
  }
  if (!fam && modelType) fam = TYPE_INDEX.get(modelType.toLowerCase());

  if (!fam) {
    const what = architectures?.[0] ?? modelType ?? "unknown";
    return {
      supported: false,
      family: null,
      moe: false,
      vision: false,
      recipes: [],
      reason: `${what} — not in surogate's supported architectures`,
    };
  }
  return {
    supported: true,
    family: fam,
    moe: !!fam.moe,
    vision: !!fam.vision,
    recipes: ALL_RECIPES,
    reason: `${fam.label}${fam.moe ? " · MoE" : ""}${fam.vision ? " · vision" : ""} — trainable`,
  };
}
