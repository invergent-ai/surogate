// Which precision recipes surogate can run on a given GPU, gated by compute
// capability (SM). Authoritative source: surogate/dsl/quantization.py —
//   bf16 + QLoRA-NF4 : any CUDA GPU
//   fp8-hybrid       : FP8 E4M3 → SM89+ (Ada / Hopper)
//   nvfp4(+quartet)  : NVFP4 E2M1 → SM100+ (Blackwell)

export interface GpuGen {
  arch: string;
  sm: number; // minimum SM for this generation
  examples: string;
  recipes: string[];
}

// Newest → oldest. `recipes` are the surogate recipes that generation unlocks.
export const GPU_GENS: GpuGen[] = [
  { arch: "Blackwell", sm: 100, examples: "B200 · GB200 · RTX 5090", recipes: ["bf16", "fp8-hybrid", "nvfp4", "nvfp4_quartet"] },
  { arch: "Hopper", sm: 90, examples: "H100 · H200", recipes: ["bf16", "fp8-hybrid"] },
  { arch: "Ada", sm: 89, examples: "RTX 4090 · L4 · L40S · RTX 6000 Ada", recipes: ["bf16", "fp8-hybrid"] },
  { arch: "Ampere", sm: 80, examples: "A100 · A10G · A40 · RTX 3090", recipes: ["bf16", "QLoRA-NF4"] },
  { arch: "Turing / Volta", sm: 70, examples: "T4 · V100 · RTX 2080", recipes: ["bf16", "QLoRA-NF4"] },
];

export function gpuArch(sm: number | null): string {
  if (sm === null) return "unknown";
  if (sm >= 100) return "Blackwell";
  if (sm >= 90) return "Hopper";
  if (sm >= 89) return "Ada";
  if (sm >= 80) return "Ampere";
  if (sm >= 70) return "Turing/Volta";
  return "pre-Volta";
}

/** Precision recipes surogate can run on a GPU of compute capability `sm`. */
export function recipesForSm(sm: number | null): string[] {
  const r = ["bf16"];
  if (sm !== null && sm >= 89) r.push("fp8-hybrid");
  if (sm !== null && sm >= 100) r.push("nvfp4", "nvfp4_quartet");
  return r;
}

/** A compact support summary for a GPU: arch, recipes, and the best one. */
export function gpuSupport(sm: number | null): { arch: string; recipes: string[]; best: string } {
  const recipes = recipesForSm(sm);
  const best = recipes.includes("nvfp4") ? "NVFP4 4-bit" : recipes.includes("fp8-hybrid") ? "fp8-hybrid" : "bf16 · QLoRA";
  return { arch: gpuArch(sm), recipes, best };
}
