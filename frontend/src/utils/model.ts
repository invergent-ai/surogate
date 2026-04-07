import type { Model } from "@/types/model";

export function isProxyModel(model: Model): boolean {
  return model.source === "openrouter" || model.source === "openai_compat";
}