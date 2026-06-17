import React, { useEffect, useState } from "react";
import { ShimmerText } from "./Shimmer.tsx";

// Playful, surogate-flavored "working…" lines (à la PostHog's Max thinking
// verbs) shown while a run spins up and the feed hasn't started yet. Rotated so
// the wait feels alive rather than stuck.
export const STATUS_VERBS = [
  // real spin-up work
  "Warming up the GPUs",
  "Spinning up CUDA",
  "Quantizing weights",
  "Packing sequences",
  "Sharding optimizer states",
  "Compiling kernels",
  "Loading the tokenizer",
  "Calibrating fp8 scales",
  "Priming the dataloader",
  "Fusing cross-entropy",
  "Counting parameters",
  "Marshalling tensors",
  "Wiring up the feed",
  "Stretching the adapters",
  "Allocating VRAM",
  "Booting vLLM",
  "Building the KV cache",
  "Casting to NVFP4",
  "Pinning host memory",
  "Negotiating with NCCL",
  // speed-of-light / throughput flavor
  "Chasing the speed of light",
  "Approaching SOL%",
  "Squeezing more tok/s",
  "Saturating the tensor cores",
  "Melting the FLOPs barrier",
  "Overclocking the gradients",
  "Redlining the GEMMs",
  // bunny / jackalope flavor 🐰🦌
  "Feeding the jackalope",
  "Letting the bunny loose",
  "Hopping through batches",
  "Twitching whiskers",
  "Growing the antlers",
  "Bounding across epochs",
  "Nibbling on tokens",
  "Out-running the loss",
  "Burrowing into the data",
  "Zooming past the baseline",
  "Going down the rabbit hole",
  "Following the carrot",
  // playful classics
  "Pondering",
  "Crunching",
  "Galumphing",
  "Limbering up",
  "Greasing the gradients",
  "Reticulating splines",
  "Summoning the optimizer",
];

/** A shimmering, rotating status line. `intervalMs` controls how fast verbs
 *  cycle; `prefix` is prepended (e.g. "GRPO · "). */
export function StatusVerb({ prefix = "", intervalMs = 1900 }: { prefix?: string; intervalMs?: number }) {
  const [i, setI] = useState(0);
  useEffect(() => {
    const id = setInterval(() => setI((x) => x + 1), intervalMs);
    return () => clearInterval(id);
  }, [intervalMs]);
  return <ShimmerText text={`${prefix}${STATUS_VERBS[i % STATUS_VERBS.length]}…`} />;
}
