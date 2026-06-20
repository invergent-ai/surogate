// Grounded, real-time tips feed — facts distilled from the surogate docs
// (guides: precision-and-recipes, qlora, offloading, memory, optimizers,
// performance, moe, rl-training, long-context, multi-gpu, metrics).
//
// Each tip may carry context `tags`; the rail surfaces tips whose tags match
// the live run (recipe, lora, moe, grpo, hot GPU, …) plus general tips, and
// rotates through them over time so it reads like a live feed.

export type TipTag =
  | "bf16"
  | "fp8"
  | "nvfp4"
  | "qlora"
  | "lora"
  | "fft"
  | "moe"
  | "grpo"
  | "memtight"
  | "memroom"
  | "hot"
  | "lowutil"
  | "plateau"
  | "diverge"
  | "longseq"
  | "multigpu"
  | "throughput";

export interface Tip {
  t: string;
  tags?: TipTag[];
}

// Concise so they fit the narrow rail. Grounded in surogate's documented behavior.
export const TIPS: Tip[] = [
  // ---- recipes / precision ----
  { t: "bf16 = max accuracy, most VRAM — the safe default", tags: ["bf16"] },
  { t: "fp8-hybrid keeps a bf16 master; ~1.5–2× faster GEMMs", tags: ["fp8"] },
  { t: "fp8 uses delayed scaling — fp8_amax_history smooths spikes", tags: ["fp8"] },
  { t: "if fp8 loss spikes, raise fp8_amax_history (default 16)", tags: ["fp8", "diverge"] },
  { t: "NVFP4 needs Blackwell (SM100+); 4-bit weights + scales", tags: ["nvfp4"] },
  { t: "NVFP4 Four-Over-Six (4/6) is on by default — lower quant error", tags: ["nvfp4"] },
  { t: "fp4_backend: cutlass (default) or cudnn — try cudnn if slow", tags: ["nvfp4"] },
  { t: "FP4 weight caching is on by default on SM100+ (no per-fwd quant)", tags: ["nvfp4"] },
  { t: "skip_quant_first/last_layers keeps embed/lm_head in bf16 for stability", tags: ["fp8", "nvfp4"] },
  { t: "want speed without accuracy loss? fp8-hybrid before nvfp4", tags: ["bf16"] },
  // ---- qlora ----
  { t: "QLoRA freezes base weights quantized, trains adapters — least VRAM", tags: ["qlora"] },
  { t: "qlora_bnb = NF4 (any CUDA GPU); qlora_fp8 / qlora_fp4 = surogate-native", tags: ["qlora"] },
  { t: "FP8 QLoRA delayed scaling uses a 1024-step amax history by default", tags: ["qlora"] },
  { t: "bf16 LoRA dtype = higher adapter accuracy but slower than fp32 path", tags: ["qlora", "lora"] },
  { t: "pre-quantized models (FP8/NVFP4/MXFP4) skip the quantize step at load", tags: ["qlora"] },
  // ---- lora ----
  { t: "LoRA rank 16 + alpha 32 is a solid default; raise rank for harder tasks", tags: ["lora"] },
  { t: "target all of q,k,v,o,gate,up,down for best LoRA coverage", tags: ["lora"] },
  { t: "merge_adapter: true bakes LoRA into the base model after training", tags: ["lora"] },
  { t: "grad-norm tiny + loss flat? you have LR headroom — try a higher LR", tags: ["lora", "plateau"] },
  { t: "lora_dropout > 0 can help generalization on small datasets", tags: ["lora"] },
  // ---- memory / offloading ----
  { t: "OOM? first levers: lower batch, shorter seq, enable recompute", tags: ["memtight"] },
  { t: "recompute + offload_residual → activation memory O(1), not O(layers)", tags: ["memtight"] },
  { t: "offload_optimizer moves Adam states to CPU (double-buffered over PCIe)", tags: ["memtight"] },
  { t: "offload_master keeps the FP32 master on CPU — frees GPU at update time", tags: ["memtight"] },
  { t: "offloading trades VRAM for PCIe time — use write-combined to soften it", tags: ["memtight"] },
  { t: "lots of free VRAM? raise per_device_train_batch_size for throughput", tags: ["memroom", "throughput"] },
  { t: "free VRAM but want stability? raise grad-accum instead of batch", tags: ["memroom"] },
  { t: "adamw_8bit already cuts optimizer memory ~4× vs fp32 Adam", tags: ["memtight"] },
  // ---- optimizers ----
  { t: "adamw_8bit is the default — 8-bit blockwise states, near-fp32 quality", tags: [] },
  { t: "NorMuon orthogonalizes 2D updates — often better, ~15% slower", tags: [] },
  { t: "NorMuon cautious weight decay (default) only decays agreeing updates", tags: [] },
  { t: "max_grad_norm clips gradients; 1.0 is a safe default, 0 disables", tags: ["diverge"] },
  { t: "warmup_ratio 0.1–0.15 stabilizes the first steps of LoRA SFT", tags: [] },
  { t: "wsd scheduler = warmup-stable-decay; good for known step budgets", tags: [] },
  // ---- moe ----
  { t: "MoE LoRA tunes experts only by default — add train_router to unfreeze the gate", tags: ["moe"] },
  { t: "watch expert_utilization — below 0.5 after warmup = dataset too narrow", tags: ["moe", "plateau"] },
  { t: "raise router_aux_loss_coef (~0.001–0.01) if routing collapses", tags: ["moe"] },
  { t: "ep_size > 1 shards MoE experts across GPUs (expert parallelism)", tags: ["moe", "multigpu"] },
  { t: "omit router_*_loss_coef to keep the model's tuned defaults", tags: ["moe"] },
  { t: "MoE needs varied data to keep all experts active — avoid narrow corpora", tags: ["moe"] },
  // ---- grpo / rl ----
  { t: "GRPO split-GPU: trainer + vLLM on disjoint GPUs (must not overlap)", tags: ["grpo"] },
  { t: "co-locate mode shares GPUs via CUDA IPC when you can't split them", tags: ["grpo"] },
  { t: "vLLM hot-reloads LoRA from broadcasts/step_N once the STABLE marker lands", tags: ["grpo"] },
  { t: "raise max_async_level to 2 when weight broadcast latency is high (network)", tags: ["grpo"] },
  { t: "GRPO reward stuck? check the environment + that rollouts vary", tags: ["grpo", "plateau"] },
  { t: "RULER judge runs a 2nd vLLM — give it its own --judge-gpus", tags: ["grpo"] },
  // ---- long context ----
  { t: "seq ≥ 8K? enable long_context (tiled MLP) — trades ~5–10% speed for memory", tags: ["longseq"] },
  { t: "long_context disables CUDA graphs (varying seq len) — works with packing", tags: ["longseq"] },
  { t: "at long seq, MLP activations dominate VRAM — tiling is the fix", tags: ["longseq"] },
  // ---- multi-GPU ----
  { t: "surogate always shards optimizer states (ZeRO-1) — strictly better than DDP", tags: ["multigpu"] },
  { t: "zero_level 2 shards grads, 3 shards weights — more saving, more comm", tags: ["multigpu"] },
  { t: "ZeRO-3 streams weights; FP4 weight caching is disabled in that mode", tags: ["multigpu", "nvfp4"] },
  { t: "one process per GPU avoids the GIL and scales across nodes", tags: ["multigpu"] },
  // ---- throughput / perf ----
  { t: "fused cross-entropy avoids materializing the per-token loss tensor", tags: ["throughput"] },
  { t: "use_fused_rope computes cos/sin on the fly — saves memory", tags: ["throughput"] },
  { t: "lmhead_chunks splits the LM head to shrink the logit buffer", tags: ["memtight"] },
  { t: "attn_bwd_chunks splits attention backward to cut workspace memory", tags: ["memtight"] },
  { t: "sample_packing packs sequences — fewer pad tokens, more real tokens/s", tags: ["throughput"] },
  { t: "SOL% (speed-of-light) shows how close you are to peak FLOPs", tags: ["throughput"] },
  // ---- device health ----
  { t: "GPU ≥ 80°? check airflow/power — thermal throttling cuts throughput", tags: ["hot"] },
  { t: "SM utilization low mid-step? dataloader or PCIe may be the bottleneck", tags: ["lowutil"] },
  { t: "raise dataloader_num_workers if GPUs starve between steps", tags: ["lowutil"] },
  // ---- monitoring / general ----
  { t: "loss plateau? try a lower LR, more data, or check eval vs train gap", tags: ["plateau"] },
  { t: "loss rising? lower LR, clip grads (max_grad_norm), or check data", tags: ["diverge"] },
  { t: "eval ≫ train loss = overfitting — add data or dropout", tags: [] },
  { t: "report_to: [surogate] is what feeds this dashboard", tags: [] },
  { t: "log_gpu_util: N writes device telemetry every N steps", tags: [] },
  { t: "logging_steps: 1 gives the smoothest live curve", tags: [] },
  { t: "each run here gets its own feed — launch many, watch any in Runs", tags: [] },
  { t: "save_total_limit keeps only the last N checkpoints", tags: [] },
  { t: "eval_steps controls how often the eval ◆ markers appear", tags: [] },
  { t: "max_steps: -1 derives steps from epochs × dataset size", tags: [] },
  // ---- getting started / dashboard (friendly) ----
  { t: "new here? bf16 + LoRA on one GPU is the can't-go-wrong default", tags: [] },
  { t: "in Runs, press c on a past run to overlay its loss curve on this one", tags: [] },
  { t: "every launch is its own folder — config, logs and checkpoints together", tags: [] },
  { t: "finished a run? Runs shows its final loss, duration and checkpoints", tags: [] },
  { t: "browse Models/Datasets to search HuggingFace, then head to Launch", tags: [] },
  { t: "a green ● in Runs means live; ✓ means it reached max_steps", tags: [] },
];
