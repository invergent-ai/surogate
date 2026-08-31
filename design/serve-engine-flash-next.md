# Qwen3.8-Flash-Next: CPU offload, pipeline parallelism, expert parallelism

Owner directive (2026-08-28): serve
`models/Qwen3.8-Flash-Next-UD-Q4_K_XL-0000{1..4}-of-00004.gguf` on **one GPU with
CPU offloading** and on **multiple GPUs for throughput**, with offloading
"super-optimized like FreeToken and llama.cpp" and PP/EP "like vLLM". This
supersedes the plan's §8b verdict that refused EP: EP is designed here for the
no-P2P PCIe box we have, and measured against PP before it is kept.

Companion notes: `design/serve-engine-offload.md` (FreeToken read),
`design/serve-engine-backlog.md` (parked 35B kernel work).

## 0. What the target actually is

This is **not a model the engine knows**. GGUF arch `qwen4exp`, HF
`Qwen/Qwen3.8-Flash-Next` (`model_type qwen4_exp`), 111.3 GB across 1,224 tensors:

| component | shape / count | bytes | format |
|---|---|---:|---|
| routed experts, gate+up | 48 layers × 512 experts × (2560→640)×2 | 45.5 GB | Q4_K / Q5_K |
| routed experts, down | 48 × 512 × (640→2560) | 31.5 GB | Q5_1 / Q8_0 |
| n-gram PLE table | 320,001,536 rows × 160 | 28.8 GB | IQ4_NL |
| everything else (attention, GDN, hyper-connections, shared experts, router, embeddings, head) | | ~5.5 GB | Q8_0 / F32 / BF16 |

48 layers, hidden 2560, **4 hyper-connection residual streams** (the residual is
10,240 wide), 36 gated-delta-net layers + 12 full-attention layers (every 4th),
**512 experts top-10** (FFN 640) plus a sigmoid-gated shared expert, a
**QSA indexer** on the attention layers (4 heads × 128, budget 2048, block 4),
one **PLE layer** (layer index 1 in GGUF, `ple_layer_ids=[2]` in HF) that hashes
each token's bigram and trigram into 16 heads of a 20M-row-per-head table, and
interleaved mRoPE on 64 of 256 head dims. Vocabulary is the Qwen3.5 tokenizer
(248,320; eos 248044). No output norm — the final hyper-connection mix is it.
The MTP head is not in the GGUF (llama.cpp's converter drops it).

The exact forward pass is recorded in §5; the reference is llama.cpp upstream
master (`study/llama.cpp-master/src/models/qwen4exp.cpp`, added 2026-08-2x —
neither the vendored copy nor `tools/llama.cpp` knows the arch) with
`transformers` main `modeling_qwen4_exp.py` as the second source.

## 1. The measured envelope (this box, 2026-08-28)

| path | measured |
|---|---|
| pinned host → GPU `cudaMemcpyAsync`, PCIe 5.0 x16 | **56.8 GB/s** |
| FreeToken-style zero-copy gather kernel reading pinned host rows (3 MiB rows) | **52 GB/s** (92 % of memcpy) |
| host DRAM read, one NUMA node, 8-64 threads | **160 GB/s** per socket (2× EPYC 9124, 503 GB) |
| GPU↔GPU `cudaMemcpyPeerAsync`, P2P **unsupported** (`canAccessPeer=0`), driver host-staged | **43 GB/s**, same for NUMA-local (0→1) and cross-socket (0→4) |
| cross-socket pinned H2D | no penalty (56.8 GB/s) |

Topology: GPUs 0-3 on NUMA0, 4-7 on NUMA1, no NVLink, P2P off on GeForce. Host
memory bandwidth is **3× PCIe**, which is the regime where FreeToken's rule
puts expert compute on the CPU rather than streaming weights to the GPU.

Per decode round, the expert bytes a batch of B tokens touches:

| B | experts touched (of 24,576) | bytes | over PCIe | tok/s if fully streamed |
|---:|---:|---:|---:|---:|
| 1 | 480 | 1.5 GB | 29 ms | 34 |
| 32 | ~11,500 | 36 GB | 0.7 s | 46 |
| 128 | ~22,500 | 70 GB | 1.35 s | 95 |
| 256 | ~24,300 | 76 GB | 1.46 s | 175 |

So a naively streamed single GPU tops out near 100-175 tok/s aggregate and
34 tok/s single-stream — that is the floor, not the design. The design lifts it
with a GPU expert cache (hits are free), CPU expert compute (160-320 GB/s of
host bandwidth against 52 of PCIe), and a bandwidth-matched split between the
two, exactly the FreeToken hybrid on a host 6× stronger than the laptop it was
built for. Eight GPUs hold every expert resident (9.6 GB per GPU at EP8, or per
6-layer PP stage) and turn the same model into a kernel-bound one.

## 2. Decisions

**D0. The core stays model-agnostic and SM-agnostic** (owner directive).
Offload, pipeline parallelism and expert parallelism are built on engine
contracts that any target implements, never on `qwen4exp`:

- `ExpertBank` — L layers × E experts × K banks (one per quantised component,
  each with its own row format). The slot cache, the host gather, the CPU
  compute path, prefill streaming and EP sharding all operate on an
  `ExpertBank` plus a routing-id tensor. A target declares its bank schema
  and calls one `moe(layer, ids, weights, in, out)`; it never sees where the
  bytes live. The 35B, this model, and every future MoE share it.
- `HostTable` — a host-resident row table with a device-side zero-copy gather
  (device indices, device row count). The PLE table is the first user; any
  model with a large sparse embedding (n-gram memories, huge vocabularies) is
  the next.
- `PipelineStage` — the executor runs a *layer range* of any target program
  that exposes `layer_count()`, `run_layers(range, round)`, a residual view
  (whatever its width — 10,240 here, `hidden` elsewhere) and its per-layer
  per-sequence state descriptors. Hops move the residual view; state pools
  are per stage. The target never knows it is split.
- `EpGroup` — the dispatch/combine and the placement planner take an
  `ExpertBank` and a communicator; the LPT planner is already model-free.
- SM: kernels dispatch on the device's compute capability at runtime and are
  instantiated for the whole `SUROGATE_SERVE_CUDA_ARCHS` list; no SM-specific
  constant may be baked into a kernel (the MoE prefill kernel's
  `kRtx5090SmCount = 170` is exactly the kind of thing this rule removes — it
  becomes a device query). Where a format needs hardware a device lacks
  (NVFP4 tensor cores below sm_120a/sm_100a), the artifact profile selects
  another format for that device rather than the kernel pretending.

The per-model surface is the target: its config, its tensor bindings, and its
layer program written in engine ops. That is the same boundary the five
existing targets already respect; this work adds three contracts beneath it
and no new dependency from the core on any target.


**D1. Onboard the architecture first, on one GPU, with the simplest possible
expert streaming.** Nothing can be validated until the model runs, and it
cannot run on one GPU without *some* offload. Phase 1 streams every touched
expert of a layer from pinned host memory with plain `cudaMemcpyAsync` — the
34 tok/s floor — because that is ~200 lines and gives a parity oracle against
llama.cpp within days. Everything "super-optimized" is layered on a model that
already produces the right tokens.

**D2. Contexts ≤ 2,051 tokens run dense attention and are exact; the indexer
is phase 2.** With budget 2048 and block 4, llama.cpp selects
`min(n_kv, 2048 + 4 − 1)` cells — every cell when the context is shorter — so
below that length the sparse path *is* dense attention. The board's shapes
(512/128, 2048/16) never exceed it. Longer contexts are refused at admission
until the indexer lands (its own KV cache of 128-wide keys per token, block
pooling, a per-token top-k, and a masked-attention kernel).

**D3. Single-GPU offload is the FreeToken hybrid, host-side compute first.**
Host bandwidth is 3× PCIe here, so misses are computed on the CPU (AVX-512
Q4_K/Q5_1/Q8_0 GEMV on NUMA-local thread pools, the experts split by socket)
while a bandwidth-matched fraction of them is gathered over PCIe into a GPU
LRU slot cache in parallel. Hits run on the GPU MoE kernels. The PLE table
never moves: 16 rows × 160 dims per token is nothing, so it stays pinned on
the host and is read by a zero-copy gather kernel (device-side indices, so it
captures into the decode graph).

**D4. Multi-GPU is pipeline parallelism, one process, one thread per GPU,
host-staged activation hops.** The numbers decide it: a PP hop moves
B × 10,240 × 2 B (the 4-stream residual) — 5 MB at B=256, 0.12 ms at 43 GB/s,
seven hops per round — while EP8 moves B × 10 × 2,560 × 2 B *per layer per
direction* through 96 all-to-all phases, 1.25 GB per round at B=256, and
every layer is a lockstep barrier across eight cards with no P2P. PP holds
each stage's experts resident (9.6 GB), needs no collective at all, and
pipelines micro-batches so all eight stages stay busy. EP is phase 4, inside a
PP stage, with NCCL's SHM transport, and it stays only if it beats PP-only on
the board. vLLM's EP is built on DeepEP over NVLink/RDMA; this box has
neither, and pretending otherwise would just be slower.

**D5. Reuse from the engine and the trainer.** From the engine: the GDN
mixer (this model's GDN is the Qwen3.5/3.6 one — 16 k-heads, 48 v-heads,
head 128, conv 4 — with a sigmoid instead of silu output gate), gated
attention with interleaved q|gate rows, the sparse-MoE family
(re-instantiated at 512 experts / 640 / 2560 / top-10), W8 row-split linears
at k=2560, the paged KV cache, the executor and its mixed rounds. From the
trainer (per the survey): `runtime/ep/ring_arena.h` (pure CUDA-runtime slab
ring), `runtime/ep/lpt_planner.{h,cpp}` (pure host LPT expert placement),
`nvidia::nccl` as an imported CMake target for phase 4, and three rules —
pinned staging for every small async copy (the pageable-copy convoy deadlock
documented at `ep_strategy.h:212`), read-then-`cudaHostRegister` for big host
banks (3× faster than `cudaHostAlloc`), and per-expert pointer arrays rather
than merged tensors. Not reused: the trainer's thread-per-GPU NCCL
communicator (it assumes a shared address space), its dispatch pipeline
(written against the DSL executor), and its resolution of the graphs-versus-
offload conflict, which is to turn graphs off.

**D6. What llama.cpp contributes** (per the survey): the batch-size-gated
`offload_op` (compute on CPU below N tokens, stream weights to the GPU above —
both paths from one predicate, default 32); selective expert streaming for
prefill keyed on the routing ids (bitset the touched experts, coalesce
consecutive ids into single memcpys, cache the id read across gate/up/down);
`-ot`-style regex placement overrides so the ecosystem's tuning idioms work
(`--cpu-moe`, `--n-cpu-moe N`); and the warning that a rotating input-copy
ring invalidates captured graphs (pin copy 0 for decode).

**D7. Experts stay 8-bit in the artifact; native K-quant kernels are the
bandwidth lever, not requantisation.** Measured on layers 0 and 3 (16 experts
each, relative L2 against gguf-py's dequantisation of the shipped blocks):

| source → engine format | Q4G64 | Q5G64 | Q6G64 | **W8G32** |
|---|---:|---:|---:|---:|
| gate/up `Q4_K` | 11.7 % | 5.5 % | 2.6 % | **0.57 %** |
| down `Q5_1` | — | 5.2 % | 2.5 % | **0.55 %** |
| down `Q8_0` | — | — | — | **exact** (bridge) |

The symmetric group-64 formats the sparse-MoE kernels take would be a second
quantisation of every expert (the K-quants carry per-32 scale *and* min, which
group-64 scale-only cannot represent) — the same class of silent quality loss
the 4B conversion rule forbids. W8G32 is one byte per weight and doubles the
expert bytes (62 → 125 GB host-resident, 15.7 GB per PP stage), so every
bandwidth figure in §1 and §4 is for the 8-bit banks until the engine's
`ExpertBank` speaks `Q4_K`/`Q5_1` natively (llama.cpp's formats, its
dequant kernels as the reference) — a Phase 2 item that halves the stream
without touching the weights. The PLE table repacks IQ4_NL → W8G32 exactly
(51 GB pinned host); the dense Q8_0 tensors repack exactly through the
existing GGUF bridge; hyper-connection and PLE projections dequantise to BF16.


### Phase 1 — the model runs (single GPU, streamed experts, exact ≤ 2,051 ctx)

1. **Converter** `surogate/serve/tools/convert/qwen4exp/`: GGUF (4 shards,
   lazy) → `.sinfer`. Objects per §5.6. Experts Q4_K/Q5_K/Q5_1/Q8_0 → the
   engine's Q4G64/Q5G64/Q6G64/W8G32 row-split formats through the existing
   GGUF bridge; hyper-connection and PLE projection weights dequantised to
   BF16 (1.3 GB total; they run on the cuBLAS BF16 path); the PLE table kept
   IQ4_NL as one host-resident object with its 16 head offsets/moduli and 3
   multipliers in the artifact metadata; norm gammas already carry the `+1`
   from the GGUF (do not add again); `ssm_a` already holds `−exp(A_log)`.
   Verification: decode every object class back and diff against the GGUF
   (the 4B lesson).
2. **Target** `csrc/src/serve/targets/qwen4exp/`: a new family (the 4-stream
   residual changes the layer loop, so it is not a `qwen3_6` variant), with
   the runtime program built from the engine's ops. New ops: hyper-connection
   mix/combine (per-stream RMSNorm over [2560,4], three BF16 GEMMs, sigmoid
   gates, stream mean, `2·sigmoid` scatter); PLE (host hash → device index
   tensor, zero-copy IQ4_NL row gather, key/value projections, per-stream
   norms, signed-sqrt sigmoid gate, dilated depthwise conv with a 9-column
   recurrent state per sequence); sigmoid-gated GDN output; the 512-expert
   sparse-MoE instantiation; the shared-expert scalar sigmoid gate; the final
   HC mix as output norm. Contract: `is_recr = (il+1) % 4 != 0`.
3. **Expert streaming v0**: routed experts in pinned host memory (read the
   GGUF into pageable memory, then `cudaHostRegister`); per layer, read the
   routing ids, gather the touched experts' rows with coalesced
   `cudaMemcpyAsync` into a per-layer staging plane, run the resident MoE
   kernels. Not captured (byte counts are host-side); the decode graph is
   simply off in phase 1.
4. **Parity**: greedy tokens against `study/llama.cpp-master` on the same
   prompts; the per-object numerical diff from the converter; llama.cpp's
   8-GPU and 1-GPU numbers are the baseline row.

Exit: coherent output at parity, a first single-stream tok/s, ≤ 2,051 ctx.

### Phase 2 — super-optimized offload (single GPU)

1. **GPU expert slot cache**: a flat pool of expert slots keyed
   `layer × 512 + expert`, LRU by timestamp with active-slot protection
   (reconstructed from FreeToken's Triton mirror), sized MoE-first against a
   KV floor; hits run on the resident MoE kernels in place.
2. **Device-side host gather**: FreeToken's `fast_index_copy_multi` pattern —
   one launch copies all banks of the missing experts from pinned host
   pointers, row count read from a device word — so the whole miss path
   captures into the decode graph. Ported with the L1-no-allocate / write-
   through hints the multi-bank variant lacks.
3. **CPU expert compute**: AVX-512 GEMV for Q4_K/Q5_1/Q8_0 experts (llama.cpp's
   `mul_mat_id` structure — per-expert row lists, cache-line-padded atomic
   chunk counters, one pinned thread per physical core, one pool per NUMA node
   over that node's half of the expert bank), GPU↔CPU handshake with
   `cuStreamWriteValue64`/`cuStreamWaitValue64` (never a spin kernel — it
   clamps CPU clocks through the power governor), `cudaLaunchHostFunc` as the
   fallback.
4. **Bandwidth-matched split**: fetch `pcie/(pcie+cpu)` of each round's
   misses over PCIe, compute the rest on the CPU, in parallel; the fraction
   computed in-kernel in Q16 fixed point from a measured profile
   (`ft bench bw` equivalent), because the miss count only exists on the
   device under capture.
5. **Prefill**: selective streaming of used experts (llama.cpp), whole-layer
   double buffering on a side stream (FreeToken), batched prefill (#88 —
   pays on MoE targets).
6. Placement overrides (`--cpu-moe`, `--n-cpu-moe`, `-ot` regex) and a
   memory planner that solves experts and KV against one budget.

Exit: the single-GPU board row, measured at 1, 16 and 100 users; target
≥ 3× llama.cpp's `--cpu-moe` on the same shapes.

### Phase 3 — pipeline parallelism (8 GPUs)

One process, one executor thread per stage, contiguous layer ranges (6 per
stage; the cut placed on the residual tensor, explicitly), each stage owning
its layers' experts resident, its full-attention KV pages and GDN/PLE states,
the PLE table pinned on the host for stage 0. Activation hops by
`cudaMemcpyPeerAsync` (host-staged, 43 GB/s) with events on the stage
streams; a rotating ring of ≥ 4 hop buffers (copy 0 pinned for the captured
decode round). Continuous batching across stages: the scheduler keeps ≥ 8
micro-batches in flight so every stage is busy; requests live in one stage's
lane space at a time — the front stage admits, the last stage samples, tokens
flow back to the front through the host. State per sequence is the reason
lanes are per stage: the GDN state is 786,432 floats per layer per sequence
(3.1 MB fp32, kept bf16 as today's 27B does), which caps a stage at ~256
lanes with its experts and KV.

Exit: the 8-GPU board row against llama.cpp's layer split and against 8
independent single-GPU-offload replicas; the "massive" number.

### Phase 4 — expert parallelism (measured, then kept or dropped)

Inside a PP stage of N GPUs, shard the 512 experts N ways (LPT-planned from
observed routing, re-planned on a sticky interval), all-to-all dispatch and
combine over NCCL on its SHM transport (`sinfer_dist` target linking the
imported `nvidia::nccl`, one communicator per stage), token counts exchanged
through pinned staging. Compared on the board against PP-only at the same GPU
count. The arithmetic in D4 says it loses on this box; it is built anyway
because the directive names it and because it is the shape that wins on
NVLink hardware.

## 4. Throughput model, before any code

Single GPU, phase 2, at 100 users (B ≈ 100 decode columns): the routed
experts a round touches are ~66 GB; the GPU cache holds ~15 GB (~5,000 slots)
and Zipfian routing puts perhaps 40 % of touches on hits; the CPU computes the
rest at DRAM speed — 40 GB read at 320 GB/s aggregate plus ~470 GFLOP of
dequant+dot on 32 Zen4 cores (~1 TFLOP/s) — ≈ 0.5 s, overlapped with a ~17 %
PCIe share. Round ≈ 0.5 s at B=100 → **~200 tok/s aggregate, ~60-100 tok/s
single-stream** (1.5 GB of experts at DRAM speed ≈ 10 ms + compute). llama.cpp's
`--cpu-moe` on the same host is the baseline being measured now.

Eight GPUs, phase 3: each stage is 6 layers ≈ 1/8 of the model, resident;
the 35B (19.6 GB of experts, 8 active × 40 layers) does 1,942 tok/s on one
card, and this model activates 4× the expert bytes per token, so a stage
behaves like a quarter of a 35B → ~500 tok/s per stage → with eight stages
pipelined at high concurrency **~3,000-4,000 tok/s aggregate**, against
llama.cpp's ~35 tok/s single-stream layer split. That is the number the
directive is after, and it is why PP comes before EP.

## 5. The forward pass (implementation contract)

From `qwen4exp.cpp` and `modeling_qwen4_exp.py`; cited lines are llama.cpp.

**5.1 Residual.** `res_hc[2560, 4, T]` starts as 4 copies of the token
embedding. Per layer: `(PLE if this is the PLE layer) → hc_mix(attn) →
mixer → hc_combine → hc_mix(ffn) → MoE+shared → hc_combine`. No layer norms
anywhere; the mixes are the norms.

**5.2 hc_mix** (`:218-264`): `xn = rms_norm per stream(x) * w_norm[10240]`;
`lo = silu((w_down·xn)/4)` (320); `gate = sigmoid(w_up·lo)` (10240);
`mixed = mean over streams(xn * gate)` (2560); `inject = w_inject·xn` (4).
**hc_combine** (`:266-286`): `res_hc[:,c,:] += out * 2·sigmoid(inject[c]/4)`.
Final head: `hc_mix(output_hc_*)` with no inject, then `output.weight`.

**5.3 Attention layers** (`il+1 ≡ 0 mod 4`): `attn_q [2560→12288]` is
q|gate interleaved per head (`[q_h(256) | gate_h(256)] × 24`), `attn_k/v
[2560→512]` (2 kv heads), q/k RMSNorm(256) + interleaved mRoPE on 64 dims
(sections [11,11,10]; for text tokens all position axes are equal, so this is
plain NeoX partial rope), scale 1/16, output `* sigmoid(gate)`, `attn_output
[6144→2560]`. Indexer (`:469-608`, phase 2): `indexer.k_proj [2560→128]`
cached raw per token; blocks of 4 mean-pooled, RMSNorm'd, roped at the block's
first position; `indexer.q_proj [2560→512]` = 4 heads; score =
`sum_heads relu(q·k_block)`; top `min(n_kv, 2051)` cells unmasked; attention
runs dense over the cache with that mask.

**5.4 GDN layers**: `attn_qkv [2560→10240]` = q(2048)|k(2048)|v(6144),
`attn_gate [2560→6144]` = z, `beta = sigmoid(ssm_beta·x)` (48),
`g = softplus(ssm_alpha·x + ssm_dt.bias) * ssm_a` (48, `ssm_a` is
`−exp(A_log)`), depthwise conv 4 over 10240 with a 3-column state, silu,
q/k L2-normalised, gated delta net with state `[128,128,48]` per sequence,
`out = rms_norm(out) * ssm_norm[128]`, **`out *= sigmoid(z)`** (silu in
Qwen3.5/3.6 — the one numerical difference), `ssm_out [6144→2560]`. V heads
are already in tiled order in the GGUF.

**5.5 MoE**: `probs = softmax(ffn_gate_inp·x)` over 512; top-10; weights
renormalised over the 10 (clamp 6.1e-5); `swiglu(gate, up)`, down, weighted
sum; shared expert `swiglu` FFN(640) scaled by `sigmoid(ffn_gate_inp_shexp·x)`
(one scalar per token); sum.

**5.6 PLE** (layer 1 in GGUF; `:982-1043`, `:1093-1199`): per token, with
`ctx[0]` = the token and `ctx[s]` = the token `s` positions back (eos, and
everything older, once an eos or the sequence start is crossed; a token's own
eos does not cut it): for `n ∈ {2,3}`, `mixed = ctx[0]·m[0] ^ ctx[1]·m[1] (^
ctx[2]·m[2])` in wrapping uint64; head `h = (n−2)·8 + g` gets row
`mixed mod vocab[h] + offset[h]`. Gather 16 rows of 160 (IQ4_NL) → 2560;
`key = ple_key·emb` (10240), `value = ple_value·emb` (2560); `s = sum per
stream(norm(key) * norm(res_hc)) / sqrt(2560)`; `gate = sigmoid(sgn(s)·
sqrt(|s|))` per stream; `gated = value ⊗ gate`; `conv = silu(depthwise
conv4, dilation 3, over norm(gated))` with a 9-column history per sequence;
`res_hc += gated + conv`.

**5.7 Per-sequence state**: attention KV `[256,2,ctx]`×2 on 12 layers (fp8 on
the board); indexer keys `[128,ctx]` on 12 layers (phase 2); GDN conv history
`3×10240` and state `128×128×48` on 36 layers; PLE conv history `9×10240` on
one layer; the last two token ids (for the hash).

**5.8 GGUF → artifact objects** (names as in the GGUF; shapes ggml-order):
globals `token_embd {2560,248320} Q8_0`, `output {2560,248320} Q8_0`,
`output_hc_{norm,down,up}`, `per_layer_token_embd {160, 320001536} IQ4_NL`;
every layer `hc_{attn,ffn}_{norm,down,up,inject}`, `ffn_gate_inp {2560,512}
F32`, `ffn_{gate,up}_exps {2560,640,512}`, `ffn_down_exps {640,2560,512}`,
`ffn_gate_inp_shexp {2560}`, `ffn_{gate,up,down}_shexp`; attention layers
`attn_{q,k,v,output}`, `attn_{q,k}_norm`, `indexer.{q,k}_proj`,
`indexer.{q,k}_norm`; GDN layers `attn_qkv`, `attn_gate`, `ssm_conv1d
{4,10240}`, `ssm_dt.bias {48}`, `ssm_a {48}`, `ssm_{alpha,beta} {2560,48}`,
`ssm_norm {128}`, `ssm_out`; the PLE layer `ple_key {2560,10240}`,
`ple_value {2560,2560}`, `ple_norm_{key,query,conv} {10240}`,
`ple_conv1d {4,10240}`.

## 6. Baselines and harness

- `study/llama.cpp-master/build/bin/llama-server` (CUDA, sm_120, built
  2026-08-28): A) 8 GPUs `-sm layer`, B) 1 GPU `-ot exps=CPU -ot
  per_layer_token_embd=CPU`. Results in BENCHMARKS.md as they land.
- Parity: greedy continuations of fixed prompts, token-for-token against A.
- Throughput: the board's loadgen at 1/16/100 users, 512/128 and 2048/16.
- Micro: the PCIe/gather/DRAM numbers above (`scratchpad/bw.cu`), P2P probe
  (`p2p.cu`), `sinfer_sparse_moe_bench` at the new instantiation.
