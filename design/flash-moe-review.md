# study/flash-moe — reviewed 2026-08-30

**What it is.** A pure C/Objective-C + Metal inference engine that runs
Qwen3.5-397B-A17B on a 48 GB Apple laptop by streaming 209 GB of 4-bit experts off
an NVMe SSD, one layer at a time. 60 layers (45 GatedDeltaNet + 15 full attention),
512 experts per layer, top-4. Best result **4.36 tok/s**, single user.

**Verdict: no code transfers, but the experiment log is worth more than the code.**
Metal shaders are useless to us, the storage tier is wrong (SSD rather than pinned
host RAM), and the headline architectural finding — that on Apple Silicon SSD DMA
and GPU compute share a memory controller and *cannot* be profitably overlapped —
is the opposite of our situation, where gather/compute overlap is measured and
real. What it does have is 58 logged experiments, mostly negative, three of which
land on levers we either hold or were considering.

## The three that matter to us

**1. MTP / speculative decoding on an offloaded MoE is break-even.** This answers
the standing question about Flash-Next directly, and with data rather than
argument. Their reasoning, which is tier-independent: each speculated token needs
its *own* expert routing and therefore its own expert fetch, so batched
verification of k tokens costs k times the I/O. That is exactly unlike a dense
model, where verification of k tokens reads the same weights once and the cost is
constant in k. Any offloaded MoE inherits this: the speculation amortises compute
but not the transfer that dominates. Our Flash-Next single-card row is 28.7 tok/s
with the experts on the host, so the transfer is the budget — expect
speculation to buy nothing there, and to keep buying on the VRAM-resident models
where it already does.

**2. Expert routing prediction does not pay, in three independent forms.**
Temporal prediction reached a 25 % hit rate and cost 18 %; an MLP routing
predictor reached 31 % accuracy; speculative early routing off the pre-attention
state reached 53 % accuracy and still cost 38 %, because the mispredictions
polluted the cache. **This does not generalise to our expert slot cache and it is
worth being clear why**: we do not predict, we retain. A 3,000-slot residency
cache over the working set measures a **97.9 % hit rate at 16 users** (2026-08-30)
because the same experts recur, not because anything guessed them. Their result is
evidence against building a predictor on top of our cache, which is the form the
idea would take if it came back.

**3. Expert compression does not pay.** LZ4 took the expert store 209 → 175 GB and
cost **13 %**: decompression exceeded what the smaller warm-cache reads saved. A
second variant (compressing into a GPU-private buffer) cost 20 %. This is directly
relevant to the four x8 PCIe links on this host, because compressing the host
expert bank is the tempting way to buy back a halved link. Their arithmetic says
the decompressor becomes the bottleneck before the link does. Different tier, same
shape of trade — worth a measurement before anyone spends a week on it, not an
assumption either way.

## What does not transfer

- **"Trust the OS page cache" (+38 %, their most foundational win).** They deleted
  their own LRU and let the page cache manage 35 GB of expert data. Our experts
  live in *pinned, device-mapped* host memory precisely so the GPU can DMA them
  without a page fault; pinned pages are not pageable and there is no OS cache to
  trust. The lesson does not port.
- **The unified-memory overlap constraint.** Apple-specific, and inverted for us.
- **FMA-rearranged dequant matvec (+12 %)** and the 4-bit/2-bit Metal kernels.
  Ours are MMA-based on tensor cores; the arithmetic restructure is not applicable.
- **2-bit experts.** They ship it as faster but note it breaks tool calling. We
  have a standing rule against quantisation that trades accuracy in defaults.

## What to do with it

Nothing to build. Two of their negatives (speculation on an offloaded MoE, expert
prediction) should be recorded as evidence so we do not spend the week they spent,
and the compression one should be filed next to the x8 PCIe item as a caution
rather than a conclusion.
