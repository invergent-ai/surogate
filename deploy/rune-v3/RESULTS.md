# Rune v3 deployment measurements — 2026-09-27

The subsequent [prefill reducer optimization](PREFILL_PERFORMANCE.md) adds a measured
8.6% text throughput gain at eight clients while preserving the tested answer payloads.
The measurements below describe the preceding image-scheduling/deployment baseline.

One RTX PRO 6000 Blackwell, the same local NVFP4-experts/BF16-rest artifact from the handover,
vision enabled, 64 scheduler lanes, 16,384-token context, automatic KV, 8,192-token prefill
window, speculation off. No checkpoint or artifact was published or overwritten.

## Image scheduling fix

The old executor admits only one request with transient image state. Its one shared allocation
belongs to that request until prefill completes. Image text blocks are also excluded from
packed prefill. The replacement suballocates independent regions inside a bounded startup
pool, rotates encoding slices between staged images, and packs ready image text chunks when
no decoders need latency protection. Partially executed layer slices cannot switch paths.
Automatic memory planning includes the entire pool; the pool needs no request-time CUDA allocations.

At **280 image tokens**, matched eight-client throughput increased from **16.65 to 24.01
requests/s (+44.2%)**, and median latency fell from **480.2 to 327.3 ms**. The new server
reaches **27.90 requests/s at sixteen clients**. The baseline used 45-second levels; the new
budget sweep used 20-second levels. These are single sweeps on the same request set, not
confidence intervals for speed. The valid baseline profile and its serving log are in
`/home/flavius/work/agent/results/public-image-baseline/`.

## Image budget decision

Each accuracy run answered all 1,196 requests successfully, at concurrency eight. Accuracy
below is categorical accuracy on that fixed set. The intervals are the existing benchmark's
paired bootstrap intervals for the difference against the new 280-token run.

| Soft tokens/image | Accuracy | Gain vs 280 (95% interval) | Single-client median | Peak requests/s | Four-client requests/s |
|---:|---:|---:|---:|---:|---:|
| 280 | 77.76% | — | 99.0 ms | 27.90 | 19.70 |
| 560 | 81.44% | +3.68 pp [1.84, 5.60] | 138.1 ms | 15.11 | 12.50 |
| **1120** | **83.36%** | **+5.60 pp [3.51, 7.61]** | **236.2 ms** | **6.58** | **6.14** |

**Deploy 1120** because image accuracy is the priority established in the handover. Limit
image decisions to four in flight: four clients retain 93.3% of observed peak throughput,
with a 644 ms median versus 1,233 ms at eight clients. Higher-resolution vision still costs
more GPU work; the image scheduling fix does not remove that cost. 560 is an available
operator choice if that accuracy/latency preference changes.

The scheduling change at 280 scored 77.76% versus the old NVFP4 server's 78.09%: -0.33 pp,
paired interval [-1.59, +1.09], and 93.23% categorical agreement. Old/new probability payloads
are **not bitwise equal** across the different GEMM batching shapes. Repeated fixed
single-client execution at 1120 was bitwise equal on all 24 tested image cases across three
passes. No nondeterministic fused expert reduction was enabled.

Full speed/answer files: `/home/flavius/work/agent/results/public-image-v1-{280,560,1120}/`.
The runtime option is `--gemma-image-tokens 280|560|1120`; the original artifact remains intact.

A separate 100-second, eight-client run at 1120 completed 652 requests without errors
(6.47 requests/s, 1,235 ms median). Its valid 20-second nsys window contains 19.39 seconds
of GPU kernel time: vision attention 36.49%, BF16 GEMMs 27.25%, normalization 14.44%,
text attention 7.73%, grouped NVFP4 GEMMs 5.18%, and other kernels 8.92%. These are kernel
time shares, not wall-clock latency shares. The remaining high-resolution image cost is
primarily the vision/dense path, rather than the old single-transient admission gate.
The profile waited for a real inference answer before driving load; records are in
`/home/flavius/work/agent/results/public-image-1120-profile/`.

## Launch validation

- Native HTTP admission, option parsing, transient allocator and GPU request-memory tests pass.
  The pool test writes separate device regions, releases/reuses one, and checks its neighbour
  remains intact; a host test exercises 10,000 fragmented allocation/release steps.
- CLI suite: 84 passed, 9 skipped. Architecture serving contract suite: 25 passed, 24 skipped
  (optional model config fixtures). The native
  family frontend suite is also skipped because this box lacks its Qwen tokenizer fixture.
  Actual Rune image preparation and token-budget changes were exercised by the GPU sweep.
- 64 simultaneous thinking requests: **64 HTTP 200**, 23,025 reasoning tokens, 18.38 seconds
  total. The engine remained healthy. At mixed prefill batch 44, the graph budget correctly
  declined capture and ran eagerly, covering the former 42–43-lane capture-crash case.
- Eight concurrent two-image requests: **8 HTTP 200**. Text throughput after stress was
  23.44 requests/s at eight clients and 24.33 at 32; single-client median was 42.9 ms.
- Records: `/home/flavius/work/agent/results/public-validation-v1/` and
  `/home/flavius/work/agent/logs/public-gpu-tests.log`.
- The actual deployment container, read-only mounts and local TLS proxy passed authenticated
  text/image/thinking requests (HTTP 200) and rejected missing credentials (401). Five private
  routes returned 404. With the original 4/s, burst-4 proxy policy, an 80-request malformed-body
  burst produced 63 native and 75 edge
  rate-limit responses; every 429 carried `Retry-After`. Admitted malformed bodies returned
  400, as expected. Three simultaneous thinking requests produced two successes and one
  `thinking_limit_exceeded`; eight images produced four successes and four
  `image_limit_exceeded`. Health remained 200 after the burst. The native listener was
  127.0.0.1:8460 and TLS was 127.0.0.1:8443, with certificate verification enabled.
  Records: `/home/flavius/work/deployment/rune-v3/staging/verification.json`.
  The staging services were stopped after this verification. Subsequently the engine was
  installed and enabled as `rune-v3.service`, still on loopback. Public deployment still
  needs the chosen hostname, trusted TLS and proxy activation.
- Per user request, the proxy default is now **1 request/s per IP, with no extra burst**.
  `--per-ip-rps` and `--per-ip-burst` configure it when rendering. Real nginx checks with a
  local HTTP stub confirmed: default admits one immediate request, rejects the next with
  429 plus `Retry-After`, and admits after refill. A 2/s, burst-1 override admits two
  immediate requests and then rejects excess. Both configs pass `nginx -t`; invalid
  argument values are rejected. No GPU was needed for these proxy-specific checks.
  Records: `/home/flavius/work/deployment/rune-v3/staging/rate-policy-verification.json`.
- The installed systemd service is active and enabled at boot, with zero restarts during
  verification. Authenticated real inference and model discovery pass; missing credentials
  return 401 and health returns 200. Only 127.0.0.1:8460 is listening for this service.
  Records: `/home/flavius/work/deployment/rune-v3/staging/systemd-verification.json`.

## Remaining performance work

Source tracing confirms that ordinary BF16 prompt attention is tiled through the small-T
family (`launch_cached_prompt_tiles` / `gqa_attention_packed_prompts`), not just the short
question branches. The fixed 128-key partitions and FP32 partials were introduced to make
cache scoring stable across prefill and batching. The batch-aware split clamp deliberately
applies only to the int8 KV path. Simply applying that clamp to BF16 would not be a validated
prefill optimization. No attention numerical policy or BF16 dense/vision weight precision was
changed for this deployment.

A single-client thinking run with CUDA graph **node** tracing completed 31 requests with
zero errors in 102.3 seconds (4.317-second median); engine decode rates remained around
120 tokens/s. The 20-second profile collected 19.22 seconds of kernel time: BF16 dense
GEMV 30.59%, normalization 21.73%, sparse MoE decode 17.73%, attention 12.36%, the output
head's W8 kernel 6.28%, other BF16 GEMM 4.55%, rotary embeddings 3.99%, and other work 2.77%.
The workload includes request prefills, though 512-token thinking makes it decode dominated.
The largest remaining costs are dense vector projections and normalization. The one-token
expert path already uses the native NVFP4 decode kernels; reducing the wide TRT-LLM runner's
permutation overhead is therefore not the main remedy for this trace. Future changes should
target these measured kernels and recheck numerical reproducibility and accuracy.

Records: `/home/flavius/work/agent/results/public-decode-nodes-profile/`. The earlier
`public-decode-profile/` run validates speed but its default graph-level trace omits graph
internals and must **not** be used for kernel percentages. Use `--cuda-graph-trace=node`
when profiling decode; startup readiness must still require a real inference response.

Speculation remains off to preserve ordinary decision throughput. The handover's separate
DFlash artifact remains local and available for a future latency-oriented serving tier.

Engine changes are committed on `feat/rune-public-serving`: `75dcf3d4` (HTTP admission) and
`a72f1b59` (image scheduling, image token budget, image admission cap).
