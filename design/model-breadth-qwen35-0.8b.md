# Model breadth: first new engine target — Qwen3.5-0.8B

**Status:** mechanism fully mapped (2026-08-24); execution is mechanical from here.
**Goal:** `surogate serve models/Qwen3.5-0.8B-Q8_0.gguf` end-to-end. At BF16/W8 a
0.8B is ~0.9–1.7 GB resident — the first full E2E serve fits the small idle-GPU
windows this host actually gets (no 18 GB wait).

## What defines an exact target (measured on qwen3_6_27b: ~1,693 lines)

| Piece | File(s) | Content |
|---|---|---|
| Geometry | `impl/config.h` (93 ln) | ALL `static constexpr`: hidden/layers/intermediate/heads/head_dim/GDN dims/rope/eps |
| Weight profile | `impl/variant.{h,cpp}` (639 ln) | `WeightsProfile` enum per (model_id, weights_id); per-tensor format selection |
| Load bindings | `impl/load/bindings.{h,cpp}` (818 ln) | artifact tensor name → ModelView slot; shapes derive from `TextConfig` |
| Registration | `impl/package.cpp` (143 ln) | identity strings, sampling defaults, `resolve_weights` |
| Registry branch | `src/targets/registry.cpp:183-196` | one `if (identity.model_id == ...)` → `construct_registered<Package,...>` |
| CMake | `src/targets/<key>/CMakeLists.txt` + `src/CMakeLists.txt:309-311` | sources join `ninfer_engine` |

The family runtime (`src/targets/qwen3_6/impl/`, 28+13 files) is templated on the
Variant — **no family code changes** for a new same-family geometry.

## The 0.8B geometry (read from GGUF KV; cross-check official config.json)

Identical to 27B (family invariants — this is why the port is cheap):
head_dim (key/value_length) **256** (the `kGqaHeadDim` hardcode FITS), rotary 64,
rope_theta 1e7, full_attention_interval 4, GDN conv 4, GDN key 16×128,
rms_eps 1e-6, vocab/output_rows 248320 (same tokenizer), nextn(mtp) 1.

Differs (goes into `TextConfig`): hidden **1024**, intermediate **3584**,
main layers **24**, q/kv heads **8/2**, GDN value heads **16**×128 (symmetric —
k==v, so llama.cpp's V-reorder does NOT fire; simpler than 27B's 16/48).

## The one real obstacle: exact-shape dispatch whitelists

Kernels are `template<Geometry<N,K>>`, compile-time instantiated; launchers hold
explicit instantiation lists; dispatchers whitelist exact (N,K) and THROW
otherwise (e.g. `bf16_dispatch.cpp:12` admits only (14336,5120)+(5120,6144)).
BUT schedules are geometry-derived templates (`Bf16LinearDecodeSchedule<G>`),
so any aligned shape instantiates a sound schedule. **Breadth mechanism =
add per-target (N,K) instantiations + whitelist entries, per format family.**
(A runtime-dim generic Tier-0 kernel remains the long-term answer for
arbitrary models — plan §8.2; not needed for this target.)

Decision: reuse the **groupwise-int profile** (same per-tensor format mix as the
27B recipe, scaled) rather than inventing a pure-BF16 profile — bf16 linear_swiglu
does not exist as a kernel family, the quantized path is the real product path,
and variant/bindings/converter all transfer mechanically.

## 0.8B shape set to instantiate (derive exactly from the 27B recipe + TextConfig)

With hidden=1024, inter=3584, q=8·256=2048, kv=2·256=512, gdn qkv=3·2048=6144, z=2048:
- attn input (fused qkv+gates per family op): rows from bindings math × K=1024
- attn out (linear_add): N=1024, K=2048
- FFN gate_up (linear_swiglu): N=2·3584=7168, K=1024; FFN down (linear_add): N=1024, K=3584
- GDN input proj: N=6144(+z 2048 fused per op), K=1024; GDN out: N=1024, K=2048
- GDN ba: N=32, K=1024 (a & b, 16 heads each); gating proj bf16
- LM head: N=248320, K=1024 (also MTP shared head)
- MTP: fc N=1024, K=2048; mtp layer = one full-attention layer at the same shapes
Consult `tools/convert/qwen3_6_27b/{inventory,recipe}.py` for the authoritative
per-tensor format → kernel-family mapping, then instantiate that family's
launcher + whitelist for each 0.8B (N,K,format) cell. Expect ~8–10 distinct
shapes × ~4–6 families ≈ 30–60 mechanical edits (record as PATCHES #13).

## Execution order (each step compiles/tests standalone)

1. `src/targets/qwen3_5_0_8b/` — copy 27b structure; swap `TextConfig` constants;
   `model_id "qwen3.5-0.8b"`, `weights_id "groupwise-int"`, sampling defaults from
   `generation_config.json`; NO vision, NO dflash (strip those config blocks).
2. Registry branch + CMake; build → collect missing-instantiation/link errors as
   the definitive shape worklist (the compiler enumerates it for us).
3. Per-family instantiation + whitelist edits until link is clean (PATCHES #13).
4. Converter `tools/convert/qwen3_5_0_8b/` — copy 27B inventory/recipe, swap
   shapes/counts; drop vision/dflash sections; draft-head ranking optional
   (`--no-mtp-shortlist` if the 27B ranking file is model-specific).
5. Ingest: `converter_for_config` (hidden 1024, layers 24, model_type qwen3_5)
   + `gguf_target_key` (arch qwen35, hidden 1024) → `qwen3_5_0_8b`.
   Add `surogate/serve/resources/qwen3_5_0_8b/` (config/generation_config from
   Qwen/Qwen3.5-0.8B).
6. E2E: GGUF → convert (CPU or small-VRAM) → `surogate serve` on an idle-window
   GPU (~2 GB) → `/v1/chat/completions` coherent reply = the first full-chain
   validation of the entire serve engine on this machine.

## Risks

- Bindings/variant may hardcode 27B-specific tensor counts beyond TextConfig
  (e.g. layer-role tables) — the compiler/loader will surface these; fix in the
  new target's copies, never in family code.
- The MTP draft-head shortlist ranking file is 27B-specific; ship the 0.8B
  target with spec disabled first (engine default) and add the ranking later.
- Sampling defaults for 0.8B differ from 27B — take from its generation_config.
