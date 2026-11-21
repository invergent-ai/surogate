#pragma once

// Inkling-2-Mini's DSpark drafter as the target runs it (ENGINE-INKLING lane e2, rl/engine-inkling-2026-11-21/lanes/e2/
// DESIGN-dspark.md): the load-time derived weights and the two round entries -- append context, propose -- that the
// family's DFlash runtime reaches through the Variant's DSpark leaves. Op semantics: api/ops/dspark.h
// (ref/dspark_ref.py). The drafter's GEMMs run on lane e1's invariant GEMM (inkling_gemm_inv[_group]).
//
// Draft KV (agent-EN 13:5x): text-pool planes 48 / 49 / 50 = draft layers 0 / 1 / 2, FP8 e4m3 unit scale, the
// sequence's own block-table row; layers 0 and 2 read a block-anchored window of 1,024 keys before the anchor.

#include "targets/inkling2/impl/load/bindings.h"

#include "api/ops/dspark.h"
#include <api/family/dflash_geometry.h>
#include "core/arena.h"
#include "core/tensor.h"
#include "core/paged_kv_cache.h"

#include <cuda_runtime.h>

#include <array>
#include <cstddef>
#include <cstdint>

namespace sinfer::targets::inkling2::detail {

/// One drafter linear in FP8: e4m3 codes per row + an fp32 row scale (ops::dspark_quantize_rows_e4m3).
struct DSparkW8 {
    const std::uint8_t* codes = nullptr;
    const float* scales       = nullptr;
    std::int32_t n = 0, k = 0;
};
struct DSparkLayerW8 {
    DSparkW8 q, k, v, o, k_ctx, v_ctx, gate, up, down;
};

/// What the drafter derives at load (ARTIFACT.md 5: the consumer folds).
struct DSparkDerived {
    bool ready = false;
    DeviceBuffer storage;
    const float* alpha     = nullptr; ///< FP32 [4 rows][4 features] = softmax(fuse_weight) per row (fp32)
    const void* mix_norms  = nullptr; ///< BF16 [2304, 4]: feature_norm of draft layers 0-2, then the Markov one
    const void* w_sp       = nullptr; ///< BF16 [768, 512]   = rnn_proj[:, 0:512]    ([state | w1[prev]])
    const void* w_hf       = nullptr; ///< BF16 [768, 4608]  = rnn_proj[:, 512:5120] ([hidden | f_rnn])
    const void* fold_w     = nullptr; ///< BF16 [201024, 256] = bf16(unembed . hidden_out), fp64 (null with fp8_fold)
    /// FP8 fold (default since 8a146482; SUROGATE_SERVE_DSPARK_BF16_FOLD=1 keeps the BF16 fold; M1 `fp8fold`: no acceptance loss): per-row e4m3 codes of the
    /// BF16 fold + fp32 row scales; half the bytes of every Markov vocabulary pass.
    bool fp8_fold                   = false;
    const std::uint8_t* fold_codes  = nullptr; ///< U8 [201024, 256]
    const float* fold_scales        = nullptr; ///< FP32 [201024]
    const void* fold_b     = nullptr; ///< BF16 [201024]      = bf16(unembed . hidden_out_bias), fp64
    const float* inv_freq  = nullptr; ///< FP32 [64] RoPE inverse frequencies (theta 1e6)
    /// FP8 drafter weights (default; M1 `fp8w`: no acceptance loss): the dense linears as per-row e4m3 + scale, run by
    /// ops::dspark_gemm_w8; the BF16 originals then sit in a mapped host bank and are never read again.
    bool fp8 = false;
    DeviceBuffer w8_storage;
    DeviceBuffer fold_storage; ///< the FP8 fold's codes + scales
    std::array<DSparkLayerW8, 3> w8{};
};

/// The DFlash runtime's view of the drafter (`--speculative dflash` on inkling2): 3 layers, all "local" for DFlash's
/// own rings (unused by DSpark, 64 tokens each) so no DFlash full pool is planned; the 4 target features are captured
/// as sink layers {0, 1, 2, 3}; block 8. Checks the artifact's dspark block is Inkling-2-Mini's.
[[nodiscard]] family::DFlashGeometry dspark_dflash_geometry(const load::ArtifactConfig& config);

/// Derives the drafter's load-time weights from lane b's bound draft (w.has_draft); synchronous.
void derive_dspark(const load::InklingWeights& w, DSparkDerived& out, bool fp8_weights = false, bool fp8_fold = false);
/// The FP8 Markov fold (default; SUROGATE_SERVE_DSPARK_BF16_FOLD=1 keeps the BF16 fold).
[[nodiscard]] bool dspark_fp8_fold_requested();

/// SUROGATE_SERVE_DSPARK_BF16_WEIGHTS=1 (read once): keep the drafter's dense linears BF16 on the device (A/B).
[[nodiscard]] bool dspark_bf16_weights_forced();
/// The FP8 drafter weights (default; SUROGATE_SERVE_DSPARK_BF16_WEIGHTS=1 keeps BF16).
[[nodiscard]] bool dspark_fp8_weights_requested();

/// The draft KV: planes of draft layers 0 / 1 / 2 (text-pool layers 48 / 49 / 50).
struct DSparkKv {
    std::array<ops::DSparkPagedPlane, 3> planes{};
    /// The same planes as the pool's batch views, for lane c's inkling_draft_attention (FP8 planes only).
    std::array<PagedKVBatchLayerView, 3> views{};
    std::uint32_t max_visible_keys = 0;  ///< the sequence capacity (lane c's workspace / split plan)
    bool naive_attention           = false; ///< ops::dspark_block_attention (BF16 planes, or the A/B knob)
};

/// SUROGATE_SERVE_DSPARK_NAIVE_ATTN=1 (read once): the naive block attention instead of lane c's (A/B only).
[[nodiscard]] bool dspark_naive_attention_forced();

/// Context append: the committed columns' target features -> each draft layer's context K / V in its plane, and
/// f_rnn (mix row 3 of each lane's last committed column) into the per-lane buffer.
struct DSparkAppend {
    const void* features         = nullptr; ///< BF16 [feature_stride >= 9216, W, B]
    std::int64_t feature_stride  = 0;
    std::int32_t width           = 0;
    std::int32_t batch           = 0;
    const std::int32_t* positions  = nullptr; ///< I32 [W, B]
    const std::int32_t* counts     = nullptr; ///< I32 [B] committed columns per lane; null: all W (prefill)
    const std::int32_t* lanes      = nullptr; ///< I32 [B] f_rnn column of each lane; null: b
    const std::int32_t* table_rows = nullptr; ///< I32 [B] block-table rows of the draft planes
    void* f_rnn                    = nullptr; ///< BF16 [2304, lane capacity]
};

/// One proposal: lane b's anchor token at anchor_positions[b] (context = positions below it, already appended)
/// -> k drafts for positions anchor + 1 .. anchor + k. Writes the block's own K / V at [anchor, anchor + 8) of
/// the draft planes (the next append overwrites the committed ones).
struct DSparkPropose {
    const std::int32_t* anchors          = nullptr; ///< I32 [B] token ids
    const std::int32_t* anchor_positions = nullptr; ///< I32 [B]
    const std::int32_t* lanes            = nullptr; ///< I32 [B] f_rnn columns; null: b
    const std::int32_t* table_rows       = nullptr; ///< I32 [B]
    std::int32_t batch                   = 0;
    std::int32_t k                       = 0;       ///< 1..7
    const void* f_rnn                    = nullptr; ///< BF16 [2304, lane capacity]
    std::int32_t* drafts                 = nullptr; ///< I32 [draft_stride, B]: draft j of lane b at j + stride * b
    std::int32_t draft_stride            = 0;       ///< >= k; 0: k
    void* hidden_out                     = nullptr; ///< optional BF16 [2304, 8, B]: the block's final hidden (tests)
};

/// Workspace of one append of `columns` (= W * B) columns / one proposal of `batch` lanes at the widest round.
[[nodiscard]] std::size_t dspark_append_workspace_bytes(std::int32_t columns);
[[nodiscard]] std::size_t dspark_propose_workspace_bytes(std::int32_t batch, std::uint32_t max_visible_keys);

void dspark_append(const load::InklingWeights& w, const DSparkDerived& d, const DSparkKv& kv, const DSparkAppend& a,
                   WorkspaceArena& work, cudaStream_t stream);
void dspark_propose(const load::InklingWeights& w, const DSparkDerived& d, const DSparkKv& kv,
                    const DSparkPropose& p, WorkspaceArena& work, cudaStream_t stream);

} // namespace sinfer::targets::inkling2::detail
