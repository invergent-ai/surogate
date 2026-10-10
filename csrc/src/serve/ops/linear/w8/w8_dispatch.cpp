#include "ops/linear/w8/w8_dispatch.h"

#include "ops/linear/w8/w8_rowsplit_wgmma_sm90.h"

#include <cstdlib>
#include <stdexcept>
#include <string>

namespace sinfer::ops::detail {

void launch_w8_consistent(const Tensor&, const Weight&, Tensor&, cudaStream_t);

namespace {

/// The route a benchmarked shape's T bands give, or null for a shape nobody has measured.
W8Launch measured_w8_a16_launch(std::int32_t n, std::int32_t k, std::int32_t t) {
    switch (k) {
    case 10240:
        if (n == 5120) {
            if (t <= 48) { return launch_w8_small_t; }
            return launch_w8_mma_r64_c128;
        }
        break;
    case 5120:
        switch (n) {
        // surogate vendor patch (PATCHES.md #18): qwen3.5-4b mtp fc.
        case 2560:
            if (t <= 13) { return launch_w8_simt_r8_c4; }
            if (t <= 128) { return launch_w8_mma_r32_c128; }
            return launch_w8_mma_r64_c128;
        case 1024:
            if (t <= 4) { return launch_w8_simt_r8_c4; }
            if (t <= 16) { return launch_w8_simt_r8_c8; }
            return launch_w8_mma_r32_c128;
        case 6144:
            if (t <= 4) { return launch_w8_simt_r8_c4; }
            if (t <= 16) { return launch_w8_simt_r8_c8; }
            return launch_w8_mma_r64_c128;
        case 14336:
            if (t <= 48) { return launch_w8_small_t; }
            return launch_w8_mma_r64_c128;
        case 34816:
            if (t <= 40) { return launch_w8_small_t; }
            if (t <= 48) { return launch_w8_mma_r64x16_c48_k128_a1; }
            return launch_w8_mma_r64_c128;
        case 248320:
            if (t <= 33) { return launch_w8_small_t; }
            if (t <= 48) { return launch_w8_mma_r64x16_c48_k128_a1; }
            if (t <= 64) { return launch_w8_mma_r32_c64; }
            return launch_w8_mma_r64_c128;
        default:
            break;
        }
        break;
    case 6144:
        if (n == 5120) {
            if (t <= 48) { return launch_w8_small_t; }
            return launch_w8_mma_r64_c128;
        }
        // qwen3.5-2b's MTP MLP down projection (hidden 2048 from intermediate 6144).
        if (n == 2048) {
            if (t <= 16) { return launch_w8_simt_r8_c4; }
            return launch_w8_mma_r64_c128;
        }
        // Qwen3.8-Flash-Next attention/GDN output projections (2560 x 6144).
        if (n == 2560) {
            if (t <= 16) { return launch_w8_simt_r8_c4; }
            if (t <= 128) { return launch_w8_mma_r32_c128; }
            return launch_w8_mma_r64_c128;
        }
        break;
    case 17408:
        if (n == 5120) {
            if (t <= 48) { return launch_w8_small_t; }
            return launch_w8_mma_r64_c128;
        }
        break;
    // The 768-wide tower's merger (Qwen3.5 0.8B): fc1 is square at 4*768, fc2 projects to
    // the text model's 1024. Measured with sinfer_vision_tower_tune_bench --hidden 768.
    case 3072:
        if (n == 3072) {  // merger fc1
            if (t <= 44) { return launch_w8_simt_r8_c4; }
            if (t <= 444) { return launch_w8_mma_r32_c128; }
            if (t <= 764) { return launch_w8_mma_r64_c96; }
            return launch_w8_mma_r64_c128;
        }
        if (n == 1024) {  // merger fc2
            if (t <= 108) { return launch_w8_simt_r8_c4; }
            if (t <= 1596) { return launch_w8_mma_r32_c128; }
            if (t <= 2044) { return launch_w8_mma_r64_c96; }
            return launch_w8_mma_r64_c128;
        }
        break;
    case 4096:
        if (n == 2048) {
            if (t <= 48) { return launch_w8_small_t; }
            if (t <= 56) { return launch_w8_simt_r8_c4; }
            if (t <= 895) { return launch_w8_mma_r32_c128; }
            return launch_w8_mma_r64_c128;
        }
        // The 64-wide vision tower's merger fc1: 4 patches merge into 4*1024, and the
        // projection is square. Measured with sinfer_vision_tower_tune_bench --hidden
        // 1024, which agrees with the 4608 tower's schedule: r32_c128 through t=256,
        // r64_c128 above it.
        if (n == 4096) {
            if (t <= 8 || t == 12) { return launch_w8_simt_r8_c4; }
            if (t <= 256) { return launch_w8_mma_r32_c128; }
            return launch_w8_mma_r64_c128;
        }
        break;
    case 2560:
        // surogate vendor patch (PATCHES.md #18/#29): qwen3.5-4b heads. The
        // SIMT launcher's fp4 gate serves T<=16 with the batched W4 decode
        // kernel (310MB vs the 620MB W8 read; the T=14..16 gap previously
        // fell onto the runtime MMA tile at ~1.76ms per round).
        if (n == 248320 || n == 131072) {
            if (t <= 16) { return launch_w8_simt_r8_c4; }
            if (n == 248320 && t <= 32) { return launch_w8_small_t; } // PATCHES.md #29
            if (t <= 128) { return launch_w8_mma_r32_c128; }
            return launch_w8_mma_r64_c128;
        }
        // Qwen3.8-Flash-Next fused projections at hidden 2560: attention
        // q|k|gate|v (13312 rows) and GDN qkv|z (16384 rows) ride the generic
        // routes; the fused input-projection families are not extended for them.
        if (n == 13312 || n == 16384) {
            if (t <= 16) { return launch_w8_simt_r8_c4; }
            if (t <= 128) { return launch_w8_mma_r32_c128; }
            return launch_w8_mma_r64_c128;
        }
        break;
    // EmbeddingGemma (hidden 768, intermediate 1152). Every projection of the
    // encoder lands here: q and the attention output are square 768, k and v are
    // 256 rows, and the MLP gate/up are 1152. An encoder has no decode step, so
    // T is the whole prompt and the small-T bands below matter far less than they
    // do for a generative target -- they are kept only so a one-token request
    // does not fall onto a 64-row tile. Bands follow the schedule the other
    // narrow-hidden entries use; nothing here is measured yet.
    case 768:
        switch (n) {
        case 768:  // attention query, attention output
        case 256:  // attention key, attention value
        case 1152: // mlp gate, mlp up
            if (t <= 16) { return launch_w8_simt_r8_c4; }
            if (t <= 128) { return launch_w8_mma_r32_c128; }
            return launch_w8_mma_r64_c128;
        default:
            break;
        }
        break;
    // gemma-3-270m, hidden 640: attention query (4 x 256), key and value (1 x 256),
    // the two MLP projections, and the tied lm head. 640 is not a multiple of 256,
    // so only the row-split MMA routes could take these shapes, staging 8-byte
    // aligned scale rows in halves as they do for EmbeddingGemma's 1152 below; every
    // other MMA family would read eight bytes off on every odd row. Nobody has
    // measured them there, so they stay on SIMT, which is exact at any k.
    case 640:
        switch (n) {
        case 1024:
        case 256:
        case 2048:
        case 262144:
            if (t <= 16) { return launch_w8_simt_r8_c4; }
            return launch_w8_simt_r8_c8;
        default:
            break;
        }
        break;
    case 1152:
        // EmbeddingGemma's MLP down projection, and the one k here that is not a
        // multiple of 256. Its scale rows start only 8-byte aligned, which the
        // row-split MMA routes stage in two halves (k % 128 == 0, checked in
        // launch_route); the other MMA families would read eight bytes off on every
        // odd row. On SIMT this GEMM was 63% of a 32 x 220-token request on a DGX
        // Spark: 4.96 ms a layer against 144 us for each MMA projection beside it.
        if (n == 768) { // mlp down
            if (t <= 16) { return launch_w8_simt_r8_c4; }
            if (t <= 128) { return launch_w8_mma_r32_c128; }
            return launch_w8_mma_r64_c128;
        }
        break;
    case 1024:
        // gemma-3-270m's attention output (640 rows from 4 x 256).
        if (n == 640) {
            if (t <= 16) { return launch_w8_simt_r8_c4; }
            if (t <= 128) { return launch_w8_mma_r32_c128; }
            return launch_w8_mma_r64_c128;
        }
        // surogate vendor patch (PATCHES.md #13/#29): qwen3.5-0.8b heads
        // (lm head 248320, draft head 131072; hidden 1024).
        if (n == 248320 || n == 131072) {
            if (t <= 16) { return launch_w8_simt_r8_c4; }
            if (n == 248320 && t <= 32) { return launch_w8_small_t; } // PATCHES.md #29
            if (t <= 128) { return launch_w8_mma_r32_c128; }
            return launch_w8_mma_r64_c128;
        }
        // qwen3-0.6b's lm head (151936 rows at hidden 1024). It rides the same
        // routes as the 0.8b heads above; the T<=32 small-T arm there is an
        // instantiation measured for 248320 rows, so this shape takes the SIMT
        // band and the runtime MMA tiles either side of it.
        if (n == 151936) {
            if (t <= 16) { return launch_w8_simt_r8_c4; }
            if (t <= 128) { return launch_w8_mma_r32_c128; }
            return launch_w8_mma_r64_c128;
        }
        // The MTP block's fused attention projection (5120) and its MLP gate/up
        // (2 * 3584). The main layers reach their own fused wrappers, so these
        // shapes appear only under speculation, at the verify width (T = draft
        // window + 1) and at T=1 for the proposal steps.
        if (n == 5120 || n == 7168) {
            if (t <= 16) { return launch_w8_simt_r8_c4; }
            return launch_w8_mma_r64_c128;
        }
        // Its key/value (2 kv heads x 256) and query/gate (8 q heads x 256),
        // taken unfused because the fused pair's route tables cover only the
        // 27B and 35B geometries.
        if (n == 512 || n == 2048) {
            if (t <= 16) { return launch_w8_simt_r8_c4; }
            return launch_w8_mma_r32_c128;
        }
        break;
    // qwen3.5-0.8b's MTP MLP down projection (hidden 1024 from intermediate 3584);
    // speculation-only, like its gate/up sibling above.
    case 3584:
        if (n == 1024) {
            if (t <= 16) { return launch_w8_simt_r8_c4; }
            return launch_w8_mma_r64_c128;
        }
        break;
    case 2048:
        // gemma-3-270m's MLP down projection (hidden 640 from intermediate 2048).
        if (n == 640) {
            if (t <= 16) { return launch_w8_simt_r8_c4; }
            if (t <= 128) { return launch_w8_mma_r32_c128; }
            return launch_w8_mma_r64_c128;
        }
        // tinyllama-1.1b's lm head (32000 rows at hidden 2048). Untied, so it is
        // a matrix of its own rather than the embedding read a second time.
        if (n == 32000) {
            if (t <= 16) { return launch_w8_simt_r8_c4; }
            if (t <= 128) { return launch_w8_mma_r32_c128; }
            return launch_w8_mma_r64_c128;
        }
        switch (n) {
        // surogate vendor patch (PATCHES.md #16/#29): qwen3.5-2b heads.
        case 248320:
        case 131072:
            if (t <= 16) { return launch_w8_simt_r8_c4; }
            if (n == 248320 && t <= 32) { return launch_w8_small_t; } // PATCHES.md #29
            if (t <= 128) { return launch_w8_mma_r32_c128; }
            return launch_w8_mma_r64_c128;
        case 1024:
        // The 2b MTP block's unfused key/value (2 kv heads x 256) and its
        // query/gate (8 q heads x 256).
        case 512:
        case 2048:
            if (t <= 4) { return launch_w8_simt_r8_c4; }
            if (t <= 16) { return launch_w8_simt_r8_c8; }
            return launch_w8_mma_r32_c128;
        // The 2b MTP block's fused attention projection; see the k=1024 note.
        case 5120:
            if (t <= 16) { return launch_w8_simt_r8_c4; }
            return launch_w8_mma_r64_c128;
        case 9216:
            if (t <= 13) { return launch_w8_simt_r8_c4; }
            if (t <= 128) { return launch_w8_mma_r32_c128; }
            return launch_w8_mma_r64_c128;
        case 12288:
            if (t <= 16) { return launch_w8_simt_r8_c4; }
            return launch_w8_mma_r64_c128;
        default:
            break;
        }
        break;
    case 4608:
        if (t > 32768) { break; }
        switch (n) {
        case 2048:
            if (t <= 14 || t == 16 || t == 20 || t == 24 || t == 28 || t == 32) {
                return launch_w8_simt_r8_c4;
            }
            if (t <= 871) { return launch_w8_mma_r32_c128; }
            return launch_w8_mma_r64_c128;
        case 4608:
            if (t <= 8 || t == 12) { return launch_w8_simt_r8_c4; }
            if (t <= 256) { return launch_w8_mma_r32_c128; }
            return launch_w8_mma_r64_c128;
        case 5120:
            if (t <= 4) { return launch_w8_simt_r8_c4; }
            if (t == 5) { return launch_w8_simt_r8_c8; }
            return launch_w8_mma_r64_c128;
        default:
            break;
        }
        break;
    case 16384:
        if (n != 2048) { break; }
        if (t == 1) { return launch_w8_decode_r4; }
        if (t <= 48) { return launch_w8_exact_t_splitk; }
        if (t <= 128) { return launch_w8_dflash_medium; }
        if (t <= 144) { return launch_w8_medium_splitk_c144; }
        if (t <= 255) { return launch_w8_mma_r32_c128; }
        if (t <= 384) { return launch_w8_mma_r32_c64; }
        if (t <= 480) { return launch_w8_mma_r32_c96; }
        if (t == 481) { return launch_w8_exact_mma_r32_c96; }
        if (t <= 640) { return launch_w8_mma_r32_c128; }
        if (t <= 668) { return launch_w8_exact_mma_r32_c128; }
        if (t <= 672) { return launch_w8_mma_r48_c96; }
        if (t == 673) { return launch_w8_exact_mma_r48_c96; }
        if (t <= 704) { return launch_w8_mma_r48_c64; }
        if (t <= 784) { return launch_w8_mma_r48_c112; }
        if (t <= 896) { return launch_w8_mma_r48_c128; }
        if (t <= 912) { return launch_w8_exact_mma_r48_c128; }
        if (t <= 960) { return launch_w8_mma_r64_c96; }
        if (t <= 1007) { return launch_w8_exact_mma_r64_c96; }
        if (t == 1008) { return launch_w8_mma_r64_c112; }
        if (t <= 1119) { return launch_w8_mma_r64_c128; }
        if (t == 1120) { return launch_w8_mma_r64_c112; }
        if (t <= 1280) { return launch_w8_mma_r64_c128; }
        if (t <= 1313) { return launch_w8_exact_mma_r64_c128; }
        if (t <= 1344) { return launch_w8_mma_r128_c64; }
        if (t <= 1440) { return launch_w8_mma_r96_c96; }
        if (t <= 1500) { return launch_w8_exact_mma_r96_c96; }
        if (t <= 1680) { return launch_w8_mma_r128_c80; }
        if (t <= 1745) { return launch_w8_exact_mma_r128_c80; }
        if (t <= 1791) { return launch_w8_mma_r48_c128; }
        if (t == 1792) { return launch_w8_mma_r64_c128; }
        if (t <= 1919) { return launch_w8_mma_r48_c128; }
        if (t == 1920) { return launch_w8_mma_r64_c128; }
        if (t <= 1953) { return launch_w8_exact_mma_r64_c128; }
        if (t <= 2016) { return launch_w8_mma_r64_c96; }
        if (t <= 2048) { return launch_w8_exact_mma_r64_c96; }
        if (t <= 2112) { return launch_w8_mma_r96_c96; }
        return launch_w8_mma_r64_c128;
    default:
        break;
    }
    return nullptr;
}

/// SUROGATE_SERVE_W8_GENERIC_MMA=0 restores the earlier generic routes: the batch-consistent
/// kernel at every width, and SIMT for any shape that is not whole 128-row tiles over whole
/// 256s of K.
bool generic_mma_enabled() {
    static const bool enabled = [] {
        const char* raw = std::getenv("SUROGATE_SERVE_W8_GENERIC_MMA");
        return raw == nullptr || raw[0] != '0';
    }();
    return enabled;
}

/// The same family of launchers at default bands, for a shape nobody has measured.
///
/// The launchers are shape-generic -- they take n, k and T from the tensors and handle a
/// partial tile -- so refusing here would be a policy, not a limit. The real constraints are
/// the scale rows' alignment (16 bytes, k % 256, for every MMA family but the row-split one,
/// which stages 8-byte halves and needs k % 128) and the batch-consistent kernel's 16-row
/// tiles.
///
/// Through 64 columns (every decode round up to 64 lanes) a shape keeps one K reduction at
/// every width: the batch-consistent kernel, or SIMT where that kernel cannot tile it. Small
/// changes at a BF16 boundary can otherwise flip FP8 cache codes between a lane served alone
/// and in a batch. Wider rounds are prompt chunks, where that kernel cost what the bands
/// below save: on a DGX Spark (sinfer_w8_route_bench, 21 Gemma shapes), 1.4-2.7x at 96 to
/// 2,048 columns over whole 256s of K, 12-18x on Gemma 3 1B's k = 1,152, and 26x on Gemma 3's
/// 262,208-row head, which SIMT had served at every width. Every route there matched SIMT.
/// Hopper's batch-consistent kernel runs on wgmma from 33 columns and keeps every width.
///
/// A shape that lands here works but is not tuned. Measure it and give it an entry above.
W8Launch generic_w8_a16_launch(std::int32_t n, std::int32_t k, std::int32_t t) {
    const bool k_aligned   = k % kW8MmaScaleRowAlignmentK == 0;
    const bool row_split_k = k % kW8RowSplitMmaScaleRowAlignmentK == 0;
    if (!generic_mma_enabled()) {
        if (!k_aligned || n % kW8MmaRowAlignmentN != 0) { return launch_w8_simt_r8_c4; }
        return launch_w8_consistent;
    }
    // Rows in whole 64s: the MMA tiles' measured reach (262,208 = 4,097 x 64 matched SIMT on
    // every route); anything else, or a k no MMA family stages, keeps SIMT at every width.
    if (n % 64 != 0 || !row_split_k) { return launch_w8_simt_r8_c4; }
    if (k_aligned) {
        if (t <= 64 || w8_wgmma_available()) { return launch_w8_consistent; }
    } else {
        // Only the row-split MMA routes take these; SIMT is the consistent route, and from
        // eight columns it was 1.2-10x slower than them on Gemma 3 1B's projections.
        if (t <= 4) { return launch_w8_simt_r8_c4; }
        if (t <= 64) { return launch_w8_mma_r32_c64; }
    }
    if (t <= 128) { return launch_w8_mma_r32_c128; }
    return launch_w8_mma_r64_c128;
}

} // namespace

W8Launch select_w8_a16_launch(std::int32_t n, std::int32_t k, std::int32_t t) {
    if (t <= 0) { throw std::invalid_argument("w8 linear: unsupported shape or T"); }
    if (const W8Launch measured = measured_w8_a16_launch(n, k, t)) { return measured; }
    return generic_w8_a16_launch(n, k, t);
}

bool w8_marlin_eligible(std::int32_t n, std::int32_t k) {
    // The Marlin band serves measured shapes only, whole 128s both ways: a generic shape keeps
    // the routes above, whose numerics through decode widths do not depend on the batch.
    return k % kW8MmaScaleRowAlignmentK == 0 && n % kW8MmaRowAlignmentN == 0 &&
           measured_w8_a16_launch(n, k, 1) != nullptr;
}

W8Launch select_w8_launch(std::int32_t n, std::int32_t k, std::int32_t t, LinearPolicy policy) {
    switch (policy) {
    case LinearPolicy::A16Only:
    case LinearPolicy::AllowA8:
        return select_w8_a16_launch(n, k, t);
    case LinearPolicy::AllowA4:
        break;
    }
    throw std::invalid_argument("w8 linear: unsupported policy");
}

void w8_dispatch(const Tensor& x, const Weight& w, Tensor& out, LinearPolicy policy,
                 cudaStream_t stream) {
    W8Launch launch = select_w8_launch(w.n, w.k, x.ne[1], policy);
    // A padded weight keeps SIMT wherever the generic routes would have taken it elsewhere.
    if (w.padded_shape[1] != w.k && measured_w8_a16_launch(w.n, w.k, 1) == nullptr) {
        launch = launch_w8_simt_r8_c4;
    }
    launch(x, w, out, stream);
}

} // namespace sinfer::ops::detail
