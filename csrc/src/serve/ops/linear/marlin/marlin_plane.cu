// Derived Marlin residency registry (see marlin_plane.h for the scheme).

#include "ops/linear/marlin/marlin_plane.h"
#include "core/engine_context.h"

#include "ops/linear/marlin/marlin_gemm.h"
#include "ops/linear/marlin/marlin_repack.h"

#include "core/device.h"

#include <cstdlib>
#include <map>
#include <mutex>

namespace sinfer::ops::detail {
namespace {

bool g_enabled = false;
std::mutex g_mutex;
std::size_t g_bytes = 0;

struct PlaneEntry {
    void* b_packed    = nullptr;
    void* scales      = nullptr;
    cudaEvent_t ready = nullptr;
};

std::map<const void*, PlaneEntry> g_planes;

// Guard bytes in front of the locks array; see the note at its allocation.
constexpr std::size_t kLockGuardBytes = 256;

// The scratch is per device: pipeline stages on several devices each run the Marlin route,
// and a buffer allocated on one device is an illegal address on another.
struct ScratchState {
    MarlinScratch scratch;
    std::size_t out_bytes = 0;
    std::size_t a_bytes   = 0;
    bool frozen           = false;
};
// Everything mutable the plane owns, homed per engine so two models in one
// process cannot race the staging scratch or close each other's adoption.
struct MarlinPlaneState {
    std::map<int, ScratchState> scratch_by_device; // pipeline stages: one per device
    bool adoption_closed          = false;
    int fixed_m                   = 32;
    void* fused_parent            = nullptr;
    std::size_t fused_parent_bytes = 0;
    ~MarlinPlaneState() {
        for (auto& [device, state] : scratch_by_device) {
            if (state.scratch.gemm_out != nullptr) { (void)cudaFree(state.scratch.gemm_out); }
            if (state.scratch.a_pad != nullptr) { (void)cudaFree(state.scratch.a_pad); }
        }
        if (fused_parent != nullptr) { (void)cudaFree(fused_parent); }
    }
};
MarlinPlaneState& plane_state() { return engine_slot<MarlinPlaneState>(); }
ScratchState& scratch_state() {
    int device = 0;
    (void)cudaGetDevice(&device);
    return plane_state().scratch_by_device[device];
}
#define g_scratch (scratch_state().scratch)
#define g_scratch_out_bytes (scratch_state().out_bytes)
#define g_scratch_a_bytes (scratch_state().a_bytes)
#define g_scratch_frozen (scratch_state().frozen)
#define g_adoption_closed (plane_state().adoption_closed)
#define g_fixed_m (plane_state().fixed_m)
#define g_fused_parent (plane_state().fused_parent)
#define g_fused_parent_bytes (plane_state().fused_parent_bytes)

bool ensure_scratch(std::size_t out_bytes, std::size_t a_bytes, cudaStream_t stream) {
    if (g_scratch.gemm_out != nullptr && out_bytes <= g_scratch_out_bytes &&
        a_bytes <= g_scratch_a_bytes) {
        return true;
    }
    // The scratch grows only until the engine freezes it after capture: from
    // then on its addresses live inside captured graphs and must not move.
    if (g_scratch_frozen) { return false; }
    if (g_scratch.gemm_out != nullptr) {
        const std::size_t want_out = out_bytes > g_scratch_out_bytes ? out_bytes
                                                                     : g_scratch_out_bytes;
        const std::size_t want_a   = a_bytes > g_scratch_a_bytes ? a_bytes : g_scratch_a_bytes;
        void* grown_out = nullptr;
        void* grown_a   = nullptr;
        if (cudaMalloc(&grown_out, want_out) != cudaSuccess) { return false; }
        if (cudaMalloc(&grown_a, want_a) != cudaSuccess) {
            cudaFree(grown_out);
            return false;
        }
        cudaMemsetAsync(grown_a, 0, want_a, stream);
        // The buffers being replaced may still be referenced by kernels this
        // stream has not finished — the warmup round derives every weight in
        // one pass, and a later, larger weight (the vocab head dwarfs any
        // layer) triggers growth while earlier layers' GEMMs are still in
        // flight. Freeing them under those kernels is a use-after-free that
        // corrupts whatever is allocated next. Drain first.
        cudaStreamSynchronize(stream);
        cudaFree(g_scratch.gemm_out);
        cudaFree(g_scratch.a_pad);
        g_bytes -= g_scratch_out_bytes + g_scratch_a_bytes;
        g_scratch.gemm_out  = grown_out;
        g_scratch.a_pad     = grown_a;
        g_scratch_out_bytes = want_out;
        g_scratch_a_bytes   = want_a;
        g_bytes += want_out + want_a;
        return true;
    }
    int device = 0;
    cudaGetDevice(&device);
    int sms = 0;
    cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device);
    const std::size_t c_tmp_bytes = marlin_c_tmp_floats(sms, marlin_fixed_m()) * sizeof(float);
    std::size_t lock_bytes = marlin_workspace_locks_count(sms) * sizeof(int);
    void* out_buf   = nullptr;
    void* a_buf     = nullptr;
    void* c_tmp_buf = nullptr;
    void* lock_buf  = nullptr;
    if (cudaMalloc(&out_buf, out_bytes) != cudaSuccess) { return false; }
    if (cudaMalloc(&a_buf, a_bytes) != cudaSuccess) {
        cudaFree(out_buf);
        return false;
    }
    if (cudaMalloc(&c_tmp_buf, c_tmp_bytes) != cudaSuccess) {
        cudaFree(out_buf);
        cudaFree(a_buf);
        return false;
    }
    // Defensive, NOT a fix for the wide-band fault. The reduce computes
    // locks_off = (iters * blockIdx.x) / k_tiles - 1 on the branch taken when
    // the problem's mn-tile count is below the grid width, which is -1 for
    // block 0 (marlin_template.h:422) — a real one-int write before the array
    // that lands in the adjacent allocation instead of faulting. Front-padding
    // makes it harmless. It was measured against the 0.8B wide-band corruption
    // and did NOT change the failure rate (3 clean / 1 crash over four 90 s
    // runs, same as unguarded), so the observed fault is elsewhere.
    lock_bytes += kLockGuardBytes;
    if (cudaMalloc(&lock_buf, lock_bytes) != cudaSuccess) {
        cudaFree(out_buf);
        cudaFree(a_buf);
        cudaFree(c_tmp_buf);
        return false;
    }
    cudaMemsetAsync(a_buf, 0, a_bytes, stream);
    cudaMemsetAsync(lock_buf, 0, lock_bytes, stream);
    lock_buf = static_cast<void*>(static_cast<char*>(lock_buf) + kLockGuardBytes);
    g_scratch = MarlinScratch{out_buf, a_buf, c_tmp_buf, static_cast<int*>(lock_buf), sms};
    g_scratch_out_bytes = out_bytes;
    g_scratch_a_bytes   = a_bytes;
    g_bytes += out_bytes + a_bytes + c_tmp_bytes + lock_bytes;
    return true;
}

} // namespace

void marlin_plane_set_enabled(bool enabled) noexcept {
    const char* veto = std::getenv("SUROGATE_SERVE_MARLIN");
    g_enabled = enabled && !(veto != nullptr && veto[0] == '0');
}
bool marlin_plane_enabled() noexcept { return g_enabled; }
std::size_t marlin_plane_bytes() noexcept { return g_bytes; }
MarlinScratch marlin_scratch() noexcept { return g_scratch; }
void marlin_plane_freeze_scratch() noexcept { g_scratch_frozen = true; }

// Adoption changes an op's kernel selection, so it changes a captured graph's
// topology. It must close BEFORE the first capture, not merely before the last:
// a weight that adopts between two captures leaves the families disagreeing and
// the next exec update fails with cudaErrorGraphExecUpdateFailure — observed
// exactly that way on the 27B. Separate from the scratch freeze, which is about
// pointer stability and happens later.
void marlin_fp8_close_adoption() noexcept { g_adoption_closed = true; }

void marlin_set_fixed_m(int lanes) noexcept {
    // The band follows the lane ceiling. It was pinned to 32 while a wide
    // band corrupted the 0.8B under sustained load; that fault was the
    // mixed-round pad race (PATCHES.md #50), not Marlin, and with it fixed
    // the wide band is stable across repeated 90-second runs.
    // SUROGATE_SERVE_MARLIN_NARROW=1 pins it back to 32 for bisecting.
    const char* narrow      = std::getenv("SUROGATE_SERVE_MARLIN_NARROW");
    const bool force_narrow = narrow != nullptr && narrow[0] == '1';
    const int rounded       = (force_narrow || lanes <= 32)
                                  ? 32
                                  : (lanes <= 64 ? 64 : ((lanes + 31) / 32) * 32);
    if (g_scratch.gemm_out == nullptr) { g_fixed_m = rounded; }
}
int marlin_fixed_m() noexcept { return g_fixed_m; }

int marlin_min_band_tokens() noexcept {
    static const int floor_tokens = [] {
        const char* env = std::getenv("SUROGATE_SERVE_MARLIN_MIN_T");
        if (env == nullptr) { return 17; }
        const int parsed = std::atoi(env);
        return parsed >= 1 && parsed <= marlin_fixed_m() ? parsed : 17;
    }();
    return floor_tokens;
}

MarlinPlane marlin_plane_for(const Weight& weight, cudaStream_t stream) {
    if (!g_enabled || weight.qtype != QType::W8G32_F16S ||
        weight.layout != QuantLayout::RowSplit || weight.scale_dtype != DType::FP16 ||
        weight.group != 32 || weight.qdata == nullptr || weight.scales == nullptr ||
        (weight.k % 64) != 0 || (weight.n % 64) != 0) {
        return {};
    }

    cudaStreamCaptureStatus capture_status = cudaStreamCaptureStatusNone;
    (void)cudaStreamIsCapturing(stream, &capture_status);
    const bool capturing = capture_status != cudaStreamCaptureStatusNone;

    std::lock_guard<std::mutex> lock(g_mutex);
    auto found = g_planes.find(weight.qdata);
    if (found != g_planes.end()) {
        if (!capturing && found->second.ready != nullptr) {
            cudaStreamWaitEvent(stream, found->second.ready, 0);
        }
        return {found->second.b_packed, found->second.scales};
    }
    if (capturing) { return {}; }

    const int n = weight.n;
    const int k = weight.k;
    const std::size_t b_bytes = marlin_b_out_words(n, k) * sizeof(std::uint32_t);
    const std::size_t s_bytes = static_cast<std::size_t>(k / 32) * n * 2;
    const std::size_t gptq_bytes = static_cast<std::size_t>(k / 4) * n * sizeof(std::uint32_t);

    std::size_t free_bytes = 0, total_bytes = 0;
    if (cudaMemGetInfo(&free_bytes, &total_bytes) != cudaSuccess ||
        free_bytes < 2 * (b_bytes + s_bytes + gptq_bytes) ||
        g_bytes + b_bytes + s_bytes > total_bytes / 4) {
        // Global budget: planes duplicate the residency they accelerate, so a
        // large model would otherwise consume the memory the KV cache and the
        // decode graphs need and abort capture. A quarter of the card bounds
        // it; weights past the budget keep the engine's own kernels.
        return {};
    }
    if (!ensure_scratch(static_cast<std::size_t>(n) * marlin_fixed_m() * 2,
                        static_cast<std::size_t>(k) * marlin_fixed_m() * 2, stream)) {
        return {};
    }

    PlaneEntry entry;
    void* gptq_tmp = nullptr;
    if (cudaMalloc(&entry.b_packed, b_bytes) != cudaSuccess) { return {}; }
    if (cudaMalloc(&entry.scales, s_bytes) != cudaSuccess) {
        cudaFree(entry.b_packed);
        return {};
    }
    if (cudaMalloc(&gptq_tmp, gptq_bytes) != cudaSuccess) {
        cudaFree(entry.b_packed);
        cudaFree(entry.scales);
        return {};
    }
    marlin_repack_w8g32(weight.qdata, weight.scales, n, k, gptq_tmp, entry.b_packed,
                        entry.scales, stream);
    cudaEventCreateWithFlags(&entry.ready, cudaEventDisableTiming);
    cudaEventRecord(entry.ready, stream);
    cudaEventSynchronize(entry.ready);
    cudaFree(gptq_tmp);
    g_bytes += b_bytes + s_bytes;
    g_planes.emplace(weight.qdata, entry);
    return {entry.b_packed, entry.scales};
}

MarlinScratch marlin_scratch_for(const Weight& weight, cudaStream_t stream) {
    const MarlinPlane plane = marlin_plane_for(weight, stream);
    if (plane.b_packed == nullptr) { return {}; }
    return marlin_scratch();
}

MarlinPlane marlin_fp8_plane_for(const Weight& weight, cudaStream_t stream) {
    // Opt-in: the FP8 residency belongs to the large models, where a plane
    // duplicating it starves the KV cache and aborts graph capture (measured
    // on the 27B, which OOMs at capture with planes on and serves normally
    // with them off). The kernels are correctness-validated; what is missing
    // is a residency-replacing path rather than a duplicating one.
    static const bool fp8_opt_in = [] {
        const char* env = std::getenv("SUROGATE_SERVE_MARLIN_FP8");
        return env != nullptr && env[0] == '1';
    }();
    if (!g_enabled || weight.qtype != QType::FP8_E4M3FN_ROW_BF16S ||
        weight.scale_dtype != DType::BF16 || weight.qdata == nullptr ||
        weight.scales == nullptr || (weight.k % 64) != 0 || (weight.n % 64) != 0) {
        return {};
    }
    // Already adopted: the weight IS the plane. The scratch still has to exist —
    // an adopted weight skips the repack path that would otherwise allocate it,
    // and marlin_fp8_run declines without it, which for a Marlin-tile weight is
    // a hard error rather than a fallback.
    if (weight.layout == QuantLayout::MarlinTiles) {
        cudaStreamCaptureStatus adopted_capture = cudaStreamCaptureStatusNone;
        (void)cudaStreamIsCapturing(stream, &adopted_capture);
        if (adopted_capture == cudaStreamCaptureStatusNone) {
            std::lock_guard<std::mutex> scratch_lock(g_mutex);
            (void)ensure_scratch(static_cast<std::size_t>(weight.n) * marlin_fixed_m() * 2,
                                 static_cast<std::size_t>(weight.k) * marlin_fixed_m() * 2,
                                 stream);
        }
        return {const_cast<void*>(weight.qdata), const_cast<void*>(weight.scales)};
    }
    if (!fp8_opt_in || weight.layout != QuantLayout::RowScale) { return {}; }

    cudaStreamCaptureStatus capture_status = cudaStreamCaptureStatusNone;
    (void)cudaStreamIsCapturing(stream, &capture_status);
    const bool capturing = capture_status != cudaStreamCaptureStatusNone;

    std::lock_guard<std::mutex> lock(g_mutex);
    auto found = g_planes.find(weight.qdata);
    if (found != g_planes.end()) {
        if (!capturing && found->second.ready != nullptr) {
            cudaStreamWaitEvent(stream, found->second.ready, 0);
        }
        return {found->second.b_packed, found->second.scales};
    }
    if (capturing) { return {}; }

    const int n = weight.n;
    const int k = weight.k;
    const std::size_t b_bytes    = marlin_b_out_words(n, k) * sizeof(std::uint32_t);
    const std::size_t s_bytes    = static_cast<std::size_t>(n) * 2;
    const std::size_t gptq_bytes = static_cast<std::size_t>(k / 4) * n * sizeof(std::uint32_t);

    // Residency replacement was implemented here and REVERTED as unsafe; see
    // PATCHES.md #59. Writing the packed layout back over the weight is
    // size-exact for FP8 and costs nothing permanent, but it is only sound if
    // no consumer can ever read those bytes as anything other than Marlin —
    // and every decline path in marlin_fp8_run (capture without a cached plane,
    // an unavailable scratch, a non-contiguous view) falls back to the FP8
    // kernel, which then reinterprets Marlin bytes as e4m3 and emits garbage.
    // The fix is the layout TAG from design/serve-engine-multiarch.md item 3:
    // the converter writes Marlin tiles, the loader records the layout per
    // tensor, and routing dispatches on it, so a fallback is a type error
    // rather than a silent misread.
    std::size_t free_bytes = 0, total_bytes = 0;
    if (cudaMemGetInfo(&free_bytes, &total_bytes) != cudaSuccess ||
        free_bytes < 2 * (b_bytes + s_bytes + gptq_bytes) ||
        g_bytes + b_bytes + s_bytes > total_bytes / 4) {
        // Global budget: planes duplicate the residency they accelerate, so a
        // large model would otherwise consume the memory the KV cache and the
        // decode graphs need and abort capture. A quarter of the card bounds
        // it; weights past the budget keep the engine's own kernels.
        return {};
    }
    if (!ensure_scratch(static_cast<std::size_t>(n) * marlin_fixed_m() * 2,
                        static_cast<std::size_t>(k) * marlin_fixed_m() * 2, stream)) {
        return {};
    }

    PlaneEntry entry;
    void* gptq_tmp = nullptr;
    if (cudaMalloc(&entry.b_packed, b_bytes) != cudaSuccess) { return {}; }
    if (cudaMalloc(&entry.scales, s_bytes) != cudaSuccess) {
        cudaFree(entry.b_packed);
        return {};
    }
    if (cudaMalloc(&gptq_tmp, gptq_bytes) != cudaSuccess) {
        cudaFree(entry.b_packed);
        cudaFree(entry.scales);
        return {};
    }
    marlin_repack_fp8_row(weight.qdata, weight.scales, n, k, gptq_tmp, entry.b_packed,
                          entry.scales, stream);
    cudaEventCreateWithFlags(&entry.ready, cudaEventDisableTiming);
    cudaEventRecord(entry.ready, stream);
    cudaEventSynchronize(entry.ready);
    cudaFree(gptq_tmp);
    g_bytes += b_bytes + s_bytes;
    g_planes.emplace(weight.qdata, entry);
    return {entry.b_packed, entry.scales};
}

// Residency adoption (design/serve-engine-multiarch.md item 3). Repacks the
// weight into Marlin tiles IN PLACE and stamps QuantLayout::MarlinTiles.
//
// For FP8 the packed form is exactly n*k bytes and the scales exactly n*2 — the
// sizes the residency already holds — so adoption costs nothing permanent, only
// transient repack buffers. That is what makes it viable on a model whose
// weights leave no room for a duplicate plane, which is precisely the case
// (the 27B) where the Marlin kernels are worth the most: 2.4-2.75x on the
// families that are 47% of its device time.
//
// The tag is the safety mechanism. Once stamped, routes that gate on
// QuantLayout::RowScale no longer match the weight, so a path that cannot
// serve Marlin tiles reaches its "unsupported weight format" throw instead of
// misreading the tiles. Adoption must therefore only be applied to weights
// whose consumers can all serve Marlin; the caller decides that, and a wrong
// decision fails loudly at plan time rather than silently at run time.
// Adoption trigger for a route that can serve Marlin tiles at any T.
//
// Called from the four FP8 families that have a Marlin route — linear_add,
// linear_swiglu, attn_input_proj, gdn_input_proj. Reaching one of them IS the
// safety precondition: a weight only becomes Marlin-tiled if a route that can
// read tiles is asking for it, and any other consumer of the same weight then
// fails at plan time rather than misreading (PATCHES.md #60).
//
// Never adopts during capture — it allocates and synchronizes — so adoption
// happens on the warmup pass, before any graph is recorded.
// Dedicated staging for the fused parent an adopted weight produces.
//
// This buffer does not belong in the workspace arena. Its size depends on the
// widest call a target makes, the arena is planned per phase, and threading a
// conditional reservation through every phase that might see an adopted weight
// is both invasive and easy to get wrong — the first attempt was short by
// exactly one buffer and surfaced as an arena exhaustion during warmup. It is
// scratch with the same lifetime as the Marlin scratch beside it, so it lives
// there instead, and grows on demand before capture.

void* marlin_fused_parent(std::size_t bytes, cudaStream_t stream) {
    if (bytes == 0) { return nullptr; }
    std::lock_guard<std::mutex> lock(g_mutex);
    if (g_fused_parent != nullptr && bytes <= g_fused_parent_bytes) { return g_fused_parent; }
    cudaStreamCaptureStatus status = cudaStreamCaptureStatusNone;
    (void)cudaStreamIsCapturing(stream, &status);
    if (status != cudaStreamCaptureStatusNone) { return nullptr; }
    void* grown = nullptr;
    if (cudaMalloc(&grown, bytes) != cudaSuccess) { return nullptr; }
    if (g_fused_parent != nullptr) {
        cudaStreamSynchronize(stream);
        cudaFree(g_fused_parent);
        g_bytes -= g_fused_parent_bytes;
    }
    g_fused_parent       = grown;
    g_fused_parent_bytes = bytes;
    g_bytes += bytes;
    return g_fused_parent;
}

std::size_t marlin_fused_parent_bytes(std::int32_t parent_rows, std::int32_t columns) noexcept {
    static const bool adopt_opt_in = [] {
        const char* env = std::getenv("SUROGATE_SERVE_MARLIN_FP8");
        return env != nullptr && env[0] == '1';
    }();
    if (!adopt_opt_in || parent_rows <= 0 || columns <= 0) { return 0; }
    return static_cast<std::size_t>(parent_rows) * static_cast<std::size_t>(columns) * 2;
}

bool marlin_fp8_maybe_adopt(const Weight& weight, cudaStream_t stream) {
    static const bool adopt_opt_in = [] {
        const char* env = std::getenv("SUROGATE_SERVE_MARLIN_FP8");
        return env != nullptr && env[0] == '1';
    }();
    if (!adopt_opt_in) { return false; }
    if (weight.layout == QuantLayout::MarlinTiles) { return true; }
    // Adoption changes which kernels an op runs, so it changes a captured
    // graph's topology. Adopting a weight after any graph exists makes the next
    // exec update fail with cudaErrorGraphExecUpdateFailure — observed exactly
    // that way. The scratch freeze marks the point where capture has begun; past
    // it, a weight that has not already adopted keeps its e4m3 residency for the
    // life of the process, which is correct and stable if slower.
    if (g_adoption_closed || g_scratch_frozen) { return false; }
    cudaStreamCaptureStatus status = cudaStreamCaptureStatusNone;
    (void)cudaStreamIsCapturing(stream, &status);
    if (status != cudaStreamCaptureStatusNone) { return false; }
    return marlin_fp8_adopt_residency(const_cast<Weight&>(weight), stream);
}

bool marlin_fp8_adopt_residency(Weight& weight, cudaStream_t stream) {
    if (!g_enabled || weight.qtype != QType::FP8_E4M3FN_ROW_BF16S ||
        weight.layout != QuantLayout::RowScale || weight.scale_dtype != DType::BF16 ||
        weight.qdata == nullptr || weight.scales == nullptr || (weight.k % 64) != 0 ||
        (weight.n % 64) != 0) {
        return false;
    }
    const int n = weight.n;
    const int k = weight.k;
    const std::size_t b_bytes    = marlin_b_out_words(n, k) * sizeof(std::uint32_t);
    const std::size_t s_bytes    = static_cast<std::size_t>(n) * 2;
    const std::size_t gptq_bytes = static_cast<std::size_t>(k / 4) * n * sizeof(std::uint32_t);
    // In-place only when the packed form is size-exact against the residency;
    // decline rather than corrupt.
    if (b_bytes != static_cast<std::size_t>(n) * k) { return false; }

    std::size_t free_bytes = 0, total_bytes = 0;
    if (cudaMemGetInfo(&free_bytes, &total_bytes) != cudaSuccess ||
        free_bytes < 2 * (b_bytes + s_bytes + gptq_bytes)) {
        return false;
    }

    void* b_tmp    = nullptr;
    void* s_tmp    = nullptr;
    void* gptq_tmp = nullptr;
    if (cudaMalloc(&b_tmp, b_bytes) != cudaSuccess) { return false; }
    if (cudaMalloc(&s_tmp, s_bytes) != cudaSuccess) {
        cudaFree(b_tmp);
        return false;
    }
    if (cudaMalloc(&gptq_tmp, gptq_bytes) != cudaSuccess) {
        cudaFree(b_tmp);
        cudaFree(s_tmp);
        return false;
    }
    marlin_repack_fp8_row(weight.qdata, weight.scales, n, k, gptq_tmp, b_tmp, s_tmp, stream);
    CUDA_CHECK(cudaMemcpyAsync(const_cast<void*>(weight.qdata), b_tmp, b_bytes,
                               cudaMemcpyDeviceToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(const_cast<void*>(weight.scales), s_tmp, s_bytes,
                               cudaMemcpyDeviceToDevice, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    cudaFree(gptq_tmp);
    cudaFree(b_tmp);
    cudaFree(s_tmp);
    weight.layout = QuantLayout::MarlinTiles;
    // Prime the scratch now, while allocation is still legal, so the first
    // in-band call after adoption cannot decline for want of it.
    (void)ensure_scratch(static_cast<std::size_t>(n) * marlin_fixed_m() * 2,
                         static_cast<std::size_t>(k) * marlin_fixed_m() * 2, stream);
    return true;
}

MarlinScratch marlin_fp8_scratch_for(const Weight& weight, cudaStream_t stream) {
    const MarlinPlane plane = marlin_fp8_plane_for(weight, stream);
    if (plane.b_packed == nullptr) { return {}; }
    return marlin_scratch();
}

// Any-T FP8 Marlin. Below the band the call goes through the padded scratch, so
// the launch geometry is constant and a captured graph stays valid. Above it,
// A and C are passed straight through: x is [k, t] contiguous, which is [t, k]
// row-major — exactly Marlin's A layout — and out is [n, t], which is [t, n]
// row-major, exactly its C. No padding means no fixed geometry, which is fine
// because wide calls are prefill and prefill is not captured at a fixed width.
//
// This matters beyond speed: replacing a weight's residency with Marlin's
// layout only becomes safe once EVERY T can be served from it, because the
// original bytes are gone and there is no kernel left to fall back to.
bool marlin_fp8_run(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    const std::int32_t t = x.ne[1];
    if (t < 1) { return false; }
    // An adopted weight (PATCHES.md #60) always takes the exact-M path. Padding
    // M up to the band exists so that a captured decode graph sees a constant
    // launch geometry, but a decode graph is captured per batch size, so the
    // exact t is already constant within each one — and the padded form was
    // never validated for FP8: the bench measures exact M, and enabling the
    // padded path on the 27B produced a correct first token followed by
    // degenerate decode. Exact M also skips the A copy.
    const bool adopted = weight.layout == QuantLayout::MarlinTiles;
    if (adopted || t > marlin_fixed_m()) {
        if (out.ne[0] != weight.n || out.ne[1] != t) { return false; }
        if (!x.is_contiguous() || x.ne[0] != weight.k || !out.is_contiguous()) { return false; }
        const MarlinPlane wide = marlin_fp8_plane_for(weight, stream);
        if (wide.b_packed == nullptr) { return false; }
        const MarlinScratch wide_scratch = marlin_scratch();
        if (wide_scratch.c_tmp == nullptr || wide_scratch.locks == nullptr) { return false; }
        marlin_gemm_bf16(x.data, wide.b_packed, wide.scales, out.data, wide_scratch.c_tmp,
                         wide_scratch.locks, t, weight.n, weight.k, /*group_size=*/-1,
                         /*b_is_fp8=*/true, wide_scratch.sm_count, stream);
        return true;
    }
    if (out.ne[0] != weight.n || out.ne[1] != t) { return false; }
    // The padded-A copy assumes the [k, t] block is contiguous, which is what
    // makes it [t, k] row-major for Marlin. A strided view would copy the
    // wrong bytes silently, so decline instead.
    if (!x.is_contiguous() || x.ne[0] != weight.k) { return false; }
    const MarlinPlane plane = marlin_fp8_plane_for(weight, stream);
    if (plane.b_packed == nullptr) { return false; }
    const MarlinScratch scratch = marlin_scratch();
    if (scratch.gemm_out == nullptr || scratch.a_pad == nullptr ||
        static_cast<std::size_t>(weight.n) * marlin_fixed_m() * 2 > g_scratch_out_bytes ||
        static_cast<std::size_t>(weight.k) * marlin_fixed_m() * 2 > g_scratch_a_bytes) {
        return false;
    }
    const std::size_t a_used = static_cast<std::size_t>(weight.k) * t * 2;
    if (cudaMemcpyAsync(scratch.a_pad, x.data, a_used, cudaMemcpyDeviceToDevice, stream) !=
        cudaSuccess) {
        return false;
    }
    marlin_gemm_bf16(scratch.a_pad, plane.b_packed, plane.scales, scratch.gemm_out, scratch.c_tmp,
                     scratch.locks, marlin_fixed_m(), weight.n, weight.k, /*group_size=*/-1,
                     /*b_is_fp8=*/true, scratch.sm_count, stream);
    if (out.data != scratch.gemm_out) {
        // The copy-out is contiguous [t, n]; a strided destination would take
        // the wrong bytes without erroring.
        if (!out.is_contiguous()) { return false; }
        const std::size_t c_used = static_cast<std::size_t>(weight.n) * t * 2;
        CUDA_CHECK(cudaMemcpyAsync(out.data, scratch.gemm_out, c_used, cudaMemcpyDeviceToDevice,
                                   stream));
    }
    return true;
}

bool marlin_w8_run(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    const std::int32_t t = x.ne[1];
    if (t < 1 || t > marlin_fixed_m()) { return false; }
    if (out.ne[0] != weight.n || out.ne[1] != t) { return false; }
    // The padded-A copy assumes the [k, t] block is contiguous, which is what
    // makes it [t, k] row-major for Marlin. A strided view would copy the
    // wrong bytes silently, so decline instead.
    if (!x.is_contiguous() || x.ne[0] != weight.k) { return false; }
    const MarlinPlane plane = marlin_plane_for(weight, stream);
    if (plane.b_packed == nullptr) { return false; }
    const MarlinScratch scratch = marlin_scratch();
    if (scratch.gemm_out == nullptr || scratch.a_pad == nullptr ||
        static_cast<std::size_t>(weight.n) * marlin_fixed_m() * 2 > g_scratch_out_bytes ||
        static_cast<std::size_t>(weight.k) * marlin_fixed_m() * 2 > g_scratch_a_bytes) {
        return false;
    }

    // The activation block is [k, t] column-major, i.e. [t, k] row-major and
    // contiguous, so the round's rows copy straight into the head of the
    // zero-padded A. The pad rows below stay zero for the life of the
    // scratch.
    const std::size_t a_used = static_cast<std::size_t>(weight.k) * t * 2;
    if (cudaMemcpyAsync(scratch.a_pad, x.data, a_used, cudaMemcpyDeviceToDevice, stream) !=
        cudaSuccess) {
        return false;
    }
    marlin_gemm_bf16(scratch.a_pad, plane.b_packed, plane.scales, scratch.gemm_out,
                     scratch.c_tmp, scratch.locks, marlin_fixed_m(), weight.n, weight.k,
                     /*group_size=*/32, /*b_is_fp8=*/false, scratch.sm_count, stream);
    // C is [marlin_fixed_m(), n] row-major, so its first t rows are exactly the
    // caller's [n, t] result; a caller writing into the scratch itself reads
    // them in place.
    if (out.data != scratch.gemm_out) {
        // The copy-out is contiguous [t, n]; a strided destination would take
        // the wrong bytes without erroring.
        if (!out.is_contiguous()) { return false; }
        const std::size_t c_used = static_cast<std::size_t>(weight.n) * t * 2;
        CUDA_CHECK(cudaMemcpyAsync(out.data, scratch.gemm_out, c_used, cudaMemcpyDeviceToDevice,
                                   stream));
    }
    return true;
}

} // namespace sinfer::ops::detail
