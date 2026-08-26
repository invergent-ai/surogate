// Derived Marlin residency registry (see marlin_plane.h for the scheme).

#include "ops/linear/marlin/marlin_plane.h"

#include "ops/linear/marlin/marlin_gemm.h"
#include "ops/linear/marlin/marlin_repack.h"

#include "core/device.h"

#include <cstdlib>
#include <map>
#include <mutex>

namespace ninfer::ops::detail {
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

MarlinScratch g_scratch;
std::size_t g_scratch_out_bytes = 0;
std::size_t g_scratch_a_bytes   = 0;
bool g_scratch_frozen           = false;

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

int g_fixed_m = 32;
void marlin_set_fixed_m(int lanes) noexcept {
    // Pinned to 32 pending a fix. A 64-wide band measured faster over 40 s
    // (0.8B 6,003, 4B 2,633) but BOTH models fail under 90 s of sustained
    // load — the 0.8B with an invalid-UTF-8 fatal, the 4B with collapsing
    // throughput and truncated streams — while the same 90 s load is clean at
    // 32 lanes with Marlin, and clean at 64 lanes with Marlin off
    // (SUROGATE_SERVE_MARLIN=0, 5,429 tok/s, zero errors). So the fault is
    // Marlin at wide batches, not the lane count. At 32 the band only serves
    // rounds it is proven on and wider rounds take the engine's own kernels.
    // SUROGATE_SERVE_MARLIN_WIDE=1 restores the 64-wide band for debugging.
    const char* wide = std::getenv("SUROGATE_SERVE_MARLIN_WIDE");
    const bool allow_wide = wide != nullptr && wide[0] == '1';
    const int rounded = (!allow_wide || lanes <= 32) ? 32
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
    if (!fp8_opt_in || !g_enabled || weight.qtype != QType::FP8_E4M3FN_ROW_BF16S ||
        weight.layout != QuantLayout::RowScale || weight.scale_dtype != DType::BF16 ||
        weight.qdata == nullptr || weight.scales == nullptr || (weight.k % 64) != 0 ||
        (weight.n % 64) != 0) {
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
    const std::size_t b_bytes    = marlin_b_out_words(n, k) * sizeof(std::uint32_t);
    const std::size_t s_bytes    = static_cast<std::size_t>(n) * 2;
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

MarlinScratch marlin_fp8_scratch_for(const Weight& weight, cudaStream_t stream) {
    const MarlinPlane plane = marlin_fp8_plane_for(weight, stream);
    if (plane.b_packed == nullptr) { return {}; }
    return marlin_scratch();
}

bool marlin_fp8_run(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    const std::int32_t t = x.ne[1];
    if (t < 1 || t > marlin_fixed_m()) { return false; }
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

} // namespace ninfer::ops::detail
