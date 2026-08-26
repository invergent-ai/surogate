// Derived Marlin residency registry (see marlin_plane.h for the scheme).

#include "ops/linear/marlin/marlin_plane.h"

#include "ops/linear/marlin/marlin_gemm.h"
#include "ops/linear/marlin/marlin_repack.h"

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

MarlinScratch g_scratch;
std::size_t g_scratch_out_bytes = 0;
bool g_scratch_frozen           = false;

bool ensure_scratch(std::size_t out_bytes, cudaStream_t stream) {
    if (g_scratch.gemm_out != nullptr && out_bytes <= g_scratch_out_bytes) { return true; }
    // The scratch grows only until the engine freezes it after capture: from
    // then on its addresses live inside captured graphs and must not move.
    if (g_scratch_frozen) { return false; }
    if (g_scratch.gemm_out != nullptr) {
        cudaFree(g_scratch.gemm_out);
        g_bytes -= g_scratch_out_bytes;
        g_scratch.gemm_out  = nullptr;
        g_scratch_out_bytes = 0;
        void* grown = nullptr;
        if (cudaMalloc(&grown, out_bytes) != cudaSuccess) { return false; }
        g_scratch.gemm_out  = grown;
        g_scratch_out_bytes = out_bytes;
        g_bytes += out_bytes;
        return true;
    }
    int device = 0;
    cudaGetDevice(&device);
    int sms = 0;
    cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device);
    const std::size_t c_tmp_bytes =
        marlin_c_tmp_floats(sms, kMarlinMaxBandTokens) * sizeof(float);
    const std::size_t lock_bytes = marlin_workspace_locks_count(sms) * sizeof(int);
    void* out_buf   = nullptr;
    void* c_tmp_buf = nullptr;
    void* lock_buf  = nullptr;
    if (cudaMalloc(&out_buf, out_bytes) != cudaSuccess) { return false; }
    if (cudaMalloc(&c_tmp_buf, c_tmp_bytes) != cudaSuccess) {
        cudaFree(out_buf);
        return false;
    }
    if (cudaMalloc(&lock_buf, lock_bytes) != cudaSuccess) {
        cudaFree(out_buf);
        cudaFree(c_tmp_buf);
        return false;
    }
    cudaMemsetAsync(lock_buf, 0, lock_bytes, stream);
    g_scratch = MarlinScratch{out_buf, c_tmp_buf, static_cast<int*>(lock_buf), sms};
    g_scratch_out_bytes = out_bytes;
    g_bytes += out_bytes + c_tmp_bytes + lock_bytes;
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
        free_bytes < 2 * (b_bytes + s_bytes + gptq_bytes)) {
        return {};
    }
    if (!ensure_scratch(static_cast<std::size_t>(n) * kMarlinMaxBandTokens * 2, stream)) {
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

bool marlin_w8_run(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    const std::int32_t t = x.ne[1];
    if (t < 1 || t > kMarlinMaxBandTokens) { return false; }
    if (out.ne[0] != weight.n || out.ne[1] != t) { return false; }
    const MarlinPlane plane = marlin_plane_for(weight, stream);
    if (plane.b_packed == nullptr) { return false; }
    const MarlinScratch scratch = marlin_scratch();
    if (scratch.gemm_out == nullptr ||
        static_cast<std::size_t>(weight.n) * t * 2 > g_scratch_out_bytes) {
        return false;
    }
    marlin_gemm_bf16(x.data, plane.b_packed, plane.scales, out.data, scratch.c_tmp,
                     scratch.locks, t, weight.n, weight.k, /*group_size=*/32,
                     /*b_is_fp8=*/false, scratch.sm_count, stream);
    return true;
}

} // namespace ninfer::ops::detail
