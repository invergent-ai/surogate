#include "family/vision_fp8.h"

#include "core/device.h"

#include <cstdio>
#include <map>
#include <mutex>

namespace sinfer::family {
namespace {

std::mutex& registry_mutex() {
    static std::mutex mutex;
    return mutex;
}

std::map<const VisionWeights*, std::weak_ptr<const VisionFp8Weights>>& registry() {
    static std::map<const VisionWeights*, std::weak_ptr<const VisionFp8Weights>> entries;
    return entries;
}

constexpr std::size_t kAlignment = 256;

std::size_t aligned(std::size_t bytes) { return (bytes + kAlignment - 1) / kAlignment * kAlignment; }

bool device_bf16(const Weight& weight) {
    if (weight.qtype != QType::BF16_CTRL || weight.layout != QuantLayout::Contiguous ||
        weight.qdata == nullptr || weight.n <= 0 || weight.k <= 0 || (weight.k % 8) != 0) {
        return false;
    }
    cudaPointerAttributes attributes{};
    if (cudaPointerGetAttributes(&attributes, weight.qdata) != cudaSuccess) {
        (void)cudaGetLastError();
        return false;
    }
    return attributes.type == cudaMemoryTypeDevice || attributes.type == cudaMemoryTypeManaged;
}

void skip(const char* why) {
    std::fprintf(stderr, "engine: --vision-fp8 has no effect: %s; the vision tower stays BF16\n", why);
}

} // namespace

VisionFp8Weights::~VisionFp8Weights() {
    if (storage == nullptr) { return; }
    int current = 0;
    if (cudaGetDevice(&current) == cudaSuccess && current != device) { (void)cudaSetDevice(device); }
    (void)cudaFree(storage);
    if (current != device) { (void)cudaSetDevice(current); }
}

std::shared_ptr<const VisionFp8Weights>
prepare_vision_fp8(const VisionWeights& vision, const VisionGeometry& geometry, cudaStream_t stream) {
    using ops::detail::Mxfp8Matrix;
    if (geometry.gemma_version || geometry.muse_glimmer || vision.common.layers.empty()) {
        skip("this model's vision tower has no MXFP8 route");
        return nullptr;
    }
    if (!ops::detail::mxfp8_device_supported()) {
        skip("MXFP8 needs compute capability 10.0 or newer (Blackwell)");
        return nullptr;
    }
    for (const auto& layer : vision.common.layers) {
        if (!device_bf16(layer.qkv) || !device_bf16(layer.output) || !device_bf16(layer.fc1) ||
            !device_bf16(layer.fc2)) {
            skip("the block projections are not device-resident BF16 (a quantized tower, or --offload-vision)");
            return nullptr;
        }
    }
    // The projections' shapes must have an MXFP8 algorithm at a representative width, with the
    // epilogues the tower uses; a library or device without them keeps the BF16 route.
    const auto& first = vision.common.layers.front();
    if (!ops::detail::mxfp8_gemm_available(first.qkv.n, first.qkv.k, 1024, true, false) ||
        !ops::detail::mxfp8_gemm_available(first.output.n, first.output.k, 1024, true, true) ||
        !ops::detail::mxfp8_gemm_available(first.fc1.n, first.fc1.k, 1024, true, false) ||
        !ops::detail::mxfp8_gemm_available(first.fc2.n, first.fc2.k, 1024, true, true)) {
        skip("cuBLASLt offers no MXFP8 algorithm for the tower's projections");
        return nullptr;
    }

    auto out = std::make_shared<VisionFp8Weights>();
    CUDA_CHECK(cudaGetDevice(&out->device));
    const auto size = [](const Weight& w) {
        return aligned(ops::detail::mxfp8_codes_bytes(w.n, w.k)) +
               aligned(ops::detail::mxfp8_scales_bytes(w.n, w.k));
    };
    for (const auto& layer : vision.common.layers) {
        out->bytes += size(layer.qkv) + size(layer.output) + size(layer.fc1) + size(layer.fc2);
    }
    if (const cudaError_t status = cudaMalloc(&out->storage, out->bytes); status != cudaSuccess) {
        (void)cudaGetLastError();
        out->storage = nullptr;
        skip("the MXFP8 copies do not fit in device memory");
        return nullptr;
    }
    // The weights' uploads may still be in flight on the loader's own streams when the program
    // starts (routed_nvfp4.h waits for the same reason); quantizing before they land would copy
    // whatever the buffers held.
    CUDA_CHECK(cudaDeviceSynchronize());
    auto* cursor        = static_cast<unsigned char*>(out->storage);
    const auto quantize = [&](const Weight& w) {
        Mxfp8Matrix m;
        m.rows   = w.n;
        m.k      = w.k;
        m.codes  = cursor;
        cursor  += aligned(ops::detail::mxfp8_codes_bytes(w.n, w.k));
        m.scales = cursor;
        cursor  += aligned(ops::detail::mxfp8_scales_bytes(w.n, w.k));
        ops::detail::mxfp8_quantize(w.qdata, w.k, ops::detail::Mxfp8Transform::None, m, stream);
        return m;
    };
    out->blocks.reserve(vision.common.layers.size());
    for (const auto& layer : vision.common.layers) {
        out->blocks.push_back(VisionFp8Block{quantize(layer.qkv), quantize(layer.output),
                                             quantize(layer.fc1), quantize(layer.fc2)});
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
    std::fprintf(stderr,
                 "engine: vision tower block projections run MXFP8 (--vision-fp8): %zu blocks, "
                 "%.1f MiB of block-scaled FP8 weights beside the BF16 ones\n",
                 out->blocks.size(), static_cast<double>(out->bytes) / (1024.0 * 1024.0));
    const std::lock_guard<std::mutex> lock(registry_mutex());
    registry()[&vision] = out;
    return out;
}

std::shared_ptr<const VisionFp8Weights> find_vision_fp8(const VisionWeights* vision) {
    const std::lock_guard<std::mutex> lock(registry_mutex());
    const auto found = registry().find(vision);
    if (found == registry().end()) { return nullptr; }
    auto held = found->second.lock();
    if (!held) { registry().erase(found); }
    return held;
}

} // namespace sinfer::family
