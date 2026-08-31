// The resident adapter store (api/ops/lora_store.h).

#include "api/ops/lora_store.h"

#include "core/device.h"

#include <cstring>
#include <map>
#include <mutex>
#include <stdexcept>

namespace sinfer::ops {
namespace {

Weight bf16_weight(const void* data, std::int32_t n, std::int32_t k) {
    Weight w;
    w.qtype           = QType::BF16_CTRL;
    w.payload         = data;
    w.qdata           = data;
    w.payload_bytes   = static_cast<std::uint64_t>(n) * k * sizeof(std::uint16_t);
    w.n               = n;
    w.k               = k;
    w.ndim            = 2;
    w.shape[0]        = n;
    w.shape[1]        = k;
    w.padded_shape[0] = n;
    w.padded_shape[1] = k;
    w.layout          = QuantLayout::Contiguous;
    return w;
}

float bf16_to_float(std::uint16_t bits) {
    const std::uint32_t word = static_cast<std::uint32_t>(bits) << 16U;
    float value              = 0.0F;
    std::memcpy(&value, &word, sizeof(value));
    return value;
}

std::uint16_t float_to_bf16(float value) {
    std::uint32_t word = 0;
    std::memcpy(&word, &value, sizeof(word));
    word += 0x7FFFU + ((word >> 16U) & 1U);
    return static_cast<std::uint16_t>(word >> 16U);
}

void* upload(const std::vector<std::uint16_t>& host) {
    void* device = nullptr;
    const std::size_t bytes = host.size() * sizeof(std::uint16_t);
    CUDA_CHECK(cudaMalloc(&device, bytes));
    CUDA_CHECK(cudaMemcpy(device, host.data(), bytes, cudaMemcpyHostToDevice));
    return device;
}

} // namespace

LoraStore::~LoraStore() {
    for (void* pointer : owned_) { cudaFree(pointer); }
}

void LoraStore::add(const void* base_key, const std::vector<std::uint16_t>& a,
                    const std::vector<std::uint16_t>& b, std::int32_t rank, std::int32_t in_dim,
                    std::int32_t out_dim, float scale) {
    if (base_key == nullptr || rank <= 0 || in_dim <= 0 || out_dim <= 0) {
        throw std::invalid_argument("lora_store: adapter geometry must be positive");
    }
    if (a.size() != static_cast<std::size_t>(rank) * in_dim ||
        b.size() != static_cast<std::size_t>(out_dim) * rank) {
        throw std::invalid_argument("lora_store: A must be [rank,in] and B [out,rank]");
    }
    // alpha/r rides on A, so the runtime path is two plain GEMMs and an add.
    std::vector<std::uint16_t> scaled(a.size());
    for (std::size_t i = 0; i < a.size(); ++i) {
        scaled[i] = float_to_bf16(bf16_to_float(a[i]) * scale);
    }

    void* a_device = upload(scaled);
    owned_.push_back(a_device);
    void* b_device = upload(b);
    owned_.push_back(b_device);

    LoraWeights entry;
    entry.a    = bf16_weight(a_device, rank, in_dim);
    entry.b    = bf16_weight(b_device, out_dim, rank);
    entry.rank = rank;
    entries_.emplace(base_key, entry);
    if (rank > max_rank_) { max_rank_ = rank; }
    if (out_dim > max_out_dim_) { max_out_dim_ = out_dim; }
}

LoraStore& lora_store_for_current_device() {
    static std::mutex mutex;
    static std::map<int, LoraStore> stores;
    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    const std::lock_guard<std::mutex> lock(mutex);
    return stores[device];
}

bool lora_active() {
    // Read once per process rather than per projection: the store is populated at
    // load and never changes afterwards, and this sits in front of every hook.
    static const bool active = !lora_store_for_current_device().empty();
    return active;
}

} // namespace sinfer::ops
