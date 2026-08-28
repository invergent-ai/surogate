// Expert slot cache: device pool + directory + per-round resolve/gather (api/ops/expert_slot_cache.h).
//
// Resolve is one block: hits are stamped in parallel, misses are assigned by one thread walking
// a clock hand over the slots (second-chance LRU: a slot whose last touch is older than the
// hand's sweep is a victim unless it is active in this round). Gather is FreeToken's
// multi-bank index copy: every 16-byte unit of every missing expert row, across the four planes
// (gate/up codes + scales, down codes + scales), from pinned host memory into the pool, the row
// count read from a device word so the launch captures into a graph.

#include "api/ops/expert_slot_cache.h"
#include "core/device.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace ninfer::ops {
namespace {

constexpr std::size_t kAlign        = 256;
constexpr int kW8Group              = 32;
constexpr int kResolveThreads       = 1024;
constexpr int kGatherThreads        = 256;
constexpr int kGatherBlocksPerBank  = 64;

[[nodiscard]] std::size_t align_up(std::size_t value) {
    return (value + kAlign - 1) / kAlign * kAlign;
}

/// W8G32 row-split planes for an [rows, k] matrix: codes then scales, both row-major.
struct W8Planes {
    std::size_t codes_bytes_per_row  = 0;
    std::size_t scales_bytes_per_row = 0;
    std::size_t codes_plane_bytes    = 0;
    std::size_t scales_plane_bytes   = 0;
};

[[nodiscard]] W8Planes w8_planes(std::int64_t rows, std::int32_t k) {
    if (k <= 0 || k % kW8Group != 0) {
        throw std::invalid_argument("expert_slot_cache: k must be a positive multiple of 32");
    }
    W8Planes out;
    out.codes_bytes_per_row  = static_cast<std::size_t>(k);
    out.scales_bytes_per_row = static_cast<std::size_t>(k / kW8Group) * 2;
    out.codes_plane_bytes    = out.codes_bytes_per_row * static_cast<std::size_t>(rows);
    out.scales_plane_bytes   = out.scales_bytes_per_row * static_cast<std::size_t>(rows);
    return out;
}

[[nodiscard]] Weight pool_weight(std::byte* codes, std::byte* scales, std::int32_t rows,
                                 std::int32_t k, const W8Planes& planes) {
    Weight out{};
    out.payload          = codes;
    out.payload_bytes    = planes.codes_plane_bytes + planes.scales_plane_bytes;
    out.high_plane_bytes = 0;
    out.qtype            = QType::W8G32_F16S;
    out.layout           = QuantLayout::RowSplit;
    out.group_size       = kW8Group;
    out.qdata            = codes;
    out.qhigh            = nullptr;
    out.scales           = scales;
    out.n                = rows;
    out.k                = k;
    out.group            = kW8Group;
    out.scale_dtype      = DType::FP16;
    out.ndim             = 2;
    out.shape[0]         = rows;
    out.shape[1]         = k;
    out.padded_shape[0]  = rows;
    out.padded_shape[1]  = k;
    return out;
}

void require_geometry(const SparseMoeGeometry& geometry) {
    if (geometry.hidden <= 0 || geometry.experts <= 0 || geometry.intermediate <= 0 ||
        geometry.experts_per_token <= 0) {
        throw std::invalid_argument("expert_slot_cache: invalid sparse MoE geometry");
    }
}

// ---------------------------------------------------------------------------------------------
// Kernels
// ---------------------------------------------------------------------------------------------

__global__ void directory_reset_kernel(int* __restrict__ slot_of_expert, int ids,
                                       int* __restrict__ expert_of_slot,
                                       unsigned* __restrict__ last_used,
                                       unsigned* __restrict__ active_round, int slots,
                                       unsigned* __restrict__ round, int* __restrict__ hand) {
    const int i = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) +
                  static_cast<int>(threadIdx.x);
    if (i < ids) { slot_of_expert[i] = -1; }
    if (i < slots) {
        expert_of_slot[i] = -1;
        last_used[i]      = 0U;
        active_round[i]   = 0U;
    }
    if (i == 0) {
        *round = 0U;
        *hand  = 0;
    }
}

/// One block. `ids` holds `count` expert ids of one layer (duplicates allowed). `seen` is a
/// per-layer scratch of `experts` stamps that dedupes the round's ids.
__global__ void __launch_bounds__(kResolveThreads)
    resolve_kernel(const int* __restrict__ ids, const float* __restrict__ alpha, int count,
                   int layer, int experts, int slots, int* __restrict__ slot_of_expert,
                   int* __restrict__ expert_of_slot, unsigned* __restrict__ last_used,
                   unsigned* __restrict__ active_round, unsigned* __restrict__ round,
                   int* __restrict__ hand, unsigned* __restrict__ seen,
                   unsigned* __restrict__ cpu_round, int* __restrict__ miss_slots,
                   int* __restrict__ miss_experts, long long* __restrict__ miss_count,
                   int miss_capacity, unsigned cpu_share_q16, int per_token,
                   int* __restrict__ cpu_tokens,
                   int* __restrict__ cpu_experts, float* __restrict__ cpu_weights,
                   long long* __restrict__ cpu_count, int cpu_capacity) {
    __shared__ unsigned s_round;
    __shared__ int s_pending;
    __shared__ int s_pending_experts[1024];
    __shared__ int s_cpu_pending;
    const int tid = static_cast<int>(threadIdx.x);
    if (tid == 0) {
        s_round       = *round + 1U;
        *round        = s_round;
        s_pending     = 0;
        s_cpu_pending = 0;
        *miss_count   = 0;
        if (cpu_count != nullptr) { *cpu_count = 0; }
    }
    __syncthreads();
    const unsigned this_round = s_round;
    const int* layer_table     = slot_of_expert + static_cast<std::int64_t>(layer) * experts;

    // Pass 1: dedupe and stamp hits; collect the distinct misses.
    for (int i = tid; i < count; i += kResolveThreads) {
        const int expert = ids[i];
        if (expert < 0 || expert >= experts) { continue; }
        if (atomicExch(&seen[expert], this_round) == this_round) { continue; } // duplicate
        const int slot = layer_table[expert];
        if (slot >= 0) {
            last_used[slot]    = this_round;
            active_round[slot] = this_round;
        } else {
            const int at = atomicAdd(&s_pending, 1);
            if (at < 1024) { s_pending_experts[at] = expert; }
        }
    }
    __syncthreads();

    // Pass 2: one thread assigns victims with a clock hand; the order is deterministic. With a
    // CPU split, a `cpu_share` fraction of the misses is handed to the host instead: those
    // experts keep -1 in the table and are stamped in `cpu_round` for pass 3.
    if (tid == 0) {
        const int pending = s_pending < 1024 ? s_pending : 1024;
        int h             = *hand;
        int written       = 0;
        int sent_to_cpu   = 0;
        for (int p = 0; p < pending; ++p) {
            const int expert = s_pending_experts[p];
            if (cpu_share_q16 > 0U && cpu_count != nullptr &&
                static_cast<unsigned long long>(sent_to_cpu) * 65536ULL <
                    static_cast<unsigned long long>(p + 1) * cpu_share_q16) {
                cpu_round[expert] = this_round;
                ++sent_to_cpu;
                continue;
            }
            // Find a slot that is not active in this round; prefer the least recently used
            // among the next few candidates the hand passes (bounded sweep keeps it cheap).
            int victim        = -1;
            unsigned best_age = 0U;
            for (int probe = 0; probe < slots; ++probe) {
                const int s = (h + probe) % slots;
                if (active_round[s] == this_round) { continue; }
                const unsigned age = this_round - last_used[s];
                if (victim < 0 || age > best_age) {
                    victim   = s;
                    best_age = age;
                    if (expert_of_slot[s] < 0 || age >= 64U) { break; } // empty or stale enough
                }
            }
            if (victim < 0) { break; } // every slot is active in this round: cannot serve
            h = (victim + 1) % slots;
            const int old_flat = expert_of_slot[victim];
            if (old_flat >= 0) { slot_of_expert[old_flat] = -1; }
            const int flat        = layer * experts + expert;
            slot_of_expert[flat]  = victim;
            expert_of_slot[victim] = flat;
            last_used[victim]      = this_round;
            active_round[victim]   = this_round;
            if (written < miss_capacity) {
                miss_slots[written]   = victim;
                miss_experts[written] = expert;
            }
            ++written;
        }
        *hand         = h;
        *miss_count   = written < miss_capacity ? written : miss_capacity;
        s_cpu_pending = sent_to_cpu;
    }
    __syncthreads();

    // Pass 3: every (token, path) routed to a CPU expert becomes a host job with its weight.
    if (s_cpu_pending > 0 && cpu_count != nullptr) {
        for (int i = tid; i < count; i += kResolveThreads) {
            const int expert = ids[i];
            if (expert < 0 || expert >= experts || cpu_round[expert] != this_round) { continue; }
            const long long at = atomicAdd(reinterpret_cast<unsigned long long*>(cpu_count), 1ULL);
            if (at < cpu_capacity) {
                cpu_tokens[at]  = i / per_token;
                cpu_experts[at] = expert;
                cpu_weights[at] = alpha != nullptr ? alpha[i] : 1.0f;
            }
        }
    }
}

struct GatherBank {
    const std::byte* src = nullptr; // host plane
    std::byte* dst       = nullptr; // pool plane
    std::uint64_t bytes  = 0;       // per expert
};

struct GatherParams {
    GatherBank banks[4];
    const int* miss_slots;
    const int* miss_experts;
    const long long* miss_count;
};

__global__ void __launch_bounds__(kGatherThreads)
    gather_kernel(const __grid_constant__ GatherParams p) {
    const int bank = static_cast<int>(blockIdx.x) / kGatherBlocksPerBank;
    const int blk  = static_cast<int>(blockIdx.x) % kGatherBlocksPerBank;
    const GatherBank b = p.banks[bank];
    const long long n  = *p.miss_count;
    const std::uint64_t units = b.bytes >> 4;
    const std::uint64_t total = static_cast<std::uint64_t>(n) * units;
    const std::uint64_t stride =
        static_cast<std::uint64_t>(kGatherBlocksPerBank) * kGatherThreads;
    for (std::uint64_t u = static_cast<std::uint64_t>(blk) * kGatherThreads + threadIdx.x;
         u < total; u += stride) {
        const std::uint64_t row = u / units;
        const std::uint64_t col = (u - row * units) << 4;
        const std::uint64_t src_row = static_cast<std::uint64_t>(p.miss_experts[row]);
        const std::uint64_t dst_row = static_cast<std::uint64_t>(p.miss_slots[row]);
        const uint4 value = __ldcs(reinterpret_cast<const uint4*>(b.src + src_row * b.bytes + col));
        __stcs(reinterpret_cast<uint4*>(b.dst + dst_row * b.bytes + col), value);
    }
}

__global__ void partial_add_kernel(const float* __restrict__ partial, __nv_bfloat16* __restrict__ destination,
                                   long long count) {
    const long long i = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= count) { return; }
    destination[i] = __float2bfloat16_rn(__bfloat162float(destination[i]) + partial[i]);
}

} // namespace

void expert_cpu_partial_add(const void* partial_device_ptr, Tensor& destination, cudaStream_t stream) {
    if (partial_device_ptr == nullptr || destination.data == nullptr || destination.dtype != DType::BF16) {
        throw std::invalid_argument("expert_slot_cache: partial add needs a BF16 destination");
    }
    const long long count = destination.numel();
    const int grid        = static_cast<int>((count + 255) / 256);
    partial_add_kernel<<<grid, 256, 0, stream>>>(static_cast<const float*>(partial_device_ptr),
                                                 static_cast<__nv_bfloat16*>(destination.data), count);
    CUDA_CHECK(cudaGetLastError());
}

// ---------------------------------------------------------------------------------------------
// Sizes and construction
// ---------------------------------------------------------------------------------------------

std::size_t expert_slot_pool_bytes(const SparseMoeGeometry& geometry, std::int32_t slots) {
    require_geometry(geometry);
    if (slots <= 0) { throw std::invalid_argument("expert_slot_cache: slots must be positive"); }
    const std::int64_t gate_rows = static_cast<std::int64_t>(slots) * geometry.expert_rows();
    const std::int64_t down_rows = static_cast<std::int64_t>(slots) * geometry.hidden;
    const W8Planes gate = w8_planes(gate_rows, geometry.hidden);
    const W8Planes down = w8_planes(down_rows, geometry.intermediate);
    return align_up(gate.codes_plane_bytes) + align_up(gate.scales_plane_bytes) +
           align_up(down.codes_plane_bytes) + align_up(down.scales_plane_bytes);
}

std::size_t expert_slot_directory_bytes(std::int32_t layers, std::int32_t experts,
                                        std::int32_t slots) {
    if (layers <= 0 || experts <= 0 || slots <= 0) {
        throw std::invalid_argument("expert_slot_cache: directory dimensions must be positive");
    }
    const std::size_t ids = static_cast<std::size_t>(layers) * experts;
    return align_up(ids * sizeof(int)) + align_up(static_cast<std::size_t>(slots) * sizeof(int)) +
           2 * align_up(static_cast<std::size_t>(slots) * sizeof(unsigned)) +
           align_up(sizeof(unsigned)) + align_up(sizeof(int)) +
           2 * align_up(static_cast<std::size_t>(experts) * sizeof(unsigned));
}

std::size_t expert_miss_list_bytes(std::int32_t capacity) {
    if (capacity <= 0) { throw std::invalid_argument("expert_slot_cache: capacity must be positive"); }
    return 2 * align_up(static_cast<std::size_t>(capacity) * sizeof(int)) + align_up(sizeof(long long));
}

ExpertSlotPool create_expert_slot_pool(const SparseMoeGeometry& geometry, std::int32_t slots,
                                       void* device_bytes) {
    require_geometry(geometry);
    if (device_bytes == nullptr) {
        throw std::invalid_argument("expert_slot_cache: pool memory is null");
    }
    const std::int32_t gate_rows = slots * geometry.expert_rows();
    const std::int32_t down_rows = slots * geometry.hidden;
    const W8Planes gate = w8_planes(gate_rows, geometry.hidden);
    const W8Planes down = w8_planes(down_rows, geometry.intermediate);
    auto* base = static_cast<std::byte*>(device_bytes);
    ExpertSlotPool pool;
    pool.slots          = slots;
    pool.gate_up_codes  = base;
    base += align_up(gate.codes_plane_bytes);
    pool.gate_up_scales = base;
    base += align_up(gate.scales_plane_bytes);
    pool.down_codes     = base;
    base += align_up(down.codes_plane_bytes);
    pool.down_scales    = base;
    pool.routed_gate_up = pool_weight(pool.gate_up_codes, pool.gate_up_scales, gate_rows,
                                      geometry.hidden, gate);
    pool.routed_down    = pool_weight(pool.down_codes, pool.down_scales, down_rows,
                                      geometry.intermediate, down);
    return pool;
}

ExpertSlotDirectory create_expert_slot_directory(std::int32_t layers, std::int32_t experts,
                                                 std::int32_t slots, void* device_bytes,
                                                 cudaStream_t stream) {
    if (device_bytes == nullptr) {
        throw std::invalid_argument("expert_slot_cache: directory memory is null");
    }
    const std::int32_t ids = layers * experts;
    auto* base             = static_cast<std::byte*>(device_bytes);
    ExpertSlotDirectory out;
    out.layers  = layers;
    out.experts = experts;
    auto take = [&](DType dtype, std::int32_t count, std::size_t element) {
        Tensor t(base, dtype, {count});
        base += align_up(static_cast<std::size_t>(count) * element);
        return t;
    };
    out.slot_of_expert = take(DType::I32, ids, sizeof(int));
    out.expert_of_slot = take(DType::I32, slots, sizeof(int));
    out.last_used      = take(DType::I32, slots, sizeof(unsigned));
    out.active_round   = take(DType::I32, slots, sizeof(unsigned));
    out.round          = take(DType::I32, 1, sizeof(unsigned));
    out.hand           = take(DType::I32, 1, sizeof(int));
    out.seen           = take(DType::I32, experts, sizeof(unsigned));
    out.cpu_round      = take(DType::I32, experts, sizeof(unsigned));
    expert_slot_directory_reset(out, stream);
    return out;
}

ExpertMissList create_expert_miss_list(std::int32_t capacity, void* device_bytes) {
    if (device_bytes == nullptr) {
        throw std::invalid_argument("expert_slot_cache: miss list memory is null");
    }
    auto* base = static_cast<std::byte*>(device_bytes);
    ExpertMissList out;
    out.capacity = capacity;
    out.slots    = Tensor(base, DType::I32, {capacity});
    base += align_up(static_cast<std::size_t>(capacity) * sizeof(int));
    out.experts = Tensor(base, DType::I32, {capacity});
    base += align_up(static_cast<std::size_t>(capacity) * sizeof(int));
    out.count = Tensor(base, DType::I64, {1});
    return out;
}

std::size_t expert_cpu_job_list_bytes(std::int32_t capacity) {
    if (capacity <= 0) { throw std::invalid_argument("expert_slot_cache: cpu job capacity must be positive"); }
    return 2 * align_up(static_cast<std::size_t>(capacity) * sizeof(int)) +
           align_up(static_cast<std::size_t>(capacity) * sizeof(float)) + align_up(sizeof(long long));
}

ExpertCpuJobList create_expert_cpu_job_list(std::int32_t capacity, void* device_bytes) {
    if (device_bytes == nullptr) {
        throw std::invalid_argument("expert_slot_cache: cpu job list memory is null");
    }
    auto* base = static_cast<std::byte*>(device_bytes);
    ExpertCpuJobList out;
    out.capacity = capacity;
    out.tokens   = Tensor(base, DType::I32, {capacity});
    base += align_up(static_cast<std::size_t>(capacity) * sizeof(int));
    out.experts = Tensor(base, DType::I32, {capacity});
    base += align_up(static_cast<std::size_t>(capacity) * sizeof(int));
    out.weights = Tensor(base, DType::FP32, {capacity});
    base += align_up(static_cast<std::size_t>(capacity) * sizeof(float));
    out.count = Tensor(base, DType::I64, {1});
    return out;
}

void expert_slot_directory_reset(ExpertSlotDirectory& directory, cudaStream_t stream) {
    const int ids   = directory.layers * directory.experts;
    const int slots = static_cast<int>(directory.expert_of_slot.ne[0]);
    const int span  = ids > slots ? ids : slots;
    const int grid  = (span + 255) / 256;
    directory_reset_kernel<<<grid, 256, 0, stream>>>(
        static_cast<int*>(directory.slot_of_expert.data), ids,
        static_cast<int*>(directory.expert_of_slot.data),
        static_cast<unsigned*>(directory.last_used.data),
        static_cast<unsigned*>(directory.active_round.data), slots,
        static_cast<unsigned*>(directory.round.data), static_cast<int*>(directory.hand.data));
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemsetAsync(directory.seen.data, 0, directory.seen.bytes(), stream));
    CUDA_CHECK(cudaMemsetAsync(directory.cpu_round.data, 0, directory.cpu_round.bytes(), stream));
}

ExpertHostBank expert_host_bank(const SparseMoeGeometry& geometry, const Weight& routed_gate_up,
                                const Weight& routed_down) {
    require_geometry(geometry);
    if (routed_gate_up.qtype != QType::W8G32_F16S || routed_down.qtype != QType::W8G32_F16S ||
        routed_gate_up.layout != QuantLayout::RowSplit ||
        routed_down.layout != QuantLayout::RowSplit) {
        throw std::invalid_argument("expert_slot_cache: the host bank must hold W8 row-split experts");
    }
    if (routed_gate_up.n != geometry.routed_gate_rows() || routed_gate_up.k != geometry.hidden ||
        routed_down.n != geometry.routed_down_rows() || routed_down.k != geometry.intermediate) {
        throw std::invalid_argument("expert_slot_cache: host bank shapes do not match the geometry");
    }
    const W8Planes gate = w8_planes(geometry.expert_rows(), geometry.hidden);
    const W8Planes down = w8_planes(geometry.hidden, geometry.intermediate);
    ExpertHostBank bank;
    bank.gate_up_codes                  = static_cast<const std::byte*>(routed_gate_up.qdata);
    bank.gate_up_scales                 = static_cast<const std::byte*>(routed_gate_up.scales);
    bank.down_codes                     = static_cast<const std::byte*>(routed_down.qdata);
    bank.down_scales                    = static_cast<const std::byte*>(routed_down.scales);
    bank.gate_up_codes_bytes_per_expert  = gate.codes_plane_bytes;
    bank.gate_up_scales_bytes_per_expert = gate.scales_plane_bytes;
    bank.down_codes_bytes_per_expert     = down.codes_plane_bytes;
    bank.down_scales_bytes_per_expert    = down.scales_plane_bytes;
    return bank;
}

// ---------------------------------------------------------------------------------------------
// Per-round operations
// ---------------------------------------------------------------------------------------------

void expert_slot_resolve(const Tensor& ids, std::int32_t layer, ExpertSlotDirectory& directory,
                         ExpertMissList& misses, cudaStream_t stream) {
    expert_slot_resolve(ids, Tensor{}, layer, directory, misses, nullptr, 0U, stream);
}

void expert_slot_resolve(const Tensor& ids, const Tensor& alpha, std::int32_t layer,
                         ExpertSlotDirectory& directory, ExpertMissList& misses,
                         ExpertCpuJobList* cpu_jobs, std::uint32_t cpu_share_q16,
                         cudaStream_t stream) {
    if (ids.dtype != DType::I32 || ids.data == nullptr) {
        throw std::invalid_argument("expert_slot_cache: ids must be a device I32 tensor");
    }
    if (layer < 0 || layer >= directory.layers) {
        throw std::invalid_argument("expert_slot_cache: layer out of range");
    }
    const std::int64_t count = ids.numel();
    if (count <= 0 || count > (1LL << 30)) {
        throw std::invalid_argument("expert_slot_cache: invalid id count");
    }
    // ids are [experts_per_token, tokens]; the leading extent is the per-token path count.
    const int per_token = ids.ne[0] > 0 ? ids.ne[0] : static_cast<int>(count);
    const bool split    = cpu_share_q16 > 0U && cpu_jobs != nullptr;
    if (split) {
        if (cpu_share_q16 > 65536U) {
            throw std::invalid_argument("expert_slot_cache: cpu share must be <= 65536");
        }
        if (alpha.dtype != DType::FP32 || alpha.data == nullptr || alpha.numel() != count) {
            throw std::invalid_argument("expert_slot_cache: the CPU split needs router weights matching ids");
        }
    }
    const int slots = static_cast<int>(directory.expert_of_slot.ne[0]);
    resolve_kernel<<<1, kResolveThreads, 0, stream>>>(
        static_cast<const int*>(ids.data), split ? static_cast<const float*>(alpha.data) : nullptr,
        static_cast<int>(count), layer, directory.experts, slots,
        static_cast<int*>(directory.slot_of_expert.data),
        static_cast<int*>(directory.expert_of_slot.data),
        static_cast<unsigned*>(directory.last_used.data),
        static_cast<unsigned*>(directory.active_round.data),
        static_cast<unsigned*>(directory.round.data), static_cast<int*>(directory.hand.data),
        static_cast<unsigned*>(directory.seen.data), static_cast<unsigned*>(directory.cpu_round.data),
        static_cast<int*>(misses.slots.data), static_cast<int*>(misses.experts.data),
        static_cast<long long*>(misses.count.data), misses.capacity, split ? cpu_share_q16 : 0U,
        per_token, split ? static_cast<int*>(cpu_jobs->tokens.data) : nullptr,
        split ? static_cast<int*>(cpu_jobs->experts.data) : nullptr,
        split ? static_cast<float*>(cpu_jobs->weights.data) : nullptr,
        split ? static_cast<long long*>(cpu_jobs->count.data) : nullptr,
        split ? cpu_jobs->capacity : 0);
    CUDA_CHECK(cudaGetLastError());
}

void expert_slot_gather(const ExpertHostBank& bank, const ExpertMissList& misses,
                        ExpertSlotPool& pool, cudaStream_t stream) {
    if (bank.gate_up_codes == nullptr || pool.gate_up_codes == nullptr) {
        throw std::invalid_argument("expert_slot_cache: gather needs a bank and a pool");
    }
    const std::uint64_t per_expert[4] = {
        bank.gate_up_codes_bytes_per_expert, bank.gate_up_scales_bytes_per_expert,
        bank.down_codes_bytes_per_expert, bank.down_scales_bytes_per_expert};
    for (const std::uint64_t bytes : per_expert) {
        if (bytes == 0 || bytes % 16 != 0) {
            throw std::invalid_argument("expert_slot_cache: expert planes must be 16-byte multiples");
        }
    }
    GatherParams p{};
    p.banks[0] = {bank.gate_up_codes, pool.gate_up_codes, per_expert[0]};
    p.banks[1] = {bank.gate_up_scales, pool.gate_up_scales, per_expert[1]};
    p.banks[2] = {bank.down_codes, pool.down_codes, per_expert[2]};
    p.banks[3] = {bank.down_scales, pool.down_scales, per_expert[3]};
    p.miss_slots   = static_cast<const int*>(misses.slots.data);
    p.miss_experts = static_cast<const int*>(misses.experts.data);
    p.miss_count   = static_cast<const long long*>(misses.count.data);
    gather_kernel<<<4 * kGatherBlocksPerBank, kGatherThreads, 0, stream>>>(p);
    CUDA_CHECK(cudaGetLastError());
}

SparseMoeWeights expert_slot_weights(const ExpertSlotPool& pool,
                                     const ExpertSlotDirectory& directory, std::int32_t layer,
                                     const SparseMoeWeights& resident_parts) {
    if (layer < 0 || layer >= directory.layers) {
        throw std::invalid_argument("expert_slot_cache: layer out of range");
    }
    SparseMoeWeights out   = resident_parts;
    out.routed_gate_up     = pool.routed_gate_up;
    out.routed_down        = pool.routed_down;
    out.slot_of_expert     = static_cast<const std::int32_t*>(directory.slot_of_expert.data) +
                         static_cast<std::int64_t>(layer) * directory.experts;
    return out;
}

} // namespace ninfer::ops
