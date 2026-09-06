// Expert slot cache: device pool + directory + per-round resolve/gather (api/ops/expert_slot_cache.h).
//
// Resolve is one block: hits are stamped in parallel, misses are assigned by one thread walking
// a clock hand over the slots (second-chance LRU: a slot whose last touch is older than the
// hand's sweep is a victim unless it is active in this round). Gather is FreeToken's
// multi-bank index copy: every 16-byte unit of every missing expert row, across the four planes
// (gate/up codes + scales, down codes + scales), from pinned host memory into the pool, the row
// count read from a device word so the launch captures into a graph.

#include "ops/linear/ggml/ggml_dispatch.h"
#include "ops/linear/ggml/ggml_moe_codec.cuh"
#include "api/ops/expert_slot_cache.h"
#include "core/device.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace sinfer::ops {
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
    // hand[0] = LRU clock, hand[1] = scan-ring cursor.
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
        hand[0] = 0;
        hand[1] = 0;
    }
}

/// One block. `ids` holds `count` expert ids of one layer (duplicates allowed). `seen` is a
/// per-layer scratch of `experts` stamps that dedupes the round's ids.
__global__ void __launch_bounds__(kResolveThreads)
    resolve_kernel(const int* __restrict__ ids, const float* __restrict__ alpha, int count,
                   int layer, int experts, int slots, int scan_ring, int scan,
                   int* __restrict__ slot_of_expert,
                   int* __restrict__ expert_of_slot, unsigned* __restrict__ last_used,
                   unsigned* __restrict__ active_round, unsigned* __restrict__ round,
                   int* __restrict__ hand, unsigned* __restrict__ seen,
                   unsigned* __restrict__ cpu_round, int* __restrict__ miss_slots,
                   int* __restrict__ miss_experts, long long* __restrict__ miss_count,
                   int miss_capacity, unsigned cpu_share_q16, int per_token,
                   int* __restrict__ cpu_tokens,
                   int* __restrict__ cpu_experts, float* __restrict__ cpu_weights,
                   long long* __restrict__ cpu_count, int cpu_capacity,
                   long long* __restrict__ stats) {
    __shared__ unsigned s_round;
    __shared__ int s_pending;
    __shared__ int s_pending_experts[1024];
    __shared__ int s_cpu_pending;
    __shared__ int s_distinct;
    const int tid = static_cast<int>(threadIdx.x);
    if (tid == 0) {
        s_round       = *round + 1U;
        *round        = s_round;
        s_pending     = 0;
        s_cpu_pending = 0;
        s_distinct    = 0;
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
        atomicAdd(&s_distinct, 1);
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
        // The LRU clock owns slots [0, lru_slots); scan-mode misses cycle through the trailing
        // ring [lru_slots, slots) instead, so a prompt's expert sweep cannot evict the decode
        // working set. Within one round the ring never wraps onto itself: misses are deduped,
        // so pending <= experts <= scan_ring, and active slots are skipped.
        const bool use_ring = scan != 0 && scan_ring > 0;
        const int lru_slots = slots - scan_ring;
        int h               = use_ring ? hand[1] : *hand;
        int written         = 0;
        int sent_to_cpu     = 0;
        for (int p = 0; p < pending; ++p) {
            const int expert = s_pending_experts[p];
            if (cpu_share_q16 > 0U && cpu_count != nullptr &&
                static_cast<unsigned long long>(sent_to_cpu) * 65536ULL <
                    static_cast<unsigned long long>(p + 1) * cpu_share_q16) {
                cpu_round[expert] = this_round;
                ++sent_to_cpu;
                continue;
            }
            int victim = -1;
            if (use_ring) {
                for (int probe = 0; probe < scan_ring; ++probe) {
                    const int s = lru_slots + (h + probe) % scan_ring;
                    if (active_round[s] == this_round) { continue; }
                    victim = s;
                    h      = (h + probe + 1) % scan_ring;
                    break;
                }
            } else {
                // Find a slot that is not active in this round; prefer the least recently used
                // among the next few candidates the hand passes (bounded sweep keeps it cheap).
                unsigned best_age = 0U;
                for (int probe = 0; probe < lru_slots; ++probe) {
                    const int s = (h + probe) % lru_slots;
                    if (active_round[s] == this_round) { continue; }
                    const unsigned age = this_round - last_used[s];
                    if (victim < 0 || age > best_age) {
                        victim   = s;
                        best_age = age;
                        if (expert_of_slot[s] < 0 || age >= 64U) { break; } // empty or stale
                    }
                }
                if (victim >= 0) { h = (victim + 1) % lru_slots; }
            }
            if (victim < 0) { break; } // every candidate slot is active in this round
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
        if (use_ring) {
            hand[1] = h;
        } else {
            *hand = h;
        }
        *miss_count   = written < miss_capacity ? written : miss_capacity;
        s_cpu_pending = sent_to_cpu;
        if (stats != nullptr) {
            // One block, one thread, launches serialised on the stream: a plain
            // read-modify-write is enough, and the counters survive graph replay because
            // the kernel is what increments them.
            const int gathered = written < miss_capacity ? written : miss_capacity;
            stats[kExpertSlotStatRounds] += 1;
            stats[kExpertSlotStatLookups] += count;
            stats[kExpertSlotStatDistinct] += s_distinct;
            stats[kExpertSlotStatResident] += s_distinct - s_pending;
            stats[kExpertSlotStatGathered] += gathered;
            stats[kExpertSlotStatHostRouted] += sent_to_cpu;
        }
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

// One Q4G32AM matrix of the bank: nibble codes plus FP16 scale/min planes on the host, W8
// codes and scales in the pool. One thread owns one 32-group: it decodes the affine grid,
// requantises to the pool's symmetric int8 (fresh amax/127 scale) and writes 32 codes + 1
// scale — the kernels reading the pool never learn the bank changed format.
struct GatherQ4Bank {
    const std::uint8_t* src_codes   = nullptr; // 16 B per group
    const std::uint16_t* src_scales = nullptr;
    const std::uint16_t* src_mins   = nullptr;
    std::byte* dst_codes            = nullptr; // pool planes (32 B codes, 2 B scale per group)
    std::byte* dst_scales           = nullptr;
    std::uint64_t groups_per_expert = 0;
    std::uint64_t dst_codes_bytes   = 0; // per expert (pool stride)
    std::uint64_t dst_scales_bytes  = 0;
};

struct GatherQ4Params {
    GatherQ4Bank banks[2];
    const int* miss_slots;
    const int* miss_experts;
    const long long* miss_count;
};

__global__ void __launch_bounds__(kGatherThreads)
    gather_unpack_q4_kernel(const __grid_constant__ GatherQ4Params p) {
    const int bank         = static_cast<int>(blockIdx.x) / kGatherBlocksPerBank;
    const int blk          = static_cast<int>(blockIdx.x) % kGatherBlocksPerBank;
    const GatherQ4Bank b   = p.banks[bank];
    const long long n      = *p.miss_count;
    const std::uint64_t total = static_cast<std::uint64_t>(n) * b.groups_per_expert;
    const std::uint64_t stride =
        static_cast<std::uint64_t>(kGatherBlocksPerBank) * kGatherThreads;
    for (std::uint64_t u = static_cast<std::uint64_t>(blk) * kGatherThreads + threadIdx.x;
         u < total; u += stride) {
        const std::uint64_t row     = u / b.groups_per_expert;
        const std::uint64_t group   = u - row * b.groups_per_expert;
        const std::uint64_t src_row = static_cast<std::uint64_t>(p.miss_experts[row]);
        const std::uint64_t dst_row = static_cast<std::uint64_t>(p.miss_slots[row]);
        const std::uint64_t g       = src_row * b.groups_per_expert + group;
        const uint4 packed = __ldcs(reinterpret_cast<const uint4*>(b.src_codes + g * 16));
        const float step   = __half2float(reinterpret_cast<const __half&>(b.src_scales[g]));
        const float lo     = __half2float(reinterpret_cast<const __half&>(b.src_mins[g]));
        float value[32];
        const std::uint32_t words[4] = {packed.x, packed.y, packed.z, packed.w};
        float amax = 0.0F;
#pragma unroll
        for (int w = 0; w < 4; ++w) {
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                const int q     = (words[w] >> (4 * i)) & 0xF;
                const float v   = step * static_cast<float>(q) + lo;
                value[w * 8 + i] = v;
                amax             = fmaxf(amax, fabsf(v));
            }
        }
        const float scale = amax / 127.0F;
        const float inv   = scale > 0.0F ? 1.0F / scale : 0.0F;
        std::uint32_t out[8]; // 32 int8 codes, byte j = code j
#pragma unroll
        for (int w = 0; w < 8; ++w) {
            std::uint32_t packed_out = 0;
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const int c = __float2int_rn(value[w * 4 + i] * inv);
                packed_out |= (static_cast<std::uint32_t>(c) & 0xFFU) << (8 * i);
            }
            out[w] = packed_out;
        }
        std::byte* dst = b.dst_codes + dst_row * b.dst_codes_bytes + group * 32;
        __stcs(reinterpret_cast<uint4*>(dst), make_uint4(out[0], out[1], out[2], out[3]));
        __stcs(reinterpret_cast<uint4*>(dst + 16), make_uint4(out[4], out[5], out[6], out[7]));
        *reinterpret_cast<__half*>(b.dst_scales + dst_row * b.dst_scales_bytes + group * 2) =
            __float2half_rn(scale);
    }
}

// One GGML-block matrix of the bank: the GGUF's own bytes on the host, the pool's W8 codes and
// scales on the device. Same shape as the Q4G32AM gather above -- one thread owns one 32-value
// group, decodes it and requantises to the pool's symmetric int8 -- so the kernels reading the
// pool never learn what the bank holds. What differs is only the decode, which is the same one
// the sparse-MoE codec uses, so a block format the codec can read is a bank format too.
struct GatherGgmlBank {
    const std::uint8_t* src_blocks  = nullptr;
    std::byte* dst_codes            = nullptr;
    std::byte* dst_scales           = nullptr;
    std::uint64_t groups_per_expert = 0;
    std::uint64_t src_bytes_per_expert = 0;
    std::uint64_t dst_codes_bytes   = 0;
    std::uint64_t dst_scales_bytes  = 0;
};

struct GatherGgmlIds {
    const int* miss_slots;
    const int* miss_experts;
    const long long* miss_count;
};

template <detail::ggml::GgmlType type>
__global__ void __launch_bounds__(kGatherThreads)
    gather_unpack_ggml_kernel(const __grid_constant__ GatherGgmlBank b,
                              const __grid_constant__ GatherGgmlIds ids) {
    const long long n         = *ids.miss_count;
    const std::uint64_t total = static_cast<std::uint64_t>(n) * b.groups_per_expert;
    const std::uint64_t stride =
        static_cast<std::uint64_t>(kGatherBlocksPerBank) * kGatherThreads;
    for (std::uint64_t u = static_cast<std::uint64_t>(blockIdx.x) * kGatherThreads + threadIdx.x;
         u < total; u += stride) {
        const std::uint64_t row     = u / b.groups_per_expert;
        const std::uint64_t group   = u - row * b.groups_per_expert;
        const std::uint64_t src_row = static_cast<std::uint64_t>(ids.miss_experts[row]);
        const std::uint64_t dst_row = static_cast<std::uint64_t>(ids.miss_slots[row]);
        const std::uint8_t* blocks  = b.src_blocks + src_row * b.src_bytes_per_expert;
        float value[32];
        detail::ggml::decode_group_32<type>(blocks, static_cast<std::int64_t>(group), value);
        float amax = 0.0F;
#pragma unroll
        for (int i = 0; i < 32; ++i) { amax = fmaxf(amax, fabsf(value[i])); }
        const float scale = amax / 127.0F;
        const float inv   = scale > 0.0F ? 1.0F / scale : 0.0F;
        std::uint32_t out[8];
#pragma unroll
        for (int w = 0; w < 8; ++w) {
            std::uint32_t packed_out = 0;
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const int c = __float2int_rn(value[w * 4 + i] * inv);
                packed_out |= (static_cast<std::uint32_t>(c) & 0xFFU) << (8 * i);
            }
            out[w] = packed_out;
        }
        std::byte* dst = b.dst_codes + dst_row * b.dst_codes_bytes + group * 32;
        __stcs(reinterpret_cast<uint4*>(dst), make_uint4(out[0], out[1], out[2], out[3]));
        __stcs(reinterpret_cast<uint4*>(dst + 16), make_uint4(out[4], out[5], out[6], out[7]));
        *reinterpret_cast<__half*>(b.dst_scales + dst_row * b.dst_scales_bytes + group * 2) =
            __float2half_rn(scale);
    }
}

__global__ void partial_add_kernel(const float* __restrict__ partial, __nv_bfloat16* __restrict__ destination,
                                   long long count) {
    const long long i = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= count) { return; }
    destination[i] = __float2bfloat16_rn(__bfloat162float(destination[i]) + partial[i]);
}

/// One launch per routed half, each resolved over the whole block vocabulary. Splitting the
/// two halves is what keeps this linear: the gate/up and down projections of a K_XL mix are
/// routinely different formats, and enumerating pairs would be quadratic and would silently
/// refuse whichever combination nobody thought to list.
void launch_ggml_gather(QType type, const GatherGgmlBank& bank, const GatherGgmlIds& ids,
                        cudaStream_t stream) {
    using T = detail::ggml::GgmlType;
    switch (type) {
#define SINFER_GATHER_CASE(NAME)                                                                   \
    case QType::NAME:                                                                              \
        gather_unpack_ggml_kernel<T::NAME>                                                         \
            <<<kGatherBlocksPerBank, kGatherThreads, 0, stream>>>(bank, ids);                       \
        CUDA_CHECK(cudaGetLastError());                                                            \
        return;
        SINFER_GGML_FOR_EACH_TYPE(SINFER_GATHER_CASE)
#undef SINFER_GATHER_CASE
    default:
        break;
    }
    throw std::invalid_argument(
        "expert_slot_cache: routed experts are not a GGML block format (QType " +
        std::to_string(static_cast<int>(type)) + ")");
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
                                                 std::int32_t slots, std::int32_t scan_ring,
                                                 void* device_bytes, cudaStream_t stream) {
    if (scan_ring != 0 && (scan_ring < experts || scan_ring > slots / 2)) {
        throw std::invalid_argument(
            "expert_slot_cache: scan ring must be 0, or in [experts, slots/2] so one deduped "
            "per-layer scan always fits without wrapping onto itself");
    }
    if (device_bytes == nullptr) {
        throw std::invalid_argument("expert_slot_cache: directory memory is null");
    }
    const std::int32_t ids = layers * experts;
    auto* base             = static_cast<std::byte*>(device_bytes);
    ExpertSlotDirectory out;
    out.layers    = layers;
    out.scan_ring = scan_ring;
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
    out.hand           = take(DType::I32, 2, 2 * sizeof(int));
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

namespace {

/// Fills one half of the bank from its source. Three shapes: W8 row-split is the pool's own
/// layout, copied straight through; GGML blocks are the GGUF's bytes, decoded a group at a time
/// on the way into the pool; Q4G32AM planes are the requantised bank, decoded the same way.
void fill_bank_half(const SparseMoeGeometry& geometry, const ExpertBankHalfSource& source,
                    bool gate_up, ExpertHostBank& bank) {
    const std::int64_t rows_per_expert = gate_up ? geometry.expert_rows() : geometry.hidden;
    const std::int32_t k               = gate_up ? geometry.hidden : geometry.intermediate;
    ExpertBankFormat& format          = gate_up ? bank.gate_up_format : bank.down_format;
    const std::byte*& codes           = gate_up ? bank.gate_up_codes : bank.down_codes;
    const std::byte*& scales          = gate_up ? bank.gate_up_scales : bank.down_scales;
    const std::byte*& mins            = gate_up ? bank.gate_up_mins : bank.down_mins;
    std::uint64_t& codes_bytes        = gate_up ? bank.gate_up_codes_bytes_per_expert
                                                : bank.down_codes_bytes_per_expert;
    std::uint64_t& scales_bytes       = gate_up ? bank.gate_up_scales_bytes_per_expert
                                                : bank.down_scales_bytes_per_expert;
    QType& ggml                       = gate_up ? bank.gate_up_ggml : bank.down_ggml;
    const char* half                  = gate_up ? "gate/up" : "down";
    if (source.q4_base != nullptr) {
        const Q4BankPlanes planes =
            q4_bank_planes(static_cast<std::int64_t>(geometry.experts) * rows_per_expert, k);
        const auto experts = static_cast<std::uint64_t>(geometry.experts);
        format       = ExpertBankFormat::Q4G32AM;
        codes        = static_cast<const std::byte*>(source.q4_base);
        scales       = static_cast<const std::byte*>(source.q4_base) + planes.scales_offset;
        mins         = static_cast<const std::byte*>(source.q4_base) + planes.mins_offset;
        codes_bytes  = planes.codes_bytes / experts;
        scales_bytes = planes.groups / experts * 2;
        return;
    }
    if (source.weight == nullptr) {
        throw std::invalid_argument(std::string("expert_slot_cache: the bank's ") + half +
                                    " half has no source");
    }
    const Weight& w = *source.weight;
    if (w.n != static_cast<std::int32_t>(geometry.experts * rows_per_expert) || w.k != k) {
        throw std::invalid_argument(std::string("expert_slot_cache: host bank ") + half +
                                    " shape does not match the geometry");
    }
    if (detail::ggml::is_ggml_qtype(w.qtype) && w.layout == QuantLayout::GgmlBlocks) {
        // A GGML block carries its own scale, so there is no scales plane: the per-expert
        // stride is the block bytes of one expert's rows, and the "scales bytes" the gather
        // reads is only how it derives the group count.
        const auto values = detail::ggml::block_values(detail::ggml::ggml_type_for(w.qtype));
        const auto bytes  = detail::ggml::block_bytes(detail::ggml::ggml_type_for(w.qtype));
        format       = ExpertBankFormat::GgmlBlocks;
        ggml         = w.qtype;
        codes        = static_cast<const std::byte*>(w.qdata);
        scales       = nullptr;
        mins         = nullptr;
        codes_bytes  = static_cast<std::uint64_t>(rows_per_expert) * (w.k / values) * bytes;
        scales_bytes = static_cast<std::uint64_t>(rows_per_expert) * (k / 32) * 2;
        return;
    }
    if (w.qtype == QType::W8G32_F16S && w.layout == QuantLayout::RowSplit) {
        const W8Planes planes = w8_planes(rows_per_expert, k);
        format       = ExpertBankFormat::W8G32;
        codes        = static_cast<const std::byte*>(w.qdata);
        scales       = static_cast<const std::byte*>(w.scales);
        mins         = nullptr;
        codes_bytes  = planes.codes_plane_bytes;
        scales_bytes = planes.scales_plane_bytes;
        return;
    }
    throw std::invalid_argument(std::string("expert_slot_cache: the host bank's ") + half +
                                " half is neither W8 row-split planes nor GGML blocks");
}

} // namespace

ExpertHostBank expert_host_bank(const SparseMoeGeometry& geometry, ExpertBankHalfSource gate_up,
                                ExpertBankHalfSource down) {
    require_geometry(geometry);
    ExpertHostBank bank;
    fill_bank_half(geometry, gate_up, true, bank);
    fill_bank_half(geometry, down, false, bank);
    return bank;
}

ExpertHostBank expert_host_bank(const SparseMoeGeometry& geometry, const Weight& routed_gate_up,
                                const Weight& routed_down) {
    return expert_host_bank(geometry, ExpertBankHalfSource{&routed_gate_up, nullptr},
                            ExpertBankHalfSource{&routed_down, nullptr});
}

// ---------------------------------------------------------------------------------------------
// Per-round operations
// ---------------------------------------------------------------------------------------------

void expert_slot_resolve(const Tensor& ids, std::int32_t layer, ExpertSlotDirectory& directory,
                         ExpertMissList& misses, cudaStream_t stream, bool scan) {
    expert_slot_resolve(ids, Tensor{}, layer, directory, misses, nullptr, 0U,
                        ids.ne[0] > 0 ? ids.ne[0] : 1, stream, scan);
}

void expert_slot_resolve(const Tensor& ids, const Tensor& alpha, std::int32_t layer,
                         ExpertSlotDirectory& directory, ExpertMissList& misses,
                         ExpertCpuJobList* cpu_jobs, std::uint32_t cpu_share_q16,
                         std::int32_t experts_per_token, cudaStream_t stream, bool scan) {
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
    // A job's token is its index over the per-token path count, which the caller states:
    // the prefill path passes a flat [assignments] view of the same token-major layout, and
    // reading the count off the leading extent put every job of such a round on token 0.
    if (experts_per_token <= 0 || count % experts_per_token != 0) {
        throw std::invalid_argument("expert_slot_cache: id count is not a multiple of the paths per token");
    }
    const int per_token = experts_per_token;
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
        static_cast<int>(count), layer, directory.experts, slots, directory.scan_ring,
        scan ? 1 : 0, static_cast<int*>(directory.slot_of_expert.data),
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
        split ? cpu_jobs->capacity : 0,
        static_cast<long long*>(directory.stats.data));
    CUDA_CHECK(cudaGetLastError());
}

Q4BankPlanes q4_bank_planes(std::int64_t rows_total, std::int32_t k) {
    if (rows_total <= 0 || k <= 0 || k % kW8Group != 0) {
        throw std::invalid_argument("expert_slot_cache: q4 planes need rows and k % 32 == 0");
    }
    Q4BankPlanes out;
    out.groups        = static_cast<std::uint64_t>(rows_total) * static_cast<std::uint64_t>(k) /
                 static_cast<std::uint64_t>(kW8Group);
    out.codes_bytes   = out.groups * 16;
    out.scales_offset = out.codes_bytes;
    out.mins_offset   = out.scales_offset + out.groups * 2;
    out.total_bytes   = out.mins_offset + out.groups * 2;
    return out;
}

ExpertHostBank expert_host_bank_q4(const SparseMoeGeometry& geometry, const void* gate_up_base,
                                   const void* down_base) {
    if (gate_up_base == nullptr || down_base == nullptr) {
        throw std::invalid_argument("expert_slot_cache: q4 bank needs both object base pointers");
    }
    return expert_host_bank(geometry, ExpertBankHalfSource{nullptr, gate_up_base},
                            ExpertBankHalfSource{nullptr, down_base});
}

namespace {

/// Gathers one half of the missed experts into the pool by that half's format. The pool
/// always holds W8, so a W8 half is a copy and the other two decode a 32-group at a time.
void gather_half(const ExpertHostBank& bank, bool gate_up, const ExpertMissList& misses,
                 ExpertSlotPool& pool, cudaStream_t stream) {
    const ExpertBankFormat format = gate_up ? bank.gate_up_format : bank.down_format;
    const std::byte* codes        = gate_up ? bank.gate_up_codes : bank.down_codes;
    const std::byte* scales       = gate_up ? bank.gate_up_scales : bank.down_scales;
    const std::byte* mins         = gate_up ? bank.gate_up_mins : bank.down_mins;
    const std::uint64_t codes_bytes =
        gate_up ? bank.gate_up_codes_bytes_per_expert : bank.down_codes_bytes_per_expert;
    const std::uint64_t scales_bytes =
        gate_up ? bank.gate_up_scales_bytes_per_expert : bank.down_scales_bytes_per_expert;
    std::byte* dst_codes        = gate_up ? pool.gate_up_codes : pool.down_codes;
    std::byte* dst_scales       = gate_up ? pool.gate_up_scales : pool.down_scales;
    const int* miss_slots       = static_cast<const int*>(misses.slots.data);
    const int* miss_experts     = static_cast<const int*>(misses.experts.data);
    const long long* miss_count = static_cast<const long long*>(misses.count.data);
    // The destination strides are the pool's, not the bank's: one group is 32 code bytes and
    // one f16 scale however many bytes the bank's form of it took.
    const std::uint64_t groups = scales_bytes / 2;
    if (format == ExpertBankFormat::GgmlBlocks) {
        const GatherGgmlIds ids{miss_slots, miss_experts, miss_count};
        const GatherGgmlBank half{reinterpret_cast<const std::uint8_t*>(codes), dst_codes,
                                  dst_scales, groups, codes_bytes, groups * 32, groups * 2};
        launch_ggml_gather(gate_up ? bank.gate_up_ggml : bank.down_ggml, half, ids, stream);
        return;
    }
    if (format == ExpertBankFormat::Q4G32AM) {
        if (mins == nullptr) {
            throw std::invalid_argument("expert_slot_cache: q4 bank half has no min plane");
        }
        GatherQ4Params p{};
        p.banks[0] = {reinterpret_cast<const std::uint8_t*>(codes),
                      reinterpret_cast<const std::uint16_t*>(scales),
                      reinterpret_cast<const std::uint16_t*>(mins),
                      dst_codes,
                      dst_scales,
                      groups,
                      codes_bytes * 2,
                      scales_bytes};
        p.miss_slots   = miss_slots;
        p.miss_experts = miss_experts;
        p.miss_count   = miss_count;
        gather_unpack_q4_kernel<<<kGatherBlocksPerBank, kGatherThreads, 0, stream>>>(p);
        CUDA_CHECK(cudaGetLastError());
        return;
    }
    for (const std::uint64_t bytes : {codes_bytes, scales_bytes}) {
        if (bytes == 0 || bytes % 16 != 0) {
            throw std::invalid_argument("expert_slot_cache: expert planes must be 16-byte multiples");
        }
    }
    GatherParams p{};
    p.banks[0]     = {codes, dst_codes, codes_bytes};
    p.banks[1]     = {scales, dst_scales, scales_bytes};
    p.miss_slots   = miss_slots;
    p.miss_experts = miss_experts;
    p.miss_count   = miss_count;
    gather_kernel<<<2 * kGatherBlocksPerBank, kGatherThreads, 0, stream>>>(p);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace

void expert_slot_gather(const ExpertHostBank& bank, const ExpertMissList& misses,
                        ExpertSlotPool& pool, cudaStream_t stream) {
    if (bank.gate_up_codes == nullptr || bank.down_codes == nullptr ||
        pool.gate_up_codes == nullptr) {
        throw std::invalid_argument("expert_slot_cache: gather needs a bank and a pool");
    }
    gather_half(bank, true, misses, pool, stream);
    gather_half(bank, false, misses, pool, stream);
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

} // namespace sinfer::ops
