#pragma once

// CPU expert compute (phase 2, design D3): the misses an expert slot cache does not fetch over
// PCIe are computed on the host, straight from the pinned expert bank, at host DRAM speed. The
// bank is the engine's own planar W8G32 layout (int8 codes plane + fp16 scales plane per matrix,
// one expert = a contiguous row block in each plane), so no second copy of the experts exists.
//
// Numerics follow the GPU kernels: the activation is quantised per 32-wide group to int8 with a
// float scale, the dot is an exact int32 accumulation per group scaled by (w_scale * x_scale),
// gate/up use SiLU(gate) * up, and the intermediate is re-quantised the same way before the down
// projection. Model-agnostic: everything is a SparseMoeGeometry.
//
// Status (2026-08-28): host-side op, unit-tested on the CPU; the GPU round integration (miss
// split, activation/weight hand-off, host-function handshake) follows.

#include "api/ops/sparse_moe.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <vector>

namespace ninfer::ops {

/// One layer's routed experts in host memory (the pinned bank's host pointers).
/// Storage of the routed expert bank the host (and the miss gather) read.
/// - W8G32: int8 codes, one FP16 scale per 32-group (the artifact's own encoding).
/// - Q4G32AM: unsigned 4-bit codes packed two per byte (low nibble = even element), one FP16
///   scale AND one FP16 min per 32-group (`w = scale * q + min`). Requantised from W8 at load;
///   the affine form reproduces the Q4_K-derived weights almost exactly at 59 % of the bytes.
enum class ExpertBankFormat : std::uint8_t { W8G32 = 0, Q4G32AM = 1 };

struct CpuExpertBank {
    ExpertBankFormat format         = ExpertBankFormat::W8G32;
    const std::byte* gate_up_codes  = nullptr; // [experts][2*intermediate][hidden] int8 | u4x2
    const std::byte* gate_up_scales = nullptr; // [experts][2*intermediate][hidden/32] fp16
    const std::byte* gate_up_mins   = nullptr; // Q4G32AM only, same shape as the scales
    const std::byte* down_codes     = nullptr; // [experts][hidden][intermediate] int8 | u4x2
    const std::byte* down_scales    = nullptr; // [experts][hidden][intermediate/32] fp16
    const std::byte* down_mins      = nullptr; // Q4G32AM only
};

/// Requantises `groups` W8 groups (32 int8 codes + FP16 scale each, in parallel plane order)
/// into Q4G32AM: per group, the decoded values' [min, max] span becomes a 16-level affine grid
/// (both endpoints stored as FP16, and the codes are fitted against the *rounded* endpoints).
/// Runs on the calling thread; callers parallelise over disjoint group ranges.
void requantise_w8_expert_groups_to_q4(const std::int8_t* codes, const std::uint16_t* scales,
                                       std::int64_t groups, std::uint8_t* q4,
                                       std::uint16_t* q4_scales, std::uint16_t* q4_mins);

/// One (token, expert) pair the CPU computes; `weight` is the router weight of that path.
struct CpuExpertJob {
    std::int32_t token  = 0;
    std::int32_t expert = 0;
    float weight        = 0.0F;
};

/// A round: activations for `tokens` columns (BF16, column-major [hidden, tokens] as the
/// engine stores them) and the jobs; results accumulate into `out` (FP32 [hidden, tokens]),
/// which the caller zeroes.
struct CpuExpertRound {
    const std::uint16_t* x = nullptr; // BF16 bits, [hidden, tokens]
    float* out             = nullptr; // FP32 [hidden, tokens], accumulated (weight-scaled)
    std::int32_t tokens    = 0;
    std::span<const CpuExpertJob> jobs;
};

/// Computes one job on the calling thread (reference entry point; the pool uses it too).
/// `scratch` must hold cpu_expert_scratch_bytes(geometry) bytes, 64-byte aligned.
[[nodiscard]] std::size_t cpu_expert_scratch_bytes(const SparseMoeGeometry& geometry);
void cpu_expert_compute_job(const SparseMoeGeometry& geometry, const CpuExpertBank& bank,
                            const CpuExpertJob& job, const std::uint16_t* x_column, float* out_column,
                            std::byte* scratch);

/// A pool of pinned worker threads (one per physical core by default) that splits a round's
/// jobs across cores and blocks until they are done. Output columns shared by several jobs
/// are accumulated under a per-token lock, so job order does not matter.
struct CpuExpertPoolOptions {
    std::uint32_t threads = 0; // 0 = one per physical core
    bool pin_threads      = true;
    std::vector<int> cpus;     // explicit CPUs to pin to (one thread each); overrides `threads`
};

class CpuExpertPool {
public:
    using Options = CpuExpertPoolOptions;
    explicit CpuExpertPool(const SparseMoeGeometry& geometry, Options options = Options{});
    ~CpuExpertPool();
    CpuExpertPool(const CpuExpertPool&)            = delete;
    CpuExpertPool& operator=(const CpuExpertPool&) = delete;

    [[nodiscard]] std::uint32_t threads() const noexcept;
    /// Runs the round to completion on the pool (the caller's thread waits).
    void run(const CpuExpertBank& bank, const CpuExpertRound& round);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

/// True when the AVX-512 path is compiled in and the CPU supports it; otherwise the scalar
/// path runs (same numerics, slower).
[[nodiscard]] bool cpu_expert_compute_has_avx512() noexcept;
/// True when the AVX512-VNNI batched inner loop is in use (compiled, supported, and not vetoed
/// by SUROGATE_CPU_EXPERT_NO_VNNI=1).
[[nodiscard]] bool cpu_expert_compute_has_vnni() noexcept;
/// True when expert groups with enough tokens run through the repacked 16-row tiles
/// (needs VNNI; opt-in with SUROGATE_CPU_EXPERT_TILE=1, SUROGATE_CPU_EXPERT_TILE_MIN=<tokens>
/// sets the group size from which the repack pays).
[[nodiscard]] bool cpu_expert_compute_has_tile() noexcept;

} // namespace ninfer::ops
