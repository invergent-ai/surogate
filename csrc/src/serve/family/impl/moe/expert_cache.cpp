#include "family/impl/moe/expert_cache.h"

#include "core/device.h"
#include "core/numa.h"
#include "core/sleep.h"

#include <cuda.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <map>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

namespace sinfer::family {
namespace {

// The two stream memory operations, resolved from the driver at runtime: the runtime header
// does not declare them, and a driver without them just leaves the host-function path on.
using StreamWriteValue64Fn = CUresult (*)(CUstream, CUdeviceptr, cuuint64_t, unsigned int);
using StreamWaitValue64Fn  = CUresult (*)(CUstream, CUdeviceptr, cuuint64_t, unsigned int);
StreamWriteValue64Fn stream_write_value64() {
    static StreamWriteValue64Fn fn = [] {
        void* p = nullptr;
        cudaDriverEntryPointQueryResult status{};
        if (cudaGetDriverEntryPointByVersion("cuStreamWriteValue64", &p, 12000, cudaEnableDefault,
                                             &status) != cudaSuccess ||
            status != cudaDriverEntryPointSuccess) {
            return StreamWriteValue64Fn{nullptr};
        }
        return reinterpret_cast<StreamWriteValue64Fn>(p);
    }();
    return fn;
}
StreamWaitValue64Fn stream_wait_value64() {
    static StreamWaitValue64Fn fn = [] {
        void* p = nullptr;
        cudaDriverEntryPointQueryResult status{};
        if (cudaGetDriverEntryPointByVersion("cuStreamWaitValue64", &p, 12000, cudaEnableDefault,
                                             &status) != cudaSuccess ||
            status != cudaDriverEntryPointSuccess) {
            return StreamWaitValue64Fn{nullptr};
        }
        return reinterpret_cast<StreamWaitValue64Fn>(p);
    }();
    return fn;
}
constexpr unsigned int kStreamWaitValueGeq = 0x0; // CU_STREAM_WAIT_VALUE_GEQ

/// The widest round that still takes the decode share (`Impl::share_for`). One token per
/// expert is the host's fast shape; several tokens per expert is its slow one, so the band is
/// about where a batch stops being one-token-per-expert rather than a memory size.
constexpr std::int32_t kDefaultDecodeBand = 64;

[[nodiscard]] inline bool cpu_moe_verify() {
    static const bool on = std::getenv("SUROGATE_SERVE_CPU_MOE_VERIFY") != nullptr;
    return on;
}

inline void check_device_handoff(const char* what, int produced_on) {
    if (!cpu_moe_verify() || produced_on < 0) { return; }
    int current = -1;
    cudaGetDevice(&current);
    if (current != produced_on) {
        std::fprintf(stderr,
                     "expert cache: %s was produced on device %d and is being consumed on device "
                     "%d\n",
                     what, produced_on, current);
    }
}

// How long the GPU actually waits for the host round (SUROGATE_SERVE_CPU_MOE_JOIN_PROBE=1).
//
// The CPU split forks the host experts onto a side stream and joins them at the combine, and
// the design assumes the join is free -- that the host round finishes inside the GPU's own
// expert work. Whether that holds decides where single-user time goes: if the GPU waits, the
// round is paced by the host and the lever is host round latency, not the PCIe gather.
//
// A pair of events straddles the wait, and the pair is read one round later, when it has long
// since completed -- so the probe never adds a synchronise of its own. Eager only: under graph
// capture an event record is a node and elapsed time between nodes is not a quantity you can
// ask for.
struct JoinProbe {
    // A ring rather than a pair. With two buffers a pair whose events were not yet complete
    // got overwritten on its next turn, so only the joins that finished fastest were ever
    // measured -- a probe for stalls that silently dropped the stalls. The ring is deep enough
    // that the oldest entry has long completed, and it is drained with a blocking wait rather
    // than a query, so every join is counted.
    static constexpr int kRing = 16;
    bool enabled     = false;
    bool initialised = false;
    int slot         = 0;
    bool pending[kRing]{};
    cudaEvent_t before[kRing]{};
    cudaEvent_t after[kRing]{};
    double waited_ms = 0.0;
    float worst_ms     = 0.0F;
    bool open          = false;
    std::int64_t joins = 0;
    std::int64_t combines = 0;
    std::int64_t reported = 0;

    void ensure() {
        if (initialised) { return; }
        initialised = true;
        for (int i = 0; i < kRing; ++i) {
            CUDA_CHECK(cudaEventCreate(&before[i]));
            CUDA_CHECK(cudaEventCreate(&after[i]));
        }
    }

    /// Reporting for the gather instance: same cadence, its own wording.
    void tick_gather() {
        if (!enabled) { return; }
        ++combines;
        if (combines < reported + 512) { return; }
        reported = combines;
        if (joins == 0) {
            std::fprintf(stderr, "expert cache: miss-gather never ran in %lld layers\n",
                         static_cast<long long>(combines));
            return;
        }
        std::fprintf(stderr,
                     "expert cache: miss-gather occupied %.3f ms over %lld layers "
                     "(mean %.3f ms/layer, worst %.3f)\n",
                     waited_ms, static_cast<long long>(joins),
                     waited_ms / static_cast<double>(joins), static_cast<double>(worst_ms));
    }

    /// Called on every combine, join or not: the report lives here rather than in `wrap`
    /// because a run where the split never engages takes no joins at all, and that is the
    /// case the probe most needs to be able to say out loud.
    void tick() {
        if (!enabled) { return; }
        ++combines;
        if (combines < reported + 512) { return; }
        reported = combines;
        if (joins > 0) {
            std::fprintf(stderr,
                         "expert cache: host-round join waited %.3f ms over %lld joins of %lld "
                         "combines (mean %.3f ms/join, worst %.3f)\n",
                         waited_ms, static_cast<long long>(joins),
                         static_cast<long long>(combines),
                         waited_ms / static_cast<double>(joins),
                         static_cast<double>(worst_ms));
        } else {
            std::fprintf(stderr,
                         "expert cache: host-round join never taken in %lld combines -- the CPU "
                         "split did not engage (rounds below --cpu-moe-min-tokens carry no host "
                         "partial, so every miss of such a round crosses PCIe)\n",
                         static_cast<long long>(combines));
        }
    }

    /// Opens a timed span on `stream`, reclaiming the ring slot it reuses.
    void begin(cudaStream_t stream) {
        if (!enabled) { return; }
        cudaStreamCaptureStatus capture = cudaStreamCaptureStatusNone;
        if (cudaStreamIsCapturing(stream, &capture) != cudaSuccess ||
            capture != cudaStreamCaptureStatusNone) {
            return;
        }
        ensure();
        if (pending[slot]) {
            CUDA_CHECK(cudaEventSynchronize(after[slot]));
            float ms = 0.0F;
            CUDA_CHECK(cudaEventElapsedTime(&ms, before[slot], after[slot]));
            waited_ms += static_cast<double>(ms);
            if (ms > worst_ms) { worst_ms = ms; }
            ++joins;
            pending[slot] = false;
        }
        CUDA_CHECK(cudaEventRecord(before[slot], stream));
        open = true;
    }

    /// Closes the span opened by `begin`.
    void end(cudaStream_t stream) {
        if (!enabled || !open) { return; }
        open = false;
        CUDA_CHECK(cudaEventRecord(after[slot], stream));
        pending[slot] = true;
        slot          = (slot + 1) % kRing;
    }

    /// Drains whichever pair has completed, then straddles this join with the other.
    void wrap(cudaStream_t stream, cudaEvent_t join) {
        if (!enabled) {
            CUDA_CHECK(cudaStreamWaitEvent(stream, join, 0));
            return;
        }
        cudaStreamCaptureStatus capture = cudaStreamCaptureStatusNone;
        if (cudaStreamIsCapturing(stream, &capture) != cudaSuccess ||
            capture != cudaStreamCaptureStatusNone) {
            CUDA_CHECK(cudaStreamWaitEvent(stream, join, 0));
            return;
        }
        ensure();
        // Reclaim the slot we are about to reuse: it is kRing joins old, so the wait is
        // nominally free, and blocking rather than querying keeps the sample unbiased.
        if (pending[slot]) {
            CUDA_CHECK(cudaEventSynchronize(after[slot]));
            float ms = 0.0F;
            CUDA_CHECK(cudaEventElapsedTime(&ms, before[slot], after[slot]));
            waited_ms += static_cast<double>(ms);
            if (ms > worst_ms) { worst_ms = ms; }
            ++joins;
            pending[slot] = false;
        }
        CUDA_CHECK(cudaEventRecord(before[slot], stream));
        CUDA_CHECK(cudaStreamWaitEvent(stream, join, 0));
        CUDA_CHECK(cudaEventRecord(after[slot], stream));
        pending[slot] = true;
        slot          = (slot + 1) % kRing;
    }
};

/// How long the PCIe miss-gather occupies the round. The bytes-over-bandwidth estimate says
/// it is a large share of single-user decode; this measures it, because whether overlapping
/// the gather with the hit-path compute is worth the surgery depends on the real number.
JoinProbe& gather_probe() {
    static JoinProbe probe = [] {
        JoinProbe p;
        const char* on = std::getenv("SUROGATE_SERVE_GATHER_PROBE");
        p.enabled      = on != nullptr && *on != '\0' && *on != '0';
        return p;
    }();
    return probe;
}

JoinProbe& join_probe() {
    static JoinProbe probe = [] {
        JoinProbe p;
        const char* on = std::getenv("SUROGATE_SERVE_CPU_MOE_JOIN_PROBE");
        p.enabled      = on != nullptr && *on != '\0' && *on != '0';
        return p;
    }();
    return probe;
}

// A host-computed partial of the block being executed: consumed by the next combine.
struct PendingPartial {
    const float* device_alias = nullptr;
    cudaEvent_t join          = nullptr;
    int device                = -1; // the device whose buffers this partial points at
    // Memop handshake: the combine waits for the slice's `done` flag and clears it, instead of
    // waiting on the join event, when the handshake carried the round.
    CUdeviceptr done_device = 0;
};

// --- what the run configured, per device, before the cache exists ---

std::mutex& config_mutex() {
    static std::mutex mutex;
    return mutex;
}
std::unordered_map<int, std::uint32_t>& configured_expert_slots() {
    static std::unordered_map<int, std::uint32_t> configured;
    return configured;
}
std::unordered_map<int, float>& configured_cpu_share() {
    static std::unordered_map<int, float> configured;
    return configured;
}
std::unordered_map<int, std::pair<float, std::uint32_t>>& configured_cpu_prefill() {
    static std::unordered_map<int, std::pair<float, std::uint32_t>> map;
    return map;
}
bool& configured_cpu_pool_per_socket() {
    static bool per_socket = false;
    return per_socket;
}
/// What the runtime must be left after the pool: the KV floor and headroom the planner asked
/// for. Zero when no planner said (a test, or a caller that sizes the pool by hand).
std::unordered_map<int, std::size_t>& configured_derived_reserve() {
    static std::unordered_map<int, std::size_t> reserves;
    return reserves;
}
std::unordered_map<int, std::size_t>& configured_load_staging() {
    static std::unordered_map<int, std::size_t> staging;
    return staging;
}
std::unordered_map<int, std::size_t>& configured_runtime_floor() {
    static std::unordered_map<int, std::size_t> floors;
    return floors;
}
std::unordered_map<int, std::size_t>& configured_pool_floor() {
    static std::unordered_map<int, std::size_t> floors;
    return floors;
}
/// Two GiB, not one: the floor's largest term is a projection of the load's staging that is
/// exact for the arena and blind to the allocator's granularity, the input maps and whatever
/// else the load carves; on GLM's card one GiB left a pool that fit the weights by a few
/// hundred MiB and refused a 2,048-token chunk.
constexpr std::size_t kPoolMarginBytes = std::size_t{2} << 30;
std::unordered_map<int, std::uint32_t>& configured_cpu_min_tokens() {
    static std::unordered_map<int, std::uint32_t> configured;
    return configured;
}

// The NUMA node of a CUDA device (its PCI function's `numa_node`), -1 when unknown.
int device_numa_node(int device) {
    char bus_id[32] = {};
    if (cudaDeviceGetPCIBusId(bus_id, sizeof(bus_id), device) != cudaSuccess) { return -1; }
    std::string path = "/sys/bus/pci/devices/";
    for (char* c = bus_id; *c != '\0'; ++c) {
        *c = static_cast<char>(std::tolower(static_cast<unsigned char>(*c)));
    }
    path += bus_id;
    path += "/numa_node";
    FILE* file = std::fopen(path.c_str(), "r");
    if (file == nullptr) { return -1; }
    int node = -1;
    if (std::fscanf(file, "%d", &node) != 1) { node = -1; }
    std::fclose(file);
    return node;
}

// The physical cores of a NUMA node: the first half of its cpulist on SMT-2 parts (the
// usual numbering lists the siblings after all physical cores).
std::vector<int> node_physical_cpus(int node) {
    std::vector<int> cpus;
    const std::string path = "/sys/devices/system/node/node" + std::to_string(node) + "/cpulist";
    FILE* file             = std::fopen(path.c_str(), "r");
    if (file == nullptr) { return cpus; }
    char buffer[512] = {};
    if (std::fgets(buffer, sizeof(buffer), file) != nullptr) {
        const char* p = buffer;
        while (*p != '\0' && *p != '\n') {
            char* end    = nullptr;
            const long a = std::strtol(p, &end, 10);
            long b       = a;
            p            = end;
            if (*p == '-') {
                b = std::strtol(p + 1, &end, 10);
                p = end;
            }
            for (long c = a; c <= b; ++c) { cpus.push_back(static_cast<int>(c)); }
            if (*p == ',') { ++p; }
        }
    }
    std::fclose(file);
    const unsigned threads_per_core = std::thread::hardware_concurrency() / 2 > 0 ? 2 : 1;
    if (threads_per_core == 2 && cpus.size() >= 2) { cpus.resize(cpus.size() / 2); }
    return cpus;
}

/// The host pools of this process, one per NUMA node (-1 = shared) and mixture geometry:
/// pipeline stages of one model share theirs; a second mixture shape in the same process gets
/// its own, since a pool's scratch is cut to one geometry.
std::string pool_key(int node, const ops::SparseMoeGeometry& geometry) {
    return std::to_string(node) + ":" + std::to_string(geometry.hidden) + "x" +
           std::to_string(geometry.intermediate) + "x" + std::to_string(geometry.experts) + "/" +
           std::to_string(geometry.experts_per_token);
}

} // namespace

// -------------------------------------------------------------------------------------------
// The cache
// -------------------------------------------------------------------------------------------

struct ExpertCache::Impl {
    ops::SparseMoeGeometry geometry{};
    std::int32_t layers = 0;

    struct Layer {
        Impl* owner            = nullptr;
        std::int32_t index     = -1;
        ops::ExpertHostBank bank;
        ops::CpuExpertBank cpu_bank; // host addresses of the same planes
        std::int32_t round_tokens = 0; // set before the host function of a round is enqueued
        // Round bookkeeping for the CPU split: a round may reach the hook in several slices
        // (prefill and mixed rounds); the split decision is per round and each slice is staged
        // at its column offset.
        std::int32_t round_total  = 0;
        std::int32_t round_offset = 0;
        std::int32_t round_slice  = 0; // ordinal of the next slice within the round
        bool round_split          = false;
    };
    // Pinned mirrors of the device job list, one per slice ordinal: the copies run on the
    // main stream, so slice k+1's copy may land while slice k's host function still reads
    // its jobs -- each slice therefore owns a mirror.
    struct JobMirror {
        void* block          = nullptr;
        std::int32_t* tokens = nullptr;
        std::int32_t* experts = nullptr;
        float* weights       = nullptr;
        long long* count     = nullptr;
    };
    static constexpr int kJobMirrors = 8;
    std::array<JobMirror, kJobMirrors> mirrors{};
    // Host-function contexts. A hook runs at graph capture and its context pointer is baked
    // into the host-function node, so a context must stay valid and unchanged for every
    // replay: one address-stable context per distinct (layer, offset, tokens), shared by every
    // graph and eager round with that slice shape.
    struct SliceContext {
        Layer* entry            = nullptr;
        std::int32_t offset     = 0;
        std::int32_t tokens     = 0;
        std::int32_t ordinal    = 0;
        const JobMirror* mirror = nullptr;
        /// This slice's own handshake flag pair, or -1 for the host-function path. Assigned
        /// once, when the context is created, and never reused: a captured graph bakes the
        /// flag's *address* into its write and wait nodes, while the slot-to-context mapping
        /// the coordinator reads is host state that a replay cannot update. Keying the slot by
        /// (layer, ordinal) made every decode graph of a layer share one, so a replay of the
        /// 13-column graph could raise a flag the coordinator resolved to the 16-column
        /// context -- the host then reading a job list against the wrong width and the wrong
        /// staging columns. One slot per context is what makes the mapping a constant.
        std::int32_t handshake_slot = -1;
        // Issue order of this slice. Not a race detector: the host runs far ahead of the
        // stream, so a large gap between this and `staging_generation` is normal -- the
        // overwriting copy is only *enqueued*, and the main stream cannot reach it until the
        // combine's wait on `join_event` has retired this callback.
        std::uint64_t staged_generation = 0;
    };
    std::uint64_t staging_generation = 0;
    std::deque<SliceContext> slice_contexts;
    SliceContext& slice_context(Layer& entry, std::int32_t offset, std::int32_t tokens,
                                std::int32_t ordinal) {
        for (SliceContext& c : slice_contexts) {
            if (c.entry == &entry && c.offset == offset && c.tokens == tokens &&
                c.ordinal == ordinal) {
                return c;
            }
        }
        slice_contexts.push_back(SliceContext{&entry, offset, tokens, ordinal,
                                              &mirrors[static_cast<std::size_t>(ordinal)]});
        SliceContext& created = slice_contexts.back();
        // The last slot is the probe's; stop one short of it so the two never meet.
        if (handshake.enabled && handshake.used + 1 < handshake_slots()) {
            created.handshake_slot = static_cast<std::int32_t>(handshake.used);
            handshake.slice[handshake.used].store(&created, std::memory_order_release);
            // Publish the count last: the coordinator scans [0, used), so a slot is visible
            // only once the context it names is.
            handshake.used_published.store(++handshake.used, std::memory_order_release);
        } else if (handshake.enabled) {
            std::fprintf(stderr,
                         "expert cache: out of handshake slots at %zu round shapes; this slice "
                         "takes the host-function path\n",
                         handshake.used);
        }
        return created;
    }
    bool enabled = false;
    std::int32_t slots     = 0;
    std::int32_t scan_ring = 0; // trailing slots reserved for prefill scans (0 = plain LRU)
    void* pool_memory      = nullptr;
    void* directory_memory = nullptr;
    void* miss_memory      = nullptr;
    void* stats_memory     = nullptr;
    ops::ExpertSlotPool pool;
    ops::ExpertSlotDirectory directory;
    ops::ExpertMissList misses;
    std::vector<Layer> layer_entries;
    PendingPartial pending;

    Layer& layer(const BankedMixture& mixture) {
        if (mixture.layer < 0 || mixture.layer >= static_cast<std::int32_t>(layer_entries.size())) {
            throw std::logic_error("expert cache: the mixture's layer index is outside the "
                                   "directory (" +
                                   std::to_string(mixture.layer) + " of " +
                                   std::to_string(layer_entries.size()) + " layers)");
        }
        if (mixture.op == nullptr) {
            throw std::logic_error("expert cache: the mixture carries no kernel weights");
        }
        Layer& entry = layer_entries[static_cast<std::size_t>(mixture.layer)];
        if (entry.owner == nullptr) {
            entry.owner                     = this;
            entry.index                     = mixture.layer;
            const ops::SparseMoeWeights& op = *mixture.op;
            // The device view: each half by its own source -- the Q4 planes' base where the
            // bank requantised it, the Weight (W8 planes or the file's blocks) otherwise.
            const auto half_source = [](const Weight& weight, BankPlanes planes) {
                ops::ExpertBankHalfSource source{&weight, nullptr, nullptr};
                if (planes == BankPlanes::Q4) { source.q4_base = weight.qdata; }
                if (planes == BankPlanes::Q5) { source.q5_base = weight.qdata; }
                return source;
            };
            entry.bank = ops::expert_host_bank(
                geometry, half_source(op.routed_gate_up, mixture.gate_up_planes),
                half_source(op.routed_down, mixture.down_planes));
            // The host view of the same planes, for the CPU expert path: nothing unless the
            // caller gave the host addresses (an object that is device resident has none, and
            // the layer then runs its misses through the pool alone).
            if (mixture.host_gate_up == nullptr || mixture.host_down == nullptr) { return entry; }
            const auto host_half = [&](bool gate_up) {
                const std::byte* host       = gate_up ? mixture.host_gate_up : mixture.host_down;
                const BankPlanes planes     = gate_up ? mixture.gate_up_planes : mixture.down_planes;
                const Weight& weight        = gate_up ? op.routed_gate_up : op.routed_down;
                ops::ExpertBankFormat& fmt  = gate_up ? entry.cpu_bank.gate_up_format
                                                      : entry.cpu_bank.down_format;
                const std::byte*& codes     = gate_up ? entry.cpu_bank.gate_up_codes
                                                      : entry.cpu_bank.down_codes;
                const std::byte*& scales    = gate_up ? entry.cpu_bank.gate_up_scales
                                                      : entry.cpu_bank.down_scales;
                const std::byte*& mins      = gate_up ? entry.cpu_bank.gate_up_mins
                                                      : entry.cpu_bank.down_mins;
                const std::int64_t rows_total =
                    static_cast<std::int64_t>(geometry.experts) *
                    (gate_up ? geometry.expert_rows() : geometry.hidden);
                const std::int32_t k = gate_up ? geometry.hidden : geometry.intermediate;
                if (planes == BankPlanes::Q4) {
                    const ops::Q4BankPlanes layout = ops::q4_bank_planes(rows_total, k);
                    fmt    = ops::ExpertBankFormat::Q4G32AM;
                    codes  = host;
                    scales = host + layout.scales_offset;
                    mins   = host + layout.mins_offset;
                    return;
                }
                if (planes == BankPlanes::Q5) {
                    const ops::Q5BankPlanes layout = ops::q5_bank_planes(rows_total, k);
                    fmt    = ops::ExpertBankFormat::Q5G32AM;
                    codes  = host;
                    scales = host + layout.scales_offset;
                    mins   = host + layout.mins_offset;
                    return;
                }
                const ops::ExpertBankFormat device_format =
                    gate_up ? entry.bank.gate_up_format : entry.bank.down_format;
                if (device_format == ops::ExpertBankFormat::GgmlBlocks) {
                    // The blocks are the bank: the CPU path decodes a row at a time with the
                    // same codec the gather uses, so the host reads exactly the bytes the GGUF
                    // holds and the artifact still stores no second copy of the experts.
                    fmt    = ops::ExpertBankFormat::GgmlBlocks;
                    codes  = host;
                    scales = nullptr;
                    mins   = nullptr;
                    (gate_up ? entry.cpu_bank.gate_up_ggml : entry.cpu_bank.down_ggml) =
                        gate_up ? entry.bank.gate_up_ggml : entry.bank.down_ggml;
                    return;
                }
                // W8 planes: the plane offsets are the same in the host and device views.
                fmt    = ops::ExpertBankFormat::W8G32;
                codes  = host;
                scales = host + (static_cast<const std::byte*>(weight.scales) -
                                 static_cast<const std::byte*>(weight.qdata));
                mins   = nullptr;
            };
            host_half(true);
            host_half(false);
        }
        return entry;
    }

    // --- CPU expert split (SUROGATE_SERVE_CPU_MOE_SHARE=<fraction of misses>) ---
    std::uint32_t cpu_share_q16 = 0;
    std::int32_t cpu_max_tokens = 0;
    // Prefill rounds (wider than cpu_max_tokens) use their own share; 0 keeps the full gather.
    std::uint32_t cpu_prefill_share_q16 = 0;
    std::int32_t cpu_prefill_max_tokens = 0;
    bool auto_share            = false;
    bool share_measured        = false;
    bool prefill_share_default = false; // prefill share not given: follows the measured decode share
    std::shared_ptr<ops::CpuExpertPool> cpu_pool; // one per process: pipeline stages share the host cores
    cudaStream_t cpu_stream = nullptr; // side stream: the host round overlaps the GPU experts

    // --- GPU<->host handshake through stream memory operations ---
    //
    // The host round used to be a host function on a side stream: cudaLaunchHostFunc to
    // start it, an event for the combine to wait on. Each is a callback dispatch of some tens
    // of microseconds with the GPU front end idle, twice per MoE layer -- milliseconds per
    // token at 48 layers. Instead the stream itself raises a flag in mapped pinned memory
    // (cuStreamWriteValue64) once the slice's activations and jobs have landed, a persistent
    // coordinator thread sees it, runs the round, and stores a done flag the combine waits on
    // (cuStreamWaitValue64) -- no callback in either direction. Both are front-end operations,
    // so no SM is held while the host computes. Probed at startup, in a capture as well as
    // eagerly; the host-function path stays as the fallback (SUROGATE_SERVE_NO_MEMOP_HANDSHAKE=1
    // forces it).
    // One flag pair per (layer, slice ordinal). The values are 1 and 0, not a counter: a
    // captured graph replays the write with the value it was captured with, so the protocol
    // is raise-and-clear -- the stream raises `ready`, the coordinator clears it, runs the
    // round and raises `done`; the stream waits on `done` and clears it again itself, so the
    // next replay of the same graph waits properly. Ordering across layers needs nothing more:
    // the stream raises L+1's flag only after it has passed L's wait.
    /// Distinct round shapes a (layer, slice ordinal) may take: one graph profile per decode
    /// width, plus the eager shapes. The flags are two 8-byte words each, so being generous
    /// costs kilobytes, and the coordinator's scan is bounded by what is actually used rather
    /// than by this.
    static constexpr std::size_t kHandshakeShapesPerSlice = 24;
    std::size_t handshake_slots() const {
        return static_cast<std::size_t>(layers) * static_cast<std::size_t>(kJobMirrors) *
               kHandshakeShapesPerSlice;
    }
    struct Handshake {
        bool enabled              = false;
        unsigned long long* ready = nullptr; // pinned, mapped: [slots], the stream writes
        unsigned long long* done  = nullptr; // pinned, mapped: [slots], the host writes
        CUdeviceptr ready_device  = 0;
        CUdeviceptr done_device   = 0;
        std::vector<std::atomic<SliceContext*>> slice; // [slots], set once per context
        /// Slots handed out so far (host-side) and the count the coordinator may scan. Two
        /// variables because the context pointer must be visible before the slot is.
        std::size_t used = 0;
        std::atomic<std::size_t> used_published{0};
        std::thread coordinator;
        std::atomic<bool> stop{false};
    } handshake;
    ~Impl() {
        // The coordinator spins on flags this object owns; stop it before they go.
        handshake.stop.store(true, std::memory_order_relaxed);
        if (handshake.coordinator.joinable()) { handshake.coordinator.join(); }
        if (handshake.ready != nullptr) { (void)cudaFreeHost(handshake.ready); }
    }
    cudaStream_t fake_stream = nullptr; // SUROGATE_SERVE_CPU_MOE_FAKE_WAIT: timing-only fork/join
    cudaEvent_t fake_fork = nullptr, fake_join = nullptr;
    cudaEvent_t fork_event   = nullptr;
    cudaEvent_t join_event   = nullptr;
    cudaEvent_t copied_event = nullptr; // the slice's staging copies are done (the job list may be reused)
    std::int32_t stage_tokens = 0;      // columns the host staging holds
    void* cpu_jobs_memory = nullptr;
    ops::ExpertCpuJobList cpu_jobs;
    // Pinned host staging: activations, jobs, count, and the FP32 partial the GPU adds back.
    std::uint16_t* x_host   = nullptr;
    float* out_host         = nullptr;
    void* out_device_alias  = nullptr;
    void* jobs_host_block   = nullptr; // pinned mirror of the device job list (same carve)
    std::size_t jobs_block_bytes    = 0;
    std::int32_t* jobs_tokens_host  = nullptr;
    std::int32_t* jobs_experts_host = nullptr;
    float* jobs_weights_host        = nullptr;
    long long* jobs_count_host      = nullptr;
    // Rounds narrower than this take the whole gather. It was 4 ("below this the host round
    // trip costs more than it saves") when the host measured slower; at one user, where every
    // decode round is one token, 1 reads 20.2 against 15.3 tok/s on the board shape and 16
    // users are unmoved by it (2026-09-04, GPU 0).
    std::int32_t cpu_min_tokens = 1;
    // Widest round that still takes the decode share. SUROGATE_SERVE_CPU_MOE_DECODE_BAND
    // overrides it; see `share_for` for what it means and design/INFERENCE.md for the
    // measurement that set the default.
    std::int32_t cpu_decode_band = 0;
    std::vector<ops::CpuExpertJob> job_scratch;

    bool cpu_split_enabled() const { return cpu_pool != nullptr; }
    // The share for a round of `tokens` columns in total (0 = no split): rounds inside the
    // decode band use the decode share, wider ones the prefill share, and nothing wider than
    // the staging.
    //
    // The band is not the staging width. What decides the share is which host path a round
    // takes: one token per expert is a row-chunked GEMV, DRAM-bound, and the host relieves the
    // link; several tokens per expert is the grouped path, compute-bound far below VNNI peak,
    // and the host becomes the round's ceiling. `cpu_max_tokens` sizes the staging for the
    // widest decode-shaped round a mixed batch can bring; `cpu_decode_band` says how wide a
    // round may be and still be worth splitting.
    std::uint32_t share_for(std::int32_t tokens) const {
        if (tokens > stage_tokens) { return 0U; }
        if (tokens <= cpu_decode_band) { return cpu_share_q16; }
        return tokens <= cpu_prefill_max_tokens ? cpu_prefill_share_q16 : 0U;
    }
    // Called before the MoE op: fixes the round's split decision and share.
    void begin_round(Layer& entry, std::int32_t tokens) {
        entry.round_total  = tokens;
        entry.round_offset = 0;
        entry.round_slice  = 0;
        entry.round_split  = cpu_split_enabled() && share_for(tokens) > 0 &&
                             tokens >= cpu_min_tokens && entry.cpu_bank.gate_up_codes != nullptr;
    }

    /// Brings up the memop handshake if the driver has the operations and they survive a
    /// capture; otherwise the host-function path stays.
    void start_handshake() {
        if (std::getenv("SUROGATE_SERVE_NO_MEMOP_HANDSHAKE") != nullptr) { return; }
        if (stream_write_value64() == nullptr || stream_wait_value64() == nullptr) {
            std::fprintf(stderr, "expert cache: memop handshake unavailable (driver has no stream "
                                 "memops); host functions stay\n");
            return;
        }
        const std::size_t slots_total = handshake_slots();
        void* flags = nullptr;
        const std::size_t bytes = 2 * slots_total * sizeof(unsigned long long);
        if (cudaHostAlloc(&flags, bytes, cudaHostAllocMapped | cudaHostAllocPortable) !=
            cudaSuccess) {
            return;
        }
        std::memset(flags, 0, bytes);
        void* device = nullptr;
        CUDA_CHECK(cudaHostGetDevicePointer(&device, flags, 0));
        handshake.ready        = static_cast<unsigned long long*>(flags);
        handshake.done         = handshake.ready + slots_total;
        handshake.ready_device = reinterpret_cast<CUdeviceptr>(device);
        handshake.done_device  = handshake.ready_device + slots_total * sizeof(unsigned long long);
        handshake.slice        = std::vector<std::atomic<SliceContext*>>(slots_total);
        // Probe: write then wait on the last slot, eagerly and inside a capture, and see the
        // value arrive. A driver that refuses the memop in a capture would break every decode
        // graph, so it is tried here rather than discovered at the first prefill.
        const auto probe = [&](bool captured) {
            cudaStream_t s = nullptr;
            CUDA_CHECK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
            const CUdeviceptr slot =
                handshake.ready_device + (slots_total - 1) * sizeof(unsigned long long);
            bool ok = true;
            cudaGraph_t graph = nullptr;
            if (captured) { ok = cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal) == cudaSuccess; }
            if (ok) { ok = stream_write_value64()(s, slot, 7ULL, 0U) == CUDA_SUCCESS; }
            if (ok) { ok = stream_wait_value64()(s, slot, 7ULL, kStreamWaitValueGeq) == CUDA_SUCCESS; }
            if (captured) {
                if (cudaStreamEndCapture(s, &graph) != cudaSuccess) { ok = false; }
                if (ok) {
                    cudaGraphExec_t exec = nullptr;
                    ok = cudaGraphInstantiate(&exec, graph, 0) == cudaSuccess &&
                         cudaGraphLaunch(exec, s) == cudaSuccess;
                    if (exec != nullptr) { (void)cudaGraphExecDestroy(exec); }
                }
                if (graph != nullptr) { (void)cudaGraphDestroy(graph); }
            }
            if (ok) { ok = cudaStreamSynchronize(s) == cudaSuccess; }
            (void)cudaGetLastError();
            (void)cudaStreamDestroy(s);
            return ok && handshake.ready[slots_total - 1] == 7ULL;
        };
        const bool eager = probe(false);
        handshake.ready[slots_total - 1] = 0;
        const bool graphed = eager && probe(true);
        handshake.ready[slots_total - 1] = 0;
        if (!eager || !graphed) {
            std::fprintf(stderr, "expert cache: memop handshake probe failed (%s); host functions stay\n",
                         !eager ? "eager" : "in a capture");
            return;
        }
        handshake.enabled     = true;
        handshake.coordinator = std::thread([this] {
            unsigned idle = 0;
            while (!handshake.stop.load(std::memory_order_relaxed)) {
                bool did = false;
                const std::size_t live = handshake.used_published.load(std::memory_order_acquire);
                for (std::size_t k = 0; k < live; ++k) {
                    if (__atomic_load_n(&handshake.ready[k], __ATOMIC_ACQUIRE) == 0ULL) { continue; }
                    __atomic_store_n(&handshake.ready[k], 0ULL, __ATOMIC_RELAXED);
                    SliceContext* slice = handshake.slice[k].load(std::memory_order_acquire);
                    if (slice != nullptr) { run_cpu_round(slice); }
                    __atomic_store_n(&handshake.done[k], 1ULL, __ATOMIC_RELEASE);
                    did = true;
                }
                if (did) {
                    idle = 0;
                    continue;
                }
                if (++idle < 2000) { cpu_relax_pause(); } else { std::this_thread::yield(); }
            }
        });
        std::fprintf(stderr, "expert cache: memop handshake on: the host round starts on a stream "
                             "flag and the combine waits on one\n");
    }
    static void cpu_relax_pause() {
#if defined(__x86_64__)
        __builtin_ia32_pause();
#endif
    }

    static void fake_wait_round(void* context) {
        const int ms = *static_cast<const int*>(context);
        std::this_thread::sleep_for(std::chrono::milliseconds(ms));
    }

    static void run_cpu_round(void* context) {
        auto* slice               = static_cast<SliceContext*>(context);
        Layer* entry              = slice->entry;
        Impl& cache               = *entry->owner;
        const std::int32_t hidden = cache.geometry.hidden;
        const std::int32_t tokens = slice->tokens;
        const std::size_t column0 = static_cast<std::size_t>(slice->offset) * hidden;
        const JobMirror& mirror   = *slice->mirror;
        const long long count     = std::min<long long>(*mirror.count, cache.cpu_jobs.capacity);
        if (cache.stagecheck) { cache.stagecheck_round(*entry, slice, column0, tokens); }
        std::fill_n(cache.out_host + column0, static_cast<std::size_t>(hidden) * tokens, 0.0F);
        if (count <= 0) { return; }
        cache.job_scratch.resize(static_cast<std::size_t>(count));
        // SUROGATE_SERVE_CPU_MOE_VERIFY=1: the mirror this callback reads must describe *this*
        // slice. A round of T columns routes at most T * experts_per_token paths, and every job
        // names a column of the slice and an expert of the layer -- so a mirror overwritten by a
        // later slice, or read before its copy landed, shows up here instead of as a wrong token
        // several layers later. Off by default: it is a per-job check on the host's critical path.
        if (cpu_moe_verify()) {
            const ops::SparseMoeGeometry& geometry = cache.geometry;
            const long long ceiling = static_cast<long long>(tokens) * geometry.experts_per_token;
            long long bad           = count > ceiling ? -1 : 0;
            for (long long i = 0; bad == 0 && i < count; ++i) {
                if (mirror.tokens[i] < 0 || mirror.tokens[i] >= tokens || mirror.experts[i] < 0 ||
                    mirror.experts[i] >= geometry.experts) {
                    bad = i + 1;
                }
            }
            if (bad != 0) {
                std::fprintf(stderr,
                             "expert cache: CPU split job list is inconsistent (layer %d, slice "
                             "offset %d, tokens %d, ordinal %d, count %lld, ceiling %lld, %s)\n",
                             entry->index, slice->offset, tokens, slice->ordinal, count, ceiling,
                             bad < 0 ? "count over ceiling" : "job out of range");
            }
        }
        for (long long i = 0; i < count; ++i) {
            cache.job_scratch[static_cast<std::size_t>(i)] = {mirror.tokens[i], mirror.experts[i],
                                                              mirror.weights[i]};
        }
        ops::CpuExpertRound round{cache.x_host + column0, cache.out_host + column0, tokens,
                                  cache.job_scratch};
        cache.cpu_pool->run(entry->cpu_bank, round);
        // SUROGATE_SERVE_CPU_MOE_SELFCHECK=1: run the pool a second time and compare (a race
        // in the pool shows as a run-to-run difference far above accumulation-order noise),
        // then recompute two of the round's tokens on this thread with the reference job
        // path and compare those columns (a deterministic error in the pool shows there).
        // Off by default: it doubles the host round.
        static const bool selfcheck = std::getenv("SUROGATE_SERVE_CPU_MOE_SELFCHECK") != nullptr;
        if (selfcheck) { cache.selfcheck_round(*entry, slice, round); }
    }

    std::int64_t selfcheck_rounds     = 0;
    std::int64_t selfcheck_violations = 0;
    // SUROGATE_SERVE_CPU_MOE_STAGECHECK=1: a second copy of every slice's activations is taken on
    // the side stream at the fork (x_check) and compared with the main-stream staging in the
    // host round; after each combine the device's view of the host partial is copied back
    // (out_check) and compared with what the host wrote, in the next host round.
    const bool stagecheck = std::getenv("SUROGATE_SERVE_CPU_MOE_STAGECHECK") != nullptr;
    std::uint16_t* x_check        = nullptr;
    float* out_check              = nullptr;
    std::int32_t out_check_tokens = -1; // tokens the last combine covered; -1 = nothing pending
    std::int64_t stagecheck_rounds = 0, stagecheck_x_violations = 0, stagecheck_out_violations = 0;
    void stagecheck_round(Layer& entry, const SliceContext* slice, std::size_t column0,
                          std::int32_t tokens) {
        const std::int32_t hidden = geometry.hidden;
        ++stagecheck_rounds;
        // The previous combine's device-side view of the partial vs what the host wrote.
        if (out_check_tokens >= 0 && slice->offset == 0) {
            const std::size_t n = static_cast<std::size_t>(hidden) * out_check_tokens;
            std::size_t bad = 0, first = n;
            for (std::size_t i = 0; i < n; ++i) {
                if (out_check[i] != out_host[i]) {
                    if (first == n) { first = i; }
                    ++bad;
                }
            }
            if (bad != 0) {
                ++stagecheck_out_violations;
                std::fprintf(stderr,
                             "expert cache: stage check: the device read a host partial that differs "
                             "from what the host wrote (%zu of %zu values, first at token %zu row "
                             "%zu: device %.4g host %.4g; layer %d, %d tokens; violation %lld of "
                             "%lld rounds)\n",
                             bad, n, first / static_cast<std::size_t>(hidden),
                             first % static_cast<std::size_t>(hidden),
                             static_cast<double>(out_check[first]),
                             static_cast<double>(out_host[first]), entry.index, out_check_tokens,
                             static_cast<long long>(stagecheck_out_violations),
                             static_cast<long long>(stagecheck_rounds));
            }
            out_check_tokens = -1;
        }
        // The side-stream copy of the activations vs the main-stream staging.
        const std::size_t n = static_cast<std::size_t>(hidden) * tokens;
        std::size_t bad = 0, first = n;
        for (std::size_t i = 0; i < n; ++i) {
            if (x_check[column0 + i] != x_host[column0 + i]) {
                if (first == n) { first = i; }
                ++bad;
            }
        }
        if (bad != 0) {
            ++stagecheck_x_violations;
            std::fprintf(stderr,
                         "expert cache: stage check: the activations staged for the host differ "
                         "between the main-stream copy and the side-stream copy (%zu of %zu values, "
                         "first at token %zu; layer %d, slice offset %d, %d tokens; violation %lld "
                         "of %lld rounds)\n",
                         bad, n, first / static_cast<std::size_t>(hidden), entry.index,
                         slice->offset, tokens, static_cast<long long>(stagecheck_x_violations),
                         static_cast<long long>(stagecheck_rounds));
        }
        if (stagecheck_rounds % 2000 == 0) {
            std::fprintf(stderr,
                         "expert cache: stage check: %lld rounds, %lld activation and %lld partial "
                         "violations\n",
                         static_cast<long long>(stagecheck_rounds),
                         static_cast<long long>(stagecheck_x_violations),
                         static_cast<long long>(stagecheck_out_violations));
        }
    }

    void selfcheck_round(Layer& entry, const SliceContext* slice, const ops::CpuExpertRound& round) {
        const std::int32_t hidden  = geometry.hidden;
        const std::size_t elements = static_cast<std::size_t>(hidden) * round.tokens;
        ++selfcheck_rounds;
        std::vector<float> first(round.out, round.out + elements);
        std::fill_n(round.out, elements, 0.0F);
        cpu_pool->run(entry.cpu_bank, round);
        float scale = 1e-6F;
        for (const float v : first) { scale = std::max(scale, std::fabs(v)); }
        float worst_rerun              = 0.0F;
        std::int32_t worst_rerun_token = -1;
        for (std::size_t i = 0; i < elements; ++i) {
            const float d = std::fabs(first[i] - round.out[i]);
            if (d > worst_rerun) {
                worst_rerun       = d;
                worst_rerun_token = static_cast<std::int32_t>(i / static_cast<std::size_t>(hidden));
            }
        }
        // Reference: every token of a narrow round on one round in 8, sixteen tokens of a wide
        // round on one round in 32 (the single-thread job path is slow, and this runs on the
        // host's critical path). A per-job error at the percent level needs this coverage.
        const bool narrow          = round.tokens <= 32;
        const bool reference_round = narrow ? selfcheck_rounds % 8 == 0 : selfcheck_rounds % 32 == 0;
        std::vector<std::byte> scratch_bytes(
            reference_round ? ops::cpu_expert_scratch_bytes(geometry) + 64 : 0);
        auto* scratch = reinterpret_cast<std::byte*>(
            (reinterpret_cast<std::uintptr_t>(scratch_bytes.data()) + 63) & ~std::uintptr_t{63});
        std::vector<float> reference(static_cast<std::size_t>(hidden));
        std::vector<char> token_checked(static_cast<std::size_t>(round.tokens), 0);
        float worst_ref              = 0.0F;
        std::int32_t worst_ref_token = -1;
        int checked_count            = 0;
        const int check_limit        = narrow ? round.tokens : 16;
        for (const ops::CpuExpertJob& job : round.jobs) {
            if (!reference_round || checked_count >= check_limit) { break; }
            if (token_checked[static_cast<std::size_t>(job.token)] != 0) { continue; }
            token_checked[static_cast<std::size_t>(job.token)] = 1;
            ++checked_count;
            std::fill(reference.begin(), reference.end(), 0.0F);
            for (const ops::CpuExpertJob& other : round.jobs) {
                if (other.token != job.token) { continue; }
                ops::cpu_expert_compute_job(geometry, entry.cpu_bank, other,
                                            round.x + static_cast<std::size_t>(job.token) * hidden,
                                            reference.data(), scratch);
            }
            const float* column = round.out + static_cast<std::size_t>(job.token) * hidden;
            for (std::int32_t r = 0; r < hidden; ++r) {
                const float d = std::fabs(reference[static_cast<std::size_t>(r)] - column[r]);
                if (d > worst_ref) {
                    worst_ref       = d;
                    worst_ref_token = job.token;
                }
            }
        }
        const float tolerance = 5e-3F * scale;
        if (worst_rerun > tolerance || worst_ref > tolerance) {
            ++selfcheck_violations;
            std::fprintf(stderr,
                         "expert cache: host expert self-check violation (layer %d, slice offset %d, "
                         "tokens %d, jobs %zu): rerun differs by %.3g at token %d, reference "
                         "differs by %.3g at token %d, scale %.3g (violation %lld of %lld rounds)\n",
                         entry.index, slice->offset, round.tokens, round.jobs.size(),
                         static_cast<double>(worst_rerun), worst_rerun_token,
                         static_cast<double>(worst_ref), worst_ref_token,
                         static_cast<double>(scale), static_cast<long long>(selfcheck_violations),
                         static_cast<long long>(selfcheck_rounds));
        } else if (selfcheck_rounds % 500 == 0) {
            std::fprintf(stderr, "expert cache: host expert self-check: %lld rounds, %lld violations\n",
                         static_cast<long long>(selfcheck_rounds),
                         static_cast<long long>(selfcheck_violations));
        }
    }

    // Stages one slice of the round (columns [round_offset, round_offset + tokens) of the
    // block) and forks its host round onto the side stream; the block's combine joins the
    // last slice's event and adds the round-wide partial (v2 overlap).
    void cpu_round(Layer& entry, const Tensor& x, cudaStream_t stream) {
        const std::int32_t hidden = geometry.hidden;
        const std::int32_t tokens = static_cast<std::int32_t>(x.numel() / hidden);
        const std::int32_t offset = entry.round_offset;
        if (offset + tokens > stage_tokens) {
            throw std::logic_error("expert cache: CPU split round wider than its staging");
        }
        entry.round_tokens         = tokens;
        const std::int32_t ordinal = entry.round_slice++;
        if (ordinal >= kJobMirrors) {
            throw std::logic_error("expert cache: CPU split round has more slices than job mirrors");
        }
        SliceContext& slice     = slice_context(entry, offset, tokens, ordinal);
        slice.staged_generation = ++staging_generation;
        const std::size_t column0 = static_cast<std::size_t>(offset) * hidden;
        // The staging copies run on the main stream: they are small, and stream order then
        // guarantees the next slice's resolve cannot rewrite the job list before it is copied
        // -- without making the main stream wait behind the side stream (which would queue it
        // behind the previous layer's host round and serialise host and GPU).
        CUDA_CHECK(cudaMemcpyAsync(x_host + column0, x.data,
                                   static_cast<std::size_t>(hidden) * tokens * sizeof(std::uint16_t),
                                   cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaMemcpyAsync(slice.mirror->block, cpu_jobs_memory, jobs_block_bytes,
                                   cudaMemcpyDeviceToHost, stream));
        entry.round_offset = offset + tokens;
        int partial_device = -1;
        CUDA_CHECK(cudaGetDevice(&partial_device));
        if (handshake.enabled && !stagecheck && slice.handshake_slot >= 0) {
            // The slot belongs to this slice alone and was published when its context was
            // created, so the address the stream raises and the context the coordinator reads
            // agree whether this is an eager round or the replay of a graph captured long ago.
            const auto k = static_cast<std::size_t>(slice.handshake_slot);
            const CUresult rc = stream_write_value64()(
                stream, handshake.ready_device + k * sizeof(unsigned long long), 1ULL, 0U);
            if (rc != CUDA_SUCCESS) { throw std::runtime_error("expert cache: cuStreamWriteValue64 failed"); }
            pending = PendingPartial{static_cast<const float*>(out_device_alias), nullptr,
                                     partial_device,
                                     handshake.done_device + k * sizeof(unsigned long long)};
            return;
        }
        // Fork only the host function onto the side stream so it overlaps the GPU experts.
        CUDA_CHECK(cudaEventRecord(fork_event, stream));
        CUDA_CHECK(cudaStreamWaitEvent(cpu_stream, fork_event, 0));
        if (stagecheck && x_check != nullptr) {
            CUDA_CHECK(cudaMemcpyAsync(x_check + column0, x.data,
                                       static_cast<std::size_t>(hidden) * tokens * sizeof(std::uint16_t),
                                       cudaMemcpyDeviceToHost, cpu_stream));
        }
        CUDA_CHECK(cudaLaunchHostFunc(cpu_stream, &Impl::run_cpu_round, &slice));
        CUDA_CHECK(cudaEventRecord(join_event, cpu_stream));
        pending = PendingPartial{static_cast<const float*>(out_device_alias), join_event,
                                 partial_device};
    }

    // SUROGATE_SERVE_CPU_MOE_SHADOW=<n>: every n-th split round is recomputed entirely on the
    // GPU (this hook resolves without a host share, so every miss is gathered) into a shadow
    // plane, and the split's GPU part plus the host partial is compared with it per token.
    std::int64_t shadow_rounds = 0, shadow_checked = 0, shadow_violations = 0;
    static void resolve_round_shadow(void* context, const Tensor& ids, const Tensor& /*alpha*/,
                                     const Tensor& /*x*/, Tensor& /*destination*/,
                                     cudaStream_t stream) {
        auto* entry     = static_cast<Layer*>(context);
        Impl& cache     = *entry->owner;
        const bool scan = entry->round_total > 32;
        ops::expert_slot_resolve(ids, entry->index, cache.directory, cache.misses, stream, scan);
        ops::expert_slot_gather(entry->bank, cache.misses, cache.pool, stream);
    }

    static void resolve_round(void* context, const Tensor& ids, const Tensor& alpha,
                              const Tensor& x, Tensor& destination, cudaStream_t stream) {
        auto* entry = static_cast<Layer*>(context);
        Impl& cache = *entry->owner;
        // The split decision is per round (begin_round); every slice of the round follows it.
        const bool split = entry->round_split && x.data != nullptr && destination.data != nullptr;
        const std::uint32_t share = split ? cache.share_for(entry->round_total) : 0U;
        // Wide rounds are prompt scans: their misses go to the directory's scan ring so the
        // decode working set stays resident (design/INFERENCE.md, scan resistance).
        const bool scan = entry->round_total > 32;
        if (split) {
            ops::expert_slot_resolve(ids, alpha, entry->index, cache.directory, cache.misses,
                                     &cache.cpu_jobs, share, cache.geometry.experts_per_token,
                                     stream, scan);
        } else {
            ops::expert_slot_resolve(ids, entry->index, cache.directory, cache.misses, stream,
                                     scan);
        }
        gather_probe().begin(stream);
        ops::expert_slot_gather(entry->bank, cache.misses, cache.pool, stream);
        gather_probe().end(stream);
        gather_probe().tick_gather();
        if (split) { cache.cpu_round(*entry, x, stream); }
        // SUROGATE_SERVE_CPU_MOE_FAKE_WAIT=<ms>: with the split off, still fork a host function
        // that sleeps for <ms> and make the combine wait for it -- the host split's stream
        // timing without its data. Tells a data fault in the host path from a latent race
        // elsewhere that the split's idle gaps expose.
        static const int fake_wait_ms = [] {
            const char* raw = std::getenv("SUROGATE_SERVE_CPU_MOE_FAKE_WAIT");
            return raw != nullptr && *raw != '\0' ? std::atoi(raw) : 0;
        }();
        if (!split && fake_wait_ms > 0) {
            if (cache.fake_stream == nullptr) {
                CUDA_CHECK(cudaStreamCreateWithFlags(&cache.fake_stream, cudaStreamNonBlocking));
                CUDA_CHECK(cudaEventCreateWithFlags(&cache.fake_fork, cudaEventDisableTiming));
                CUDA_CHECK(cudaEventCreateWithFlags(&cache.fake_join, cudaEventDisableTiming));
            }
            CUDA_CHECK(cudaEventRecord(cache.fake_fork, stream));
            CUDA_CHECK(cudaStreamWaitEvent(cache.fake_stream, cache.fake_fork, 0));
            CUDA_CHECK(cudaLaunchHostFunc(cache.fake_stream, &Impl::fake_wait_round,
                                          const_cast<int*>(&fake_wait_ms)));
            CUDA_CHECK(cudaEventRecord(cache.fake_join, cache.fake_stream));
            CUDA_CHECK(cudaStreamWaitEvent(stream, cache.fake_join, 0));
        }
        if (cache.stats_every > 0) { cache.record_stats(stream); }
        if (cache.directory_verify) { cache.verify_directory(ids, entry->index, stream); }
    }

    // SUROGATE_SERVE_SLOT_DIRECTORY_VERIFY=1: after every resolve, read the directory back and
    // check it is a bijection (slot_of_expert and expert_of_slot agree both ways) and that every
    // expert this layer routed to is either resident or stamped for the host this round. A
    // synchronous eager-mode diagnostic; a violation is printed once per layer-round.
    const bool directory_verify = std::getenv("SUROGATE_SERVE_SLOT_DIRECTORY_VERIFY") != nullptr;
    std::int64_t directory_violations = 0;
    void verify_directory(const Tensor& ids, int layer_index, cudaStream_t stream) {
        cudaStreamCaptureStatus capture = cudaStreamCaptureStatusNone;
        if (cudaStreamIsCapturing(stream, &capture) != cudaSuccess ||
            capture != cudaStreamCaptureStatusNone) {
            return;
        }
        const std::int32_t experts = directory.experts;
        const std::int32_t dir_layers = directory.layers;
        const auto slot_count = static_cast<std::int32_t>(directory.expert_of_slot.ne[0]);
        std::vector<int> table(static_cast<std::size_t>(dir_layers) * experts);
        std::vector<int> owners(static_cast<std::size_t>(slot_count));
        std::vector<unsigned> cpu_stamp(static_cast<std::size_t>(experts));
        std::vector<int> routed(static_cast<std::size_t>(ids.numel()));
        unsigned round = 0;
        CUDA_CHECK(cudaMemcpyAsync(table.data(), directory.slot_of_expert.data,
                                   table.size() * sizeof(int), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaMemcpyAsync(owners.data(), directory.expert_of_slot.data,
                                   owners.size() * sizeof(int), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaMemcpyAsync(cpu_stamp.data(), directory.cpu_round.data,
                                   cpu_stamp.size() * sizeof(unsigned), cudaMemcpyDeviceToHost,
                                   stream));
        CUDA_CHECK(cudaMemcpyAsync(routed.data(), ids.data, routed.size() * sizeof(int),
                                   cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaMemcpyAsync(&round, directory.round.data, sizeof(round),
                                   cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        std::int64_t bad_forward = 0, bad_backward = 0, unserved = 0;
        for (std::size_t f = 0; f < table.size(); ++f) {
            const int slot = table[f];
            if (slot < 0) { continue; }
            if (slot >= slot_count || owners[static_cast<std::size_t>(slot)] != static_cast<int>(f)) {
                ++bad_forward;
            }
        }
        for (std::int32_t s = 0; s < slot_count; ++s) {
            const int flat = owners[static_cast<std::size_t>(s)];
            if (flat < 0) { continue; }
            if (flat >= static_cast<int>(table.size()) || table[static_cast<std::size_t>(flat)] != s) {
                ++bad_backward;
            }
        }
        for (const int expert : routed) {
            if (expert < 0 || expert >= experts) { continue; }
            const std::size_t flat = static_cast<std::size_t>(layer_index) * experts + expert;
            if (table[flat] < 0 && cpu_stamp[static_cast<std::size_t>(expert)] != round) {
                ++unserved;
            }
        }
        if (bad_forward != 0 || bad_backward != 0 || unserved != 0) {
            ++directory_violations;
            std::fprintf(stderr,
                         "expert cache: slot directory violation at layer %d round %u: %lld forward, "
                         "%lld backward, %lld routed experts neither resident nor on the host "
                         "(violation %lld)\n",
                         layer_index, round, static_cast<long long>(bad_forward),
                         static_cast<long long>(bad_backward), static_cast<long long>(unserved),
                         static_cast<long long>(directory_violations));
        }
    }

    // Hit-rate readout (SUROGATE_SERVE_EXPERT_STATS=<rounds>): the counters are accumulated
    // on the device by the resolve kernel and read back exactly. A "round" is one layer's
    // resolve. Every rate is over distinct experts asked for, the population the cache
    // answers; `PCIe` is what crossed the bus, `host` what the CPU split absorbed, and
    // together they are the misses. The readout runs on the host, so it prints on eager
    // rounds; under CUDA graphs the totals accumulate and appear at the next uncaptured
    // resolve -- deferred, not lost.
    std::int64_t stats_every  = 0;
    std::int64_t stats_rounds = 0;
    std::array<long long, ops::kExpertSlotStatCount> stats_last{};
    void record_stats(cudaStream_t stream) {
        if (stats_every <= 0 || directory.stats.data == nullptr) { return; }
        cudaStreamCaptureStatus capture = cudaStreamCaptureStatusNone;
        if (cudaStreamIsCapturing(stream, &capture) != cudaSuccess ||
            capture != cudaStreamCaptureStatusNone) {
            return;
        }
        if (++stats_rounds % stats_every != 0) { return; }
        std::array<long long, ops::kExpertSlotStatCount> counters{};
        CUDA_CHECK(cudaMemcpyAsync(counters.data(), directory.stats.data, sizeof(counters),
                                   cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        const auto report = [](const char* label,
                               const std::array<long long, ops::kExpertSlotStatCount>& c) {
            const double rounds    = static_cast<double>(c[ops::kExpertSlotStatRounds]);
            const double lookups   = static_cast<double>(c[ops::kExpertSlotStatLookups]);
            const double distinct  = static_cast<double>(c[ops::kExpertSlotStatDistinct]);
            const double resident  = static_cast<double>(c[ops::kExpertSlotStatResident]);
            const double gathered  = static_cast<double>(c[ops::kExpertSlotStatGathered]);
            const double host_side = static_cast<double>(c[ops::kExpertSlotStatHostRouted]);
            if (rounds <= 0.0 || distinct <= 0.0) { return; }
            std::fprintf(stderr,
                         "expert cache %s rounds=%lld distinct/round=%.1f (of %.1f paths) hit "
                         "%.1f%% miss %.1f%% = PCIe %.1f%% + host %.1f%%\n",
                         label, static_cast<long long>(c[ops::kExpertSlotStatRounds]),
                         distinct / rounds, lookups / rounds, 100.0 * resident / distinct,
                         100.0 * (gathered + host_side) / distinct, 100.0 * gathered / distinct,
                         100.0 * host_side / distinct);
        };
        std::array<long long, ops::kExpertSlotStatCount> window{};
        for (int i = 0; i < ops::kExpertSlotStatCount; ++i) {
            window[i] = counters[i] - stats_last[i];
        }
        report("window", window);
        report("total ", counters);
        stats_last = counters;
    }

    /// The pool, directory, miss list and -- when a share is configured -- the CPU split, sized
    /// from what the run configured for this device and what the card has free.
    void create(int device) {
        // --expert-slots N wins when set; otherwise the environment knob; 0 keeps zero-copy.
        long requested = 0;
        if (auto configured = configured_expert_slots().find(device);
            configured != configured_expert_slots().end() && configured->second > 0) {
            requested = static_cast<long>(configured->second);
        } else if (const char* raw = std::getenv("SUROGATE_SERVE_EXPERT_SLOTS");
                   raw != nullptr && *raw != '\0') {
            requested = std::strtol(raw, nullptr, 10);
        }
        if (requested <= 0) {
            // No slot count asked for. One layer's experts is the floor the resolve needs, and
            // it is also what the pool used to settle on -- which left most of a card idle and
            // cost more than half the decode rate, because every other expert then crossed
            // PCIe on the token that wanted it. Take what is free instead: half of it, and
            // never into what the runtime's own floor needs (with --kv-capacity auto that floor
            // is a full-context request's pages plus the prefill workspaces), stopping at the
            // whole model. Half, because this runs before the KV cache and the round's
            // workspaces are allocated, so the reading overstates what the pool may take.
            const std::size_t per_slot = ops::expert_slot_pool_bytes(geometry, 1);
            const std::size_t free     = device_free_bytes(device);
            std::size_t floor          = 0;
            if (auto it = configured_runtime_floor().find(device);
                it != configured_runtime_floor().end()) {
                floor = it->second;
            }
            constexpr std::size_t kMargin = kPoolMarginBytes;
            const std::size_t after_floor = free > floor + kMargin ? free - floor - kMargin : 0;
            const std::size_t budget      = std::min(free / 2, after_floor);
            const auto whole_model =
                static_cast<long>(layers) * static_cast<long>(geometry.experts);
            requested = per_slot == 0 ? 0 : static_cast<long>(budget / per_slot);
            requested = std::min(requested, whole_model);
            if (requested < geometry.experts && per_slot != 0) {
                // The margin exists for what the load and the runtime carve beyond their
                // projections; a pool of one layer's experts is the difference between the
                // split running and every miss crossing PCIe -- or, with a Q4 bank, between
                // running and refusing. When the minimum pool fits inside half the margin,
                // it is worth more than that half.
                const std::size_t minimum = per_slot * static_cast<std::size_t>(geometry.experts);
                const std::size_t within_margin =
                    free > floor + kMargin / 2 ? free - floor - kMargin / 2 : 0;
                if (within_margin >= minimum) {
                    requested = geometry.experts;
                    std::fprintf(stderr,
                                 "expert cache: the automatic pool takes one layer's %d experts "
                                 "(%.1f GiB) out of its margin; %.1f GiB free, %.1f GiB floor\n",
                                 geometry.experts,
                                 static_cast<double>(minimum) / (1024.0 * 1024.0 * 1024.0),
                                 static_cast<double>(free) / (1024.0 * 1024.0 * 1024.0),
                                 static_cast<double>(floor) / (1024.0 * 1024.0 * 1024.0));
                }
            }
            if (requested < geometry.experts) {
                // Said out loud: without a pool every banked expert crosses PCIe on the token
                // that wants it, and a run that expected the split would otherwise only see
                // the rate.
                std::fprintf(stderr,
                             "expert cache: no pool on device %d: %.1f GiB free, %.1f GiB "
                             "runtime floor, leaves %.1f GiB for a pool that needs %.1f GiB "
                             "for one layer's %d experts; the banked experts are read over "
                             "PCIe in place (--expert-slots %d asks for that pool regardless)\n",
                             device, static_cast<double>(free) / (1024.0 * 1024.0 * 1024.0),
                             static_cast<double>(floor) / (1024.0 * 1024.0 * 1024.0),
                             static_cast<double>(budget) / (1024.0 * 1024.0 * 1024.0),
                             static_cast<double>(per_slot) * geometry.experts /
                                 (1024.0 * 1024.0 * 1024.0),
                             geometry.experts, geometry.experts);
                return;
            }
        }
        // A round can touch every expert of a layer, and resolve must never leave a routed
        // expert unmapped, so the pool holds at least one layer's worth of experts.
        slots = std::max(static_cast<std::int32_t>(requested), geometry.experts);
        const std::size_t pool_bytes = ops::expert_slot_pool_bytes(geometry, slots);
        const std::size_t dir_bytes =
            ops::expert_slot_directory_bytes(layers, geometry.experts, slots);
        // A round can miss at most one whole layer's expert set.
        const std::size_t miss_bytes = ops::expert_miss_list_bytes(geometry.experts);
        // Said before the allocation, not after: reserving this much is seconds of silence,
        // and a reader watching a stalled console should know what the engine is waiting on.
        std::fprintf(stderr, "expert cache: reserving a %.1f GiB expert slot pool (%d slots)...\n",
                     static_cast<double>(pool_bytes) / (1024.0 * 1024.0 * 1024.0), slots);
        CUDA_CHECK(cudaMalloc(&pool_memory, pool_bytes));
        CUDA_CHECK(cudaMalloc(&directory_memory, dir_bytes));
        CUDA_CHECK(cudaMalloc(&miss_memory, miss_bytes));
        pool = ops::create_expert_slot_pool(geometry, slots, pool_memory);
        // Scan resistance: reserve one expert-set of trailing slots for prefill scans when the
        // pool cannot hold every expert of its layers but is comfortably bigger than one scan.
        // Below that, plain LRU (a ring would eat half a tiny pool for no stable set to
        // protect). The ring must be at least one expert set and at most half the pool, so it
        // needs a pool of two expert sets before it fits at all.
        const auto whole_model = static_cast<std::int64_t>(layers) * geometry.experts;
        scan_ring = (static_cast<std::int64_t>(slots) < whole_model &&
                     slots >= geometry.experts * 2 &&
                     std::getenv("SUROGATE_SERVE_NO_SCAN_RING") == nullptr)
                        ? geometry.experts
                        : 0;
        directory = ops::create_expert_slot_directory(layers, geometry.experts, slots, scan_ring,
                                                      directory_memory, nullptr);
        misses    = ops::create_expert_miss_list(geometry.experts, miss_memory);
        CUDA_CHECK(cudaStreamSynchronize(nullptr));
        layer_entries.resize(static_cast<std::size_t>(layers));
        enabled = true;
        if (const char* stats = std::getenv("SUROGATE_SERVE_EXPERT_STATS");
            stats != nullptr && *stats != '\0') {
            stats_every = std::strtol(stats, nullptr, 10);
            if (stats_every > 0) {
                // The counters live on the device and the resolve kernel increments them, so
                // a captured decode keeps counting through every replay; the host only reads.
                const std::size_t bytes = sizeof(long long) * ops::kExpertSlotStatCount;
                CUDA_CHECK(cudaMalloc(&stats_memory, bytes));
                CUDA_CHECK(cudaMemset(stats_memory, 0, bytes));
                directory.stats = Tensor(stats_memory, DType::I64, {ops::kExpertSlotStatCount});
            }
        }
        double fraction  = 0.0;
        bool auto_share_ = false;
        if (auto configured = configured_cpu_share().find(device);
            configured != configured_cpu_share().end() && configured->second != 0.0F) {
            if (configured->second < 0.0F) {
                auto_share_ = true;
                fraction    = 0.7; // placeholder until prepare_split measures the rates
            } else {
                fraction = static_cast<double>(configured->second);
            }
        } else if (const char* share = std::getenv("SUROGATE_SERVE_CPU_MOE_SHARE");
                   share != nullptr && *share != '\0') {
            if (std::string(share) == "auto") {
                auto_share_ = true;
                fraction    = 0.7;
            } else {
                fraction = std::strtod(share, nullptr);
            }
        }
        // Prefill share: -1 (unset) -> 0. The prefill split is a measured loss at every chunk
        // width on this host (2026-08-30, 28k prompt, x16 card): the host GEMM runs at ~16 % of
        // VNNI peak and every layer's combine waits on its host tail, so the split caps the
        // round at the CPU's pace. The decode split is a separate decision and keeps its own
        // share: at one token per lane the gather is misses-only and the host genuinely
        // relieves it. An explicit --cpu-moe-prefill-share still turns the prefill split on.
        double prefill_fraction     = -1.0;
        std::uint32_t prefill_chunk = 0;
        if (auto configured = configured_cpu_prefill().find(device);
            configured != configured_cpu_prefill().end()) {
            prefill_fraction = static_cast<double>(configured->second.first);
            prefill_chunk    = configured->second.second;
        }
        if (const char* share = std::getenv("SUROGATE_SERVE_CPU_MOE_PREFILL_SHARE");
            share != nullptr && *share != '\0') {
            prefill_fraction = std::strtod(share, nullptr);
            if (prefill_chunk == 0) { prefill_chunk = 2048; }
        }
        const bool prefill_default = prefill_fraction < 0.0;
        if (prefill_default) { prefill_fraction = 0.0; }
        if (prefill_chunk == 0) { prefill_chunk = 2048; }
        if (fraction > 0.0 || prefill_fraction > 0.0) {
            cpu_share_q16 = static_cast<std::uint32_t>(std::min(1.0, std::max(0.0, fraction)) * 65536.0);
            cpu_max_tokens = 64; // the widest decode-shaped round the staging must hold
            cpu_decode_band = kDefaultDecodeBand;
            if (const char* band = std::getenv("SUROGATE_SERVE_CPU_MOE_DECODE_BAND");
                band != nullptr && *band != '\0') {
                cpu_decode_band = static_cast<std::int32_t>(std::strtol(band, nullptr, 10));
            }
            cpu_decode_band = std::clamp(cpu_decode_band, 1, cpu_max_tokens);
            if (prefill_fraction > 0.0 && prefill_chunk > 0) {
                cpu_prefill_share_q16 = static_cast<std::uint32_t>(std::min(1.0, prefill_fraction) * 65536.0);
                cpu_prefill_max_tokens = static_cast<std::int32_t>(prefill_chunk);
            }
            const std::int32_t hidden = geometry.hidden;
            // Staging covers the widest round: a prefill chunk plus the decode lanes of a mixed
            // round (or just the decode lanes without a prefill share).
            stage_tokens = cpu_prefill_max_tokens > 0 ? cpu_prefill_max_tokens + 256 : cpu_max_tokens;
            // Jobs per slice: a slice is at most a prefill chunk (or the decode lanes) wide.
            const std::int32_t capacity =
                std::max(cpu_max_tokens, cpu_prefill_max_tokens) * geometry.experts_per_token;
            CUDA_CHECK(cudaMalloc(&cpu_jobs_memory, ops::expert_cpu_job_list_bytes(capacity)));
            cpu_jobs = ops::create_expert_cpu_job_list(capacity, cpu_jobs_memory);
            // The staging buffers are what this device DMAs through every round, so they go on
            // its own node rather than across both (core/numa.h). The expert bank, which every
            // core reads, is interleaved instead.
            {
                const ScopedMemoryPolicy placement = ScopedMemoryPolicy::for_device(device);
                CUDA_CHECK(cudaHostAlloc(reinterpret_cast<void**>(&x_host),
                                         static_cast<std::size_t>(hidden) * stage_tokens * sizeof(std::uint16_t),
                                         cudaHostAllocPortable));
                CUDA_CHECK(cudaHostAlloc(reinterpret_cast<void**>(&out_host),
                                         static_cast<std::size_t>(hidden) * stage_tokens * sizeof(float),
                                         cudaHostAllocMapped | cudaHostAllocPortable));
            }
            CUDA_CHECK(cudaHostGetDevicePointer(&out_device_alias, out_host, 0));
            if (stagecheck) {
                // Allocated here, not lazily: the first host round may run under graph capture.
                CUDA_CHECK(cudaHostAlloc(reinterpret_cast<void**>(&x_check),
                                         static_cast<std::size_t>(hidden) * stage_tokens * sizeof(std::uint16_t),
                                         cudaHostAllocPortable));
                CUDA_CHECK(cudaHostAlloc(reinterpret_cast<void**>(&out_check),
                                         static_cast<std::size_t>(hidden) * stage_tokens * sizeof(float),
                                         cudaHostAllocPortable));
            }
            jobs_block_bytes = ops::expert_cpu_job_list_bytes(capacity);
            for (auto& m : mirrors) {
                CUDA_CHECK(cudaHostAlloc(&m.block, jobs_block_bytes, cudaHostAllocPortable));
                // Same carve as the device list, so field offsets match after the block copy.
                const ops::ExpertCpuJobList view = ops::create_expert_cpu_job_list(capacity, m.block);
                m.tokens  = static_cast<std::int32_t*>(view.tokens.data);
                m.experts = static_cast<std::int32_t*>(view.experts.data);
                m.weights = static_cast<float*>(view.weights.data);
                m.count   = static_cast<long long*>(view.count.data);
            }
            jobs_host_block   = mirrors[0].block;
            jobs_tokens_host  = mirrors[0].tokens;
            jobs_experts_host = mirrors[0].experts;
            jobs_weights_host = mirrors[0].weights;
            jobs_count_host   = mirrors[0].count;
            if (auto configured = configured_cpu_min_tokens().find(device);
                configured != configured_cpu_min_tokens().end() && configured->second > 0) {
                cpu_min_tokens = static_cast<std::int32_t>(configured->second);
            } else if (const char* min = std::getenv("SUROGATE_SERVE_CPU_MOE_MIN_TOKENS");
                       min != nullptr && *min != '\0') {
                cpu_min_tokens = static_cast<std::int32_t>(std::strtol(min, nullptr, 10));
            }
            ops::CpuExpertPoolOptions pool_options;
            if (const char* threads = std::getenv("SUROGATE_SERVE_CPU_MOE_THREADS");
                threads != nullptr && *threads != '\0') {
                pool_options.threads = static_cast<std::uint32_t>(std::strtoul(threads, nullptr, 10));
            }
            {
                // One pool per process, or -- for pipeline stages -- one per NUMA node pinned
                // to that node's physical cores, so stages on different sockets run their host
                // rounds concurrently on their own cores and memory.
                static std::mutex pool_mutex;
                static std::map<std::string, std::weak_ptr<ops::CpuExpertPool>> pools;
                int node = -1;
                if (configured_cpu_pool_per_socket()) {
                    node = device_numa_node(device);
                    if (node >= 0) {
                        pool_options.cpus = node_physical_cpus(node);
                        if (pool_options.cpus.empty()) { node = -1; }
                    }
                }
                std::lock_guard<std::mutex> lock(pool_mutex);
                const std::string key = pool_key(node, geometry);
                cpu_pool              = pools[key].lock();
                if (cpu_pool == nullptr) {
                    cpu_pool = std::make_shared<ops::CpuExpertPool>(geometry, pool_options);
                    // One coordinator per host pool: the cache that creates the pool runs the
                    // memop handshake, and a stage that shares the pool keeps the host-function
                    // path rather than adding a spinning thread per stage to the pool's cores.
                    start_handshake();
                    pools[key] = cpu_pool;
                    if (node >= 0) {
                        std::fprintf(stderr, "expert cache: host expert pool for NUMA node %d: %zu threads\n",
                                     node, pool_options.cpus.size());
                    }
                }
            }
            CUDA_CHECK(cudaStreamCreateWithFlags(&cpu_stream, cudaStreamNonBlocking));
            CUDA_CHECK(cudaEventCreateWithFlags(&fork_event, cudaEventDisableTiming));
            CUDA_CHECK(cudaEventCreateWithFlags(&join_event, cudaEventDisableTiming));
            CUDA_CHECK(cudaEventCreateWithFlags(&copied_event, cudaEventDisableTiming));
            auto_share            = auto_share_;
            prefill_share_default = prefill_default;
            std::fprintf(stderr,
                         "expert cache: CPU expert split enabled: %.0f%% of the misses of rounds "
                         "up to %d columns, %.0f%% of prefill misses (up to %d columns) on %u host "
                         "threads%s\n",
                         100.0 * fraction, cpu_decode_band, 100.0 * prefill_fraction,
                         cpu_prefill_max_tokens, cpu_pool->threads(),
                         auto_share_ ? " (auto: measured at startup)" : "");
        }
        std::fprintf(stderr,
                     "expert cache: ready: %d slots (%.1f GiB pool) over %d layers of %d experts, "
                     "scan ring %d\n",
                     slots, static_cast<double>(pool_bytes) / (1024.0 * 1024.0 * 1024.0), layers,
                     geometry.experts, scan_ring);
        if (const std::string numa = numa_policy_description(); !numa.empty()) {
            std::fprintf(stderr, "expert cache: %s\n", numa.c_str());
        }
    }
};

// -------------------------------------------------------------------------------------------
// Public surface
// -------------------------------------------------------------------------------------------

ExpertCache::ExpertCache() : impl_(std::make_unique<Impl>()) {}
ExpertCache::~ExpertCache() = default;

ExpertCache& ExpertCache::for_current_device(const ops::SparseMoeGeometry& geometry,
                                             std::int32_t layers) {
    // One cache per (device, mixture shape): a second model on the same card gets its own pool
    // rather than a refusal or a share of a pool cut for another expert size. Every layer's
    // round asks twice, so the lookup is a scan of a handful of entries, not a keyed map.
    struct Entry {
        int device;
        std::unique_ptr<ExpertCache> cache;
    };
    static std::mutex mutex;
    static std::vector<Entry> registry;
    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    std::lock_guard<std::mutex> lock(mutex);
    for (const Entry& entry : registry) {
        if (entry.device == device && entry.cache->impl_->geometry == geometry &&
            entry.cache->impl_->layers == layers) {
            // A cache that could not size its pool is not a verdict for the device's life: a
            // pipeline preflight asks for it once per candidate placement, and the first
            // candidate is the one that leaves the least room. Nothing was allocated on that
            // path, so the derivation runs again against whatever floor is configured now.
            if (!entry.cache->impl_->enabled) {
                std::lock_guard<std::mutex> config(config_mutex());
                entry.cache->impl_->create(device);
            }
            return *entry.cache;
        }
    }
    if (layers <= 0) { throw std::invalid_argument("expert cache: a model has at least one layer"); }
    std::unique_ptr<ExpertCache> created(new ExpertCache());
    created->impl_->geometry = geometry;
    created->impl_->layers   = layers;
    {
        std::lock_guard<std::mutex> config(config_mutex());
        created->impl_->create(device);
    }
    ExpertCache& cache = *created;
    registry.push_back(Entry{device, std::move(created)});
    return cache;
}

void ExpertCache::configure(const EngineOptions& options, std::size_t runtime_floor_bytes,
                            std::int32_t mixture_experts) {
    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    std::lock_guard<std::mutex> lock(config_mutex());
    configured_expert_slots()[device]  = options.expert_slots;
    configured_runtime_floor()[device] = runtime_floor_bytes;
    {
        // A pipeline stage whose pool holds (nearly) all of its layers' experts gains nothing
        // from the CPU split -- every layer would still pay a host round trip -- so the split
        // is off on a stage unless the share was given explicitly. (At 8 stages a 68 %-resident
        // stage decodes faster without it: 514 vs 395 tok/s at 64 users.)
        float share       = options.cpu_moe_share;
        const bool staged = options.pipeline_stage_first != 0 || options.pipeline_stage_last != 0;
        if (staged && share < 0.0F && options.expert_slots > 0) {
            const int stage_layers = options.pipeline_stage_last - options.pipeline_stage_first;
            const auto stage_experts =
                static_cast<std::uint64_t>(stage_layers) * static_cast<std::uint64_t>(mixture_experts);
            share = 0.0F;
            std::fprintf(stderr,
                         "expert cache: pipeline stage holds %u of %llu experts; CPU split off "
                         "(pass --cpu-moe-share to enable)\n",
                         options.expert_slots, static_cast<unsigned long long>(stage_experts));
        }
        configured_cpu_share()[device] = share;
    }
    configured_cpu_min_tokens()[device] = options.cpu_moe_min_tokens;
    configured_cpu_prefill()[device]    = {options.cpu_moe_prefill_share, options.prefill_chunk};
    configured_cpu_pool_per_socket()    = options.cpu_moe_pool_per_socket;
}

void ExpertCache::configure_derived_reserve(std::size_t bytes) {
    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    std::lock_guard<std::mutex> lock(config_mutex());
    configured_derived_reserve()[device] = bytes;
}

std::size_t ExpertCache::derived_reserve() {
    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    std::lock_guard<std::mutex> lock(config_mutex());
    const auto it = configured_derived_reserve().find(device);
    return it == configured_derived_reserve().end() ? 0 : it->second;
}

void ExpertCache::configure_pool_floor(std::size_t bytes) {
    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    std::lock_guard<std::mutex> lock(config_mutex());
    configured_pool_floor()[device] = bytes;
}

std::size_t ExpertCache::pool_floor() {
    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    std::lock_guard<std::mutex> lock(config_mutex());
    const auto it = configured_pool_floor().find(device);
    return it == configured_pool_floor().end() ? 0 : it->second;
}

std::size_t ExpertCache::pool_floor_bytes(const ops::SparseMoeGeometry& geometry,
                                          std::int32_t layers, std::uint32_t requested_slots) {
    // The derivation below never settles under one layer's experts, and an explicit request
    // is honoured as given: the larger of the two is what the placement must leave.
    const std::int32_t slots =
        std::max(static_cast<std::int32_t>(std::min<std::uint32_t>(requested_slots, 1U << 30)),
                 geometry.experts);
    return ops::expert_slot_pool_bytes(geometry, slots) +
           ops::expert_slot_directory_bytes(layers, geometry.experts, slots) +
           ops::expert_miss_list_bytes(geometry.experts) + kPoolMarginBytes;
}

void ExpertCache::configure_load_staging(std::size_t bytes) {
    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    std::lock_guard<std::mutex> lock(config_mutex());
    configured_load_staging()[device] = bytes;
}

std::size_t ExpertCache::load_staging() {
    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    std::lock_guard<std::mutex> lock(config_mutex());
    const auto it = configured_load_staging().find(device);
    return it == configured_load_staging().end() ? 0 : it->second;
}

bool ExpertCache::enabled() const noexcept { return impl_->enabled; }

void ExpertCache::run(const BankedMixture& mixture, const Tensor& hidden, Tensor& destination,
                      WorkspaceArena& workspace, cudaStream_t stream) {
    Impl& cache = *impl_;
    if (!cache.enabled) { throw std::logic_error("expert cache: run on a disabled cache"); }
    if (mixture.op == nullptr) {
        throw std::logic_error("expert cache: the mixture carries no kernel weights");
    }
    // A layer whose experts are device resident may still run through the pool (its misses are
    // then device-to-device copies and its host split never engages); a caller that would
    // rather keep such a layer on the plain route checks `banked()` first.
    const std::int32_t tokens       = hidden.ne[1];
    const ops::SparseMoeWeights& op = *mixture.op;
    // The kernels run over the pool, which is W8 whatever the bank holds; the bank's own type
    // is what the plain route would have used, so the leaf is cut for the larger of the two.
    const std::size_t bytes = std::max(
        ops::sparse_moe_workspace_capacity_bytes(cache.geometry, op.routed_gate_up.qtype,
                                                 op.routed_down.qtype, tokens, tokens),
        ops::sparse_moe_workspace_capacity_bytes(cache.geometry, QType::W8G32_F16S,
                                                 QType::W8G32_F16S, tokens, tokens));
    auto scope               = workspace.scope();
    const DeviceSpan storage = workspace.alloc_bytes(bytes);
    WorkspaceArena leaf(storage);
    // Routed experts come from the device slot pool: the round hook resolves the routing
    // against the directory and gathers the misses from the host bank before the expert
    // kernels run; the kernels read the pool through the layer's slot table.
    Impl::Layer& layer = cache.layer(mixture);
    const ops::SparseMoeWeights pooled =
        ops::expert_slot_weights(cache.pool, cache.directory, mixture.layer, op);
    cache.begin_round(layer, tokens);
    ops::SparseMoeRoundHook hook{&Impl::resolve_round, &layer};
    ops::sparse_moe(hidden, pooled, ops::SparseMoeEpilogue::AddResidual, destination, leaf, stream,
                    hook);

    static const int shadow_every = [] {
        const char* raw = std::getenv("SUROGATE_SERVE_CPU_MOE_SHADOW");
        return raw != nullptr && *raw != '\0' ? std::atoi(raw) : 0;
    }();
    if (shadow_every <= 0 || !layer.round_split || cache.pending.device_alias == nullptr) { return; }
    cudaStreamCaptureStatus capture = cudaStreamCaptureStatusNone;
    const bool capturing = cudaStreamIsCapturing(stream, &capture) != cudaSuccess ||
                           capture != cudaStreamCaptureStatusNone;
    if (capturing || ++cache.shadow_rounds % shadow_every != 0) { return; }
    const std::int32_t hidden_rows = cache.geometry.hidden;
    Tensor shadow = workspace.alloc(DType::BF16, {hidden_rows, tokens});
    CUDA_CHECK(cudaMemsetAsync(shadow.data, 0, shadow.bytes(), stream));
    const DeviceSpan storage2 = workspace.alloc_bytes(bytes);
    WorkspaceArena leaf2(storage2);
    ops::SparseMoeRoundHook shadow_hook{&Impl::resolve_round_shadow, &layer};
    ops::sparse_moe(hidden, pooled, ops::SparseMoeEpilogue::AddResidual, shadow, leaf2, stream,
                    shadow_hook);
    // The host partial is ready when its join event fires, or -- under the memop handshake --
    // when its done flag is raised. Wait without clearing the flag: the combine that follows
    // waits on it too and is the one that clears it.
    if (cache.pending.done_device != 0) {
        if (stream_wait_value64()(stream, cache.pending.done_device, 1ULL, kStreamWaitValueGeq) !=
            CUDA_SUCCESS) {
            throw std::runtime_error("expert cache: shadow wait on the done flag failed");
        }
    } else {
        CUDA_CHECK(cudaStreamWaitEvent(stream, cache.pending.join, 0));
    }
    const std::size_t n = static_cast<std::size_t>(hidden_rows) * tokens;
    std::vector<std::uint16_t> part(n), full(n);
    CUDA_CHECK(cudaMemcpyAsync(part.data(), destination.data, n * sizeof(std::uint16_t),
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(full.data(), shadow.data, n * sizeof(std::uint16_t),
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    const float* partial = cache.out_host;
    const auto bf16 = [](std::uint16_t bits) {
        std::uint32_t w = static_cast<std::uint32_t>(bits) << 16;
        float f;
        std::memcpy(&f, &w, sizeof(f));
        return f;
    };
    float scale = 1e-6F, worst = 0.0F;
    std::int32_t worst_token = -1;
    std::int64_t bad_tokens  = 0;
    for (std::int32_t t = 0; t < tokens; ++t) {
        float token_worst = 0.0F;
        for (std::int32_t d = 0; d < hidden_rows; ++d) {
            const std::size_t i   = static_cast<std::size_t>(t) * hidden_rows + d;
            const float reference = bf16(full[i]);
            const float value     = bf16(part[i]) + partial[i];
            scale                 = std::max(scale, std::fabs(reference));
            token_worst           = std::max(token_worst, std::fabs(value - reference));
        }
        if (token_worst > worst) {
            worst       = token_worst;
            worst_token = t;
        }
        if (token_worst > 0.05F) { ++bad_tokens; }
    }
    ++cache.shadow_checked;
    if (worst > 0.02F * scale) {
        ++cache.shadow_violations;
        // The worst token, decomposed: its GPU part, host partial, full result, and the
        // magnitude of its activation (host activations are int8 per group).
        float part_max = 0.0F, partial_max = 0.0F, full_max = 0.0F, x_max = 0.0F;
        std::int32_t x_argmax = -1;
        std::vector<std::uint16_t> x_bits(static_cast<std::size_t>(hidden_rows));
        CUDA_CHECK(cudaMemcpy(x_bits.data(),
                              static_cast<const std::uint16_t*>(hidden.data) +
                                  static_cast<std::size_t>(worst_token) * hidden_rows,
                              x_bits.size() * sizeof(std::uint16_t), cudaMemcpyDeviceToHost));
        for (std::int32_t d = 0; d < hidden_rows; ++d) {
            const std::size_t i = static_cast<std::size_t>(worst_token) * hidden_rows + d;
            part_max            = std::max(part_max, std::fabs(bf16(part[i])));
            partial_max         = std::max(partial_max, std::fabs(partial[i]));
            full_max            = std::max(full_max, std::fabs(bf16(full[i])));
            const float xv      = std::fabs(bf16(x_bits[static_cast<std::size_t>(d)]));
            if (xv > x_max) {
                x_max    = xv;
                x_argmax = d;
            }
        }
        std::int64_t host_jobs = 0;
        for (const ops::CpuExpertJob& job : cache.job_scratch) {
            if (job.token == worst_token) { ++host_jobs; }
        }
        std::fprintf(stderr,
                     "expert cache: host split shadow mismatch (layer %d, %d tokens): worst "
                     "|split - full| %.4g at token %d, scale %.3g, %lld tokens over 0.05 "
                     "(violation %lld of %lld checked); token %d: max|gpu part| %.4g, "
                     "max|host partial| %.4g, max|full| %.4g, max|x| %.4g at dim %d, %lld host "
                     "jobs in the last slice\n",
                     mixture.layer, tokens, static_cast<double>(worst), worst_token,
                     static_cast<double>(scale), static_cast<long long>(bad_tokens),
                     static_cast<long long>(cache.shadow_violations),
                     static_cast<long long>(cache.shadow_checked), worst_token,
                     static_cast<double>(part_max), static_cast<double>(partial_max),
                     static_cast<double>(full_max), static_cast<double>(x_max), x_argmax,
                     static_cast<long long>(host_jobs));
    } else if (cache.shadow_checked % 200 == 0) {
        std::fprintf(stderr, "expert cache: host split shadow: %lld rounds checked, %lld mismatches\n",
                     static_cast<long long>(cache.shadow_checked),
                     static_cast<long long>(cache.shadow_violations));
    }
}

void ExpertCache::prepare_split(const BankedMixture& mixture) {
    Impl& cache = *impl_;
    if (!cache.enabled || !cache.cpu_split_enabled() || !cache.auto_share || cache.share_measured) {
        if (cache.auto_share && !cache.share_measured) {
            std::fprintf(stderr, "expert cache: CPU split auto share not measured (cache %d, pool %d)\n",
                         int(cache.enabled), int(cache.cpu_split_enabled()));
        }
        return;
    }
    if (!mixture.banked()) {
        std::fprintf(stderr, "expert cache: CPU split auto share not measured (no banked mixture layer)\n");
        return;
    }
    Impl::Layer& layer                     = cache.layer(mixture);
    const ops::SparseMoeGeometry& geometry = cache.geometry;
    constexpr int kExpertsTimed            = 64;
    constexpr int kRepeats                 = 4;
    cudaStream_t stream                    = nullptr;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    // PCIe: gather kExpertsTimed experts into the first slots (the pool is otherwise empty).
    const int timed = std::min(kExpertsTimed, geometry.experts);
    std::vector<std::int32_t> slots(static_cast<std::size_t>(timed)),
        experts(static_cast<std::size_t>(timed));
    for (int i = 0; i < timed; ++i) {
        slots[static_cast<std::size_t>(i)]   = i;
        experts[static_cast<std::size_t>(i)] = i;
    }
    const long long count = timed;
    CUDA_CHECK(cudaMemcpy(cache.misses.slots.data, slots.data(), slots.size() * sizeof(std::int32_t),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(cache.misses.experts.data, experts.data(),
                          experts.size() * sizeof(std::int32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(cache.misses.count.data, &count, sizeof(count), cudaMemcpyHostToDevice));
    ops::expert_slot_gather(layer.bank, cache.misses, cache.pool, stream); // warm
    CUDA_CHECK(cudaStreamSynchronize(stream));
    cudaEvent_t t0 = nullptr, t1 = nullptr;
    CUDA_CHECK(cudaEventCreate(&t0));
    CUDA_CHECK(cudaEventCreate(&t1));
    CUDA_CHECK(cudaEventRecord(t0, stream));
    for (int r = 0; r < kRepeats; ++r) {
        ops::expert_slot_gather(layer.bank, cache.misses, cache.pool, stream);
    }
    CUDA_CHECK(cudaEventRecord(t1, stream));
    CUDA_CHECK(cudaEventSynchronize(t1));
    float gather_ms = 0.0F;
    CUDA_CHECK(cudaEventElapsedTime(&gather_ms, t0, t1));
    const double expert_bytes = static_cast<double>(
        layer.bank.gate_up_codes_bytes_per_expert + layer.bank.gate_up_scales_bytes_per_expert +
        layer.bank.down_codes_bytes_per_expert + layer.bank.down_scales_bytes_per_expert);
    const double pcie_gbs = expert_bytes * timed * kRepeats / (gather_ms * 1e-3) / 1e9;
    // Host: the same experts as jobs over 8 tokens (the round's x is arbitrary for timing).
    const int tokens = 8;
    cache.job_scratch.resize(static_cast<std::size_t>(timed));
    for (int i = 0; i < timed; ++i) {
        cache.job_scratch[static_cast<std::size_t>(i)] = {i % tokens, i, 0.1F};
    }
    std::fill_n(cache.x_host, static_cast<std::size_t>(geometry.hidden) * tokens,
                static_cast<std::uint16_t>(0x3F80)); // 1.0 in BF16
    ops::CpuExpertRound round{cache.x_host, cache.out_host, tokens, cache.job_scratch};
    std::fill_n(cache.out_host, static_cast<std::size_t>(geometry.hidden) * tokens, 0.0F);
    cache.cpu_pool->run(layer.cpu_bank, round); // warm
    const auto h0 = std::chrono::steady_clock::now();
    for (int r = 0; r < kRepeats; ++r) { cache.cpu_pool->run(layer.cpu_bank, round); }
    const double host_s   = std::chrono::duration<double>(std::chrono::steady_clock::now() - h0).count();
    const double host_gbs = expert_bytes * timed * kRepeats / host_s / 1e9;
    // The two rates that set the split are the ones each side reaches *while the other runs*:
    // the gather's DMA and the host's reads contend for the same DRAM, and neither standalone
    // number predicts what is left for it. The gathers are enqueued first and the host rounds
    // run against them; each side is timed over its own window, which differ by at most one
    // round.
    CUDA_CHECK(cudaEventRecord(t0, stream));
    for (int r = 0; r < kRepeats * 4; ++r) {
        ops::expert_slot_gather(layer.bank, cache.misses, cache.pool, stream);
    }
    CUDA_CHECK(cudaEventRecord(t1, stream));
    const auto o0   = std::chrono::steady_clock::now();
    int host_rounds = 0;
    while (cudaEventQuery(t1) == cudaErrorNotReady || host_rounds < kRepeats) {
        cache.cpu_pool->run(layer.cpu_bank, round);
        ++host_rounds;
    }
    const double host_ov_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - o0).count();
    CUDA_CHECK(cudaEventSynchronize(t1));
    float gather_ov_ms = 0.0F;
    CUDA_CHECK(cudaEventElapsedTime(&gather_ov_ms, t0, t1));
    const double pcie_ov_gbs = expert_bytes * timed * kRepeats * 4 / (gather_ov_ms * 1e-3) / 1e9;
    const double host_ov_gbs = expert_bytes * timed * host_rounds / host_ov_s / 1e9;
    double share = host_ov_gbs / (host_ov_gbs + pcie_ov_gbs);
    share        = std::min(0.9, std::max(0.3, share));
    cache.cpu_share_q16  = static_cast<std::uint32_t>(share * 65536.0);
    cache.share_measured = true;
    // The prefill share optimum sits below the decode one and tracks host strength (measured
    // 2026-08-28: 0.5 at 32 host threads where the decode share measures ~0.8, 0.3 at 16
    // threads): share - 0.3, clamped to [0.2, 0.7], unless given explicitly.
    if (cache.prefill_share_default && cache.cpu_prefill_share_q16 > 0) {
        const double prefill = std::min(0.7, std::max(0.2, share - 0.3));
        cache.cpu_prefill_share_q16 = static_cast<std::uint32_t>(prefill * 65536.0);
        std::fprintf(stderr, "expert cache: CPU split auto prefill share: %.0f%%\n", 100.0 * prefill);
    }
    // Leave the pool directory clean for the real rounds.
    ops::expert_slot_directory_reset(cache.directory, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaEventDestroy(t0));
    CUDA_CHECK(cudaEventDestroy(t1));
    CUDA_CHECK(cudaStreamDestroy(stream));
    std::fprintf(stderr,
                 "expert cache: CPU split auto share: host %.0f GB/s, PCIe gather %.0f GB/s alone; "
                 "%.0f and %.0f GB/s overlapped -> %.0f%% of misses on the host\n",
                 host_gbs, pcie_gbs, host_ov_gbs, pcie_ov_gbs, 100.0 * share);
}

bool ExpertCache::has_pending_partial() const noexcept {
    return impl_->pending.device_alias != nullptr;
}

const float* ExpertCache::wait_pending_partial(cudaStream_t stream) {
    Impl& cache = *impl_;
    if (cache.pending.device_alias == nullptr) {
        throw std::logic_error("expert cache: no host partial is pending");
    }
    check_device_handoff("a host expert partial", cache.pending.device);
    // Join the host round before the caller reads its partial: the done flag when the memop
    // handshake carried the round, the side stream's event otherwise.
    if (cache.pending.done_device != 0) {
        // Wait for the round, then clear the flag so the next raise -- the next replay of this
        // graph, or the next round through this slot -- starts from zero. The join probe
        // straddles the wait as it does the event path's, so it measures the handshake too
        // (eagerly; the probe records nothing under capture).
        join_probe().begin(stream);
        CUresult rc = stream_wait_value64()(stream, cache.pending.done_device, 1ULL, kStreamWaitValueGeq);
        if (rc == CUDA_SUCCESS) { rc = stream_write_value64()(stream, cache.pending.done_device, 0ULL, 0U); }
        if (rc != CUDA_SUCCESS) { throw std::runtime_error("expert cache: stream memop failed at the join"); }
        join_probe().end(stream);
    } else {
        join_probe().wrap(stream, cache.pending.join);
    }
    return cache.pending.device_alias;
}

void ExpertCache::finish_pending_partial(const Tensor& block_output, cudaStream_t stream) {
    Impl& cache = *impl_;
    if (cache.stagecheck && cache.out_check != nullptr) {
        const std::int32_t tokens = block_output.ne[1];
        CUDA_CHECK(cudaMemcpyAsync(cache.out_check, cache.out_device_alias,
                                   static_cast<std::size_t>(cache.geometry.hidden) * tokens * sizeof(float),
                                   cudaMemcpyDefault, stream));
        cache.out_check_tokens = tokens;
    }
    cache.pending = PendingPartial{};
}

void ExpertCache::add_pending_partial(Tensor& destination, cudaStream_t stream) {
    // A combine in its own right, for the probe's count of rounds that carried no partial.
    tick_combine();
    if (!has_pending_partial()) { return; }
    const float* partial = wait_pending_partial(stream);
    ops::expert_cpu_partial_add(partial, destination, stream);
    finish_pending_partial(destination, stream);
}

void ExpertCache::tick_combine() { join_probe().tick(); }

} // namespace sinfer::family
