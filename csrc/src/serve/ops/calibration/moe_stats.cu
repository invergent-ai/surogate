// MoE expert statistics for structural pruning (api/ops/calibration.h).
//
// Every mixture call adds, per (token, selected expert): one count and the renormalised gate
// weight -- read from the served round by a hook that rewrites nothing -- and, with norms on, the
// expert's own output norm ||y|| and gate * ||y|| (REAP's numerator), measured by probe rounds of
// the same kernels with the router weights rewritten to one slot. Accumulators are per device,
// FP64 for the sums and U64 for the counts, so a few million calibration tokens neither lose a
// count nor round a sum; a dump synchronises every device and adds them up.

#include "api/ops/calibration.h"

#include "core/device.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <map>
#include <mutex>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace sinfer::ops {
namespace {

struct DeviceStats {
    std::int32_t rows    = 0;
    std::int32_t experts = 0;
    std::int32_t top_k   = 0;
    void* base           = nullptr;
    std::size_t bytes    = 0;
    unsigned long long* counts = nullptr; // [rows][experts]
    double* gate               = nullptr; // [rows][experts]
    double* gate_norm          = nullptr; // [rows][experts]
    double* norm               = nullptr; // [rows][experts]
    unsigned long long* tokens = nullptr; // [rows]
};

struct Registry {
    std::mutex mutex;
    MoeStatsOptions options;
    std::atomic<bool> enabled{false};
    std::atomic<bool> norms{false};
    std::atomic<bool> mtp_prompt{false};
    std::map<int, DeviceStats> devices;
    std::int32_t rows = 0, trunk_layers = 0, experts = 0, top_k = 0, hidden = 0;
    std::set<std::int32_t> unobserved;
};

Registry& registry() {
    static Registry instance;
    return instance;
}

int current_device() {
    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    return device;
}

/// The calling device's accumulators, or null where the statistics are off or not prepared.
DeviceStats* device_stats() {
    Registry& r = registry();
    if (!r.enabled.load(std::memory_order_relaxed)) { return nullptr; }
    const int device = current_device();
    const std::lock_guard<std::mutex> lock(r.mutex);
    const auto found = r.devices.find(device);
    return found == r.devices.end() ? nullptr : &found->second;
}

constexpr int kThreads = 256;

/// One count and the gate weight per routed (token, slot); one block also adds the token count.
__global__ void moe_stats_route_kernel(const int* ids, const float* alpha, int assignments,
                                       int tokens, int experts, unsigned long long* counts,
                                       double* gate, unsigned long long* token_count) {
    const int stride = static_cast<int>(gridDim.x * blockDim.x);
    for (int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x); i < assignments;
         i += stride) {
        const int expert = ids[i];
        if (expert < 0 || expert >= experts) { continue; }
        atomicAdd(counts + expert, 1ULL);
        atomicAdd(gate + expert, static_cast<double>(alpha[i]));
    }
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        atomicAdd(token_count, static_cast<unsigned long long>(tokens));
    }
}

/// A probe round's rewrite: keeps the slot's id and weight for the norm pass, then makes the round
/// compute that slot's expert alone, unweighted: alpha one-hot on the slot, no shared expert.
__global__ void moe_stats_select_kernel(const int* ids, float* alpha, float* shared_scale,
                                        int tokens, int top_k, int slot, int* saved_ids,
                                        float* saved_alpha) {
    const int token = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (token >= tokens) { return; }
    const int base     = token * top_k;
    saved_ids[token]   = ids[base + slot];
    saved_alpha[token] = alpha[base + slot];
    for (int k = 0; k < top_k; ++k) { alpha[base + k] = k == slot ? 1.0F : 0.0F; }
    if (shared_scale != nullptr) { shared_scale[token] = 0.0F; }
}

/// ||y[:, t]||_2 of a probe round's output, added to the slot's expert: Sum ||y|| and
/// Sum gate * ||y||. One block per token.
__global__ void moe_stats_norm_kernel(const __nv_bfloat16* y, int hidden, const int* saved_ids,
                                      const float* saved_alpha, int experts, double* norm,
                                      double* gate_norm) {
    __shared__ float partial[kThreads / 32];
    const int token        = static_cast<int>(blockIdx.x);
    const __nv_bfloat16* c = y + static_cast<std::size_t>(token) * hidden;
    float sum              = 0.0F;
    for (int i = static_cast<int>(threadIdx.x); i < hidden; i += blockDim.x) {
        const float v = __bfloat162float(c[i]);
        sum += v * v;
    }
    for (int offset = 16; offset > 0; offset >>= 1) { sum += __shfl_down_sync(0xffffffffU, sum, offset); }
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int warp = static_cast<int>(threadIdx.x) >> 5;
    if (lane == 0) { partial[warp] = sum; }
    __syncthreads();
    if (threadIdx.x == 0) {
        double total = 0.0;
        for (int w = 0; w < static_cast<int>(blockDim.x / 32); ++w) { total += partial[w]; }
        const int expert = saved_ids[token];
        if (expert >= 0 && expert < experts) {
            const double length = ::sqrt(total);
            atomicAdd(norm + expert, length);
            atomicAdd(gate_norm + expert, static_cast<double>(saved_alpha[token]) * length);
        }
    }
}

void observe_round(void* context, const Tensor& ids, float* alpha, float*, std::int32_t,
                   std::int32_t tokens, cudaStream_t stream) {
    const auto& observer = *static_cast<const MoeStatsObserver*>(context);
    const auto* stats    = static_cast<const DeviceStats*>(observer.stats);
    if (stats == nullptr || tokens <= 0) { return; }
    const std::size_t row = static_cast<std::size_t>(observer.layer) * stats->experts;
    const int assignments = tokens * observer.top_k;
    const int blocks      = std::max(1, std::min((assignments + kThreads - 1) / kThreads, 64));
    moe_stats_route_kernel<<<blocks, kThreads, 0, stream>>>(
        static_cast<const int*>(ids.data), alpha, assignments, tokens, stats->experts,
        stats->counts + row, stats->gate + row, stats->tokens + observer.layer);
    CUDA_CHECK(cudaGetLastError());
}

struct ProbeContext {
    std::int32_t top_k = 0;
    std::int32_t slot  = 0;
    int* saved_ids     = nullptr;
    float* saved_alpha = nullptr;
};

void probe_round(void* context, const Tensor& ids, float* alpha, float* shared_scale,
                 std::int32_t first_token, std::int32_t tokens, cudaStream_t stream) {
    const auto& probe = *static_cast<const ProbeContext*>(context);
    if (tokens <= 0) { return; }
    moe_stats_select_kernel<<<(tokens + kThreads - 1) / kThreads, kThreads, 0, stream>>>(
        static_cast<const int*>(ids.data), alpha, shared_scale, tokens, probe.top_k, probe.slot,
        probe.saved_ids + first_token, probe.saved_alpha + first_token);
    CUDA_CHECK(cudaGetLastError());
}

std::size_t aligned(std::size_t bytes) { return (bytes + 255) / 256 * 256; }

std::string utc_stamp(const char* format) {
    const std::time_t now = std::time(nullptr);
    std::tm parts{};
    gmtime_r(&now, &parts);
    char text[64] = {};
    std::strftime(text, sizeof(text), format, &parts);
    return text;
}

std::string json_string(const std::string& value) {
    std::string out = "\"";
    for (const char c : value) {
        if (c == '"' || c == '\\') { out += '\\'; out += c; }
        else if (static_cast<unsigned char>(c) < 0x20) {
            char escaped[8];
            std::snprintf(escaped, sizeof(escaped), "\\u%04x", static_cast<unsigned>(c));
            out += escaped;
        } else { out += c; }
    }
    return out + "\"";
}

/// NumPy's .npy v1.0: magic, version, a little-endian header length, the header dict padded so the
/// data starts on a 64-byte boundary, then the raw little-endian data.
void write_npy(const std::filesystem::path& path, const char* descr,
               const std::vector<std::size_t>& shape, const void* data, std::size_t bytes) {
    std::string dims;
    for (std::size_t i = 0; i < shape.size(); ++i) {
        dims += std::to_string(shape[i]) + (shape.size() == 1 || i + 1 < shape.size() ? "," : "");
        if (i + 1 < shape.size()) { dims += " "; }
    }
    std::string header =
        std::string("{'descr': '") + descr + "', 'fortran_order': False, 'shape': (" + dims + "), }";
    const std::size_t unpadded = 10 + header.size() + 1;
    header += std::string((64 - unpadded % 64) % 64, ' ') + "\n";
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out) { throw std::runtime_error("moe stats: cannot write " + path.string()); }
    const char magic[8] = {'\x93', 'N', 'U', 'M', 'P', 'Y', 1, 0};
    out.write(magic, sizeof(magic));
    const std::uint16_t length = static_cast<std::uint16_t>(header.size());
    const char length_bytes[2] = {static_cast<char>(length & 0xff), static_cast<char>(length >> 8)};
    out.write(length_bytes, 2);
    out.write(header.data(), static_cast<std::streamsize>(header.size()));
    out.write(static_cast<const char*>(data), static_cast<std::streamsize>(bytes));
    if (!out) { throw std::runtime_error("moe stats: short write to " + path.string()); }
}

struct HostStats {
    std::vector<unsigned long long> counts, tokens;
    std::vector<double> gate, gate_norm, norm;
};

/// Every device's accumulators, synchronised and summed. Holds the registry lock.
HostStats gather(Registry& r) {
    HostStats out;
    const std::size_t cells = static_cast<std::size_t>(r.rows) * r.experts;
    out.counts.assign(cells, 0);
    out.gate.assign(cells, 0.0);
    out.gate_norm.assign(cells, 0.0);
    out.norm.assign(cells, 0.0);
    out.tokens.assign(static_cast<std::size_t>(r.rows), 0);
    const int previous = current_device();
    for (const auto& [device, stats] : r.devices) {
        CUDA_CHECK(cudaSetDevice(device));
        CUDA_CHECK(cudaDeviceSynchronize());
        std::vector<unsigned long long> counts(cells), tokens(static_cast<std::size_t>(r.rows));
        std::vector<double> gate(cells), gate_norm(cells), norm(cells);
        CUDA_CHECK(cudaMemcpy(counts.data(), stats.counts, cells * sizeof(unsigned long long), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(gate.data(), stats.gate, cells * sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(gate_norm.data(), stats.gate_norm, cells * sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(norm.data(), stats.norm, cells * sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(tokens.data(), stats.tokens, tokens.size() * sizeof(unsigned long long), cudaMemcpyDeviceToHost));
        for (std::size_t i = 0; i < cells; ++i) {
            out.counts[i] += counts[i];
            out.gate[i] += gate[i];
            out.gate_norm[i] += gate_norm[i];
            out.norm[i] += norm[i];
        }
        for (std::size_t i = 0; i < tokens.size(); ++i) { out.tokens[i] += tokens[i]; }
    }
    CUDA_CHECK(cudaSetDevice(previous));
    return out;
}

void zero(Registry& r) {
    const int previous = current_device();
    for (const auto& [device, stats] : r.devices) {
        CUDA_CHECK(cudaSetDevice(device));
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemset(stats.base, 0, stats.bytes));
    }
    CUDA_CHECK(cudaSetDevice(previous));
}

} // namespace

void configure_moe_stats(const MoeStatsOptions& options) {
    Registry& r = registry();
    const std::lock_guard<std::mutex> lock(r.mutex);
    if (!r.devices.empty()) { throw std::logic_error("moe stats: configure before any model loads"); }
    r.options = options;
    r.enabled.store(!options.directory.empty(), std::memory_order_relaxed);
    r.norms.store(!options.directory.empty() && options.norms, std::memory_order_relaxed);
    r.mtp_prompt.store(!options.directory.empty() && options.mtp_prompt, std::memory_order_relaxed);
}

bool moe_stats_enabled() noexcept { return registry().enabled.load(std::memory_order_relaxed); }
bool moe_stats_norms() noexcept { return registry().norms.load(std::memory_order_relaxed); }
bool moe_stats_mtp_prompt() noexcept { return registry().mtp_prompt.load(std::memory_order_relaxed); }

void moe_stats_prepare_device(std::int32_t mixtures, std::int32_t trunk_layers, std::int32_t experts,
                              std::int32_t experts_per_token, std::int32_t hidden) {
    Registry& r = registry();
    if (!r.enabled.load(std::memory_order_relaxed)) { return; }
    if (mixtures <= 0 || experts <= 0 || experts_per_token <= 0 || trunk_layers <= 0 ||
        trunk_layers > mixtures) {
        throw std::invalid_argument("moe stats: invalid mixture shape");
    }
    const int device = current_device();
    const std::lock_guard<std::mutex> lock(r.mutex);
    if (r.rows != 0 && (r.rows != mixtures || r.experts != experts || r.top_k != experts_per_token ||
                        r.trunk_layers != trunk_layers || r.hidden != hidden)) {
        throw std::invalid_argument(
            "moe stats: one process observes one model geometry; this one differs from the first");
    }
    r.rows = mixtures;
    r.trunk_layers = trunk_layers;
    r.experts = experts;
    r.top_k = experts_per_token;
    r.hidden = hidden;
    if (r.devices.contains(device)) { return; }
    DeviceStats stats;
    stats.rows    = mixtures;
    stats.experts = experts;
    stats.top_k   = experts_per_token;
    const std::size_t cells = static_cast<std::size_t>(mixtures) * experts;
    const std::size_t counts_bytes = aligned(cells * sizeof(unsigned long long));
    const std::size_t sum_bytes    = aligned(cells * sizeof(double));
    const std::size_t token_bytes  = aligned(static_cast<std::size_t>(mixtures) * sizeof(unsigned long long));
    stats.bytes = counts_bytes + 3 * sum_bytes + token_bytes;
    CUDA_CHECK(cudaMalloc(&stats.base, stats.bytes));
    CUDA_CHECK(cudaMemset(stats.base, 0, stats.bytes));
    auto* cursor     = static_cast<std::byte*>(stats.base);
    stats.counts     = reinterpret_cast<unsigned long long*>(cursor);
    stats.gate       = reinterpret_cast<double*>(cursor + counts_bytes);
    stats.gate_norm  = reinterpret_cast<double*>(cursor + counts_bytes + sum_bytes);
    stats.norm       = reinterpret_cast<double*>(cursor + counts_bytes + 2 * sum_bytes);
    stats.tokens     = reinterpret_cast<unsigned long long*>(cursor + counts_bytes + 3 * sum_bytes);
    r.devices.emplace(device, stats);
}

SparseMoeRoundHook moe_stats_observer(std::int32_t layer, MoeStatsObserver& storage) {
    SparseMoeRoundHook hook{};
    DeviceStats* stats = device_stats();
    if (stats == nullptr || layer < 0 || layer >= stats->rows) { return hook; }
    storage.stats = stats;
    storage.layer = layer;
    storage.top_k = stats->top_k;
    hook.reweight         = &observe_round;
    hook.reweight_context = &storage;
    return hook;
}

std::size_t moe_stats_probe_workspace_bytes(std::int32_t hidden, std::int32_t max_tokens) noexcept {
    if (!moe_stats_norms() || max_tokens <= 0) { return 0; }
    const auto tokens = static_cast<std::size_t>(max_tokens);
    return aligned(static_cast<std::size_t>(hidden) * tokens * sizeof(__nv_bfloat16)) +
           aligned(tokens * sizeof(int)) + aligned(tokens * sizeof(float));
}

void moe_stats_probe(std::int32_t layer, const Tensor& x, const SparseMoeWeights& weights,
                     SparseMoeRouting routing, WorkspaceArena& moe_workspace,
                     WorkspaceArena& workspace, cudaStream_t stream) {
    if (!moe_stats_norms()) { return; }
    DeviceStats* stats = device_stats();
    if (stats == nullptr || layer < 0 || layer >= stats->rows) { return; }
    const SparseMoeGeometry geometry = sparse_moe_geometry(weights);
    const std::int32_t tokens        = x.ne[1];
    if (tokens <= 0) { return; }
    auto scope   = workspace.scope();
    Tensor y     = workspace.alloc(DType::BF16, {geometry.hidden, tokens});
    Tensor ids   = workspace.alloc(DType::I32, {tokens});
    Tensor alpha = workspace.alloc(DType::FP32, {tokens});
    ProbeContext probe{geometry.experts_per_token, 0, static_cast<int*>(ids.data),
                       static_cast<float*>(alpha.data)};
    SparseMoeRoundHook hook{};
    hook.reweight         = &probe_round;
    hook.reweight_context = &probe;
    const std::size_t row = static_cast<std::size_t>(layer) * stats->experts;
    for (std::int32_t slot = 0; slot < geometry.experts_per_token; ++slot) {
        probe.slot = slot;
        CUDA_CHECK(cudaMemsetAsync(y.data, 0, y.bytes(), stream));
        sparse_moe(x, weights, SparseMoeEpilogue::AddResidual, y, moe_workspace, stream, hook, routing);
        moe_stats_norm_kernel<<<tokens, kThreads, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(y.data), geometry.hidden,
            static_cast<const int*>(ids.data), static_cast<const float*>(alpha.data),
            stats->experts, stats->norm + row, stats->gate_norm + row);
        CUDA_CHECK(cudaGetLastError());
    }
}

void moe_stats_note_unobserved(std::int32_t layer) noexcept {
    Registry& r = registry();
    if (!r.enabled.load(std::memory_order_relaxed)) { return; }
    const std::lock_guard<std::mutex> lock(r.mutex);
    r.unobserved.insert(layer);
}

std::string moe_stats_dump(const std::string& tag, bool reset) {
    Registry& r = registry();
    if (!r.enabled.load(std::memory_order_relaxed)) {
        throw std::runtime_error("moe stats are off (start the engine with --moe-stats-out DIR)");
    }
    for (const char c : tag) {
        if (!(std::isalnum(static_cast<unsigned char>(c)) || c == '-' || c == '_' || c == '.')) {
            throw std::invalid_argument("moe stats: a tag is letters, digits, '-', '_' and '.' only");
        }
    }
    const std::lock_guard<std::mutex> lock(r.mutex);
    if (r.devices.empty()) { throw std::runtime_error("moe stats: no device has been prepared yet"); }
    const std::string name = tag.empty() ? utc_stamp("%Y%m%dT%H%M%SZ") : tag;
    const std::filesystem::path dir = std::filesystem::path(r.options.directory) / name;
    std::filesystem::create_directories(dir);
    const HostStats host = gather(r);
    const std::vector<std::size_t> shape{static_cast<std::size_t>(r.rows), static_cast<std::size_t>(r.experts)};
    write_npy(dir / "counts.npy", "<u8", shape, host.counts.data(), host.counts.size() * 8);
    write_npy(dir / "gate_sum.npy", "<f8", shape, host.gate.data(), host.gate.size() * 8);
    write_npy(dir / "gate_norm_sum.npy", "<f8", shape, host.gate_norm.data(), host.gate_norm.size() * 8);
    write_npy(dir / "norm_sum.npy", "<f8", shape, host.norm.data(), host.norm.size() * 8);
    write_npy(dir / "tokens.npy", "<u8", {static_cast<std::size_t>(r.rows)}, host.tokens.data(),
              host.tokens.size() * 8);
    std::ostringstream meta;
    meta << "{\n  \"format\": \"surogate-moe-stats-v1\",\n"
         << "  \"created_utc\": " << json_string(utc_stamp("%Y-%m-%dT%H:%M:%SZ")) << ",\n"
         << "  \"tag\": " << json_string(name) << ",\n"
         << "  \"model\": " << json_string(r.options.model) << ",\n"
         << "  \"rows\": " << r.rows << ",\n  \"trunk_layers\": " << r.trunk_layers << ",\n"
         << "  \"mtp_rows\": [";
    for (std::int32_t row = r.trunk_layers; row < r.rows; ++row) { meta << (row > r.trunk_layers ? ", " : "") << row; }
    meta << "],\n  \"experts\": " << r.experts << ",\n  \"experts_per_token\": " << r.top_k
         << ",\n  \"hidden\": " << r.hidden << ",\n"
         << "  \"norms\": " << (r.options.norms ? "true" : "false") << ",\n"
         << "  \"mtp_prompt\": " << (r.options.mtp_prompt ? "true" : "false") << ",\n"
         << "  \"devices\": [";
    bool first = true;
    for (const auto& [device, stats] : r.devices) { meta << (first ? "" : ", ") << device; first = false; }
    meta << "],\n  \"unobserved_rows\": [";
    first = true;
    for (const auto layer : r.unobserved) { meta << (first ? "" : ", ") << layer; first = false; }
    meta << "],\n  \"tokens\": [";
    for (std::size_t i = 0; i < host.tokens.size(); ++i) { meta << (i ? ", " : "") << host.tokens[i]; }
    meta << "],\n"
         << "  \"arrays\": {\n"
         << "    \"counts.npy\": \"uint64 [rows, experts]: (token, selected slot) pairs routed to the expert\",\n"
         << "    \"gate_sum.npy\": \"float64 [rows, experts]: sum of the renormalised top-k gate weight over them\",\n"
         << "    \"gate_norm_sum.npy\": \"float64 [rows, experts]: sum of gate * ||y||_2 (REAP numerator); zero without norms\",\n"
         << "    \"norm_sum.npy\": \"float64 [rows, experts]: sum of ||y||_2; zero without norms\",\n"
         << "    \"tokens.npy\": \"uint64 [rows]: token columns the mixture ran over\"\n  },\n"
         << "  \"definitions\": [\n"
         << "    \"row i < trunk_layers is trunk layer i (0-based); the rows after it are the NextN draft head's mixture\",\n"
         << "    \"y is the expert's own SwiGLU output for the token before its gate weight, as the engine's kernels compute it (BF16 out; under --moe-route auto rounds of 2+ columns run the NVFP4 W4A4 runner)\",\n"
         << "    \"REAP score: gate_norm_sum / counts; frequency-weighted: gate_norm_sum / tokens\",\n"
         << "    \"every column the mixture runs over counts: prompt columns (prefill), decode columns, and a speculative verify round's draft columns whether accepted or not\",\n"
         << "    \"columns served from the prefix cache are not recomputed and not counted (--no-prefix-reuse counts every prompt token)\"\n"
         << "  ]\n}\n";
    std::ofstream(dir / "meta.json", std::ios::trunc) << meta.str();
    if (reset) { zero(r); }
    return dir.string();
}

void moe_stats_reset() {
    Registry& r = registry();
    if (!r.enabled.load(std::memory_order_relaxed)) { return; }
    const std::lock_guard<std::mutex> lock(r.mutex);
    zero(r);
}

std::string moe_stats_tokens_json() {
    Registry& r = registry();
    if (!r.enabled.load(std::memory_order_relaxed)) { return "null"; }
    const std::lock_guard<std::mutex> lock(r.mutex);
    if (r.devices.empty()) { return "[]"; }
    const HostStats host = gather(r);
    std::string out = "[";
    for (std::size_t i = 0; i < host.tokens.size(); ++i) {
        out += (i ? ", " : "") + std::to_string(host.tokens[i]);
    }
    return out + "]";
}

} // namespace sinfer::ops
