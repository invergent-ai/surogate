#include "ops/sparse_moe/trtllm/trtllm_moe.h"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <map>
#include <mutex>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "core/device.h"

#include "tensorrt_llm/kernels/cutlass_kernels/include/common.h"
#include "tensorrt_llm/kernels/cutlass_kernels/include/moe_kernels.h"

namespace sinfer::ops::detail::trtllm_moe {
namespace {

namespace tk = tensorrt_llm::kernels;
namespace tkc = tensorrt_llm::kernels::cutlass_kernels;
namespace tce = tensorrt_llm::cutlass_extensions;

/// bf16 activations in, e2m1 weights and activations inside, bf16 out: the one instantiation the
/// vendored library compiles (third_party/trtllm_moe/fused_moe/cutlass_backend/
/// cutlass_fused_moe_instantiation.cu).
using Runner = tkc::CutlassMoeFCRunner<__nv_fp4_e2m1, __nv_fp4_e2m1, __nv_bfloat16, __nv_bfloat16>;

/// The widest round the prefill family hands over in one call.
constexpr std::int32_t kMaxTokens = 4096;

/// Identifies the tuning: bumping it invalidates every cached tactic file. Raise it when the
/// vendored kernels, the ladder, or the timing method change.
constexpr int kTuningVersion = 1;

std::size_t align_up(std::size_t bytes, std::size_t alignment) noexcept {
    return (bytes + alignment - 1) / alignment * alignment;
}

// -----------------------------------------------------------------------------------------------
// Width buckets
// -----------------------------------------------------------------------------------------------

/// The tuning ladder, matching FlashInfer's hybrid spacing (flashinfer/fused_moe/utils.py):
/// powers of two to 256, then 256-steps to 2,048, then 512-steps to 4,096. Pure powers of two
/// leave gaps a MoE round crosses several tile boundaries inside, which is how a tactic tuned at
/// 1,024 ends up serving 2,047 rows badly.
std::vector<std::int32_t> ladder() {
    std::vector<std::int32_t> buckets;
    for (std::int32_t m = 1; m <= 256; m *= 2) { buckets.push_back(m); }
    for (std::int32_t m = 512; m <= 2048; m += 256) { buckets.push_back(m); }
    for (std::int32_t m = 2560; m <= kMaxTokens; m += 512) { buckets.push_back(m); }
    return buckets;
}

// -----------------------------------------------------------------------------------------------
// Workspace
// -----------------------------------------------------------------------------------------------

struct WorkspaceLayout {
    std::size_t runner_offset = 0;
    std::size_t runner_bytes  = 0;
    std::size_t output_offset = 0; ///< BF16 [tokens][hidden] the runner writes
    std::size_t output_bytes  = 0;
    std::size_t map_offset    = 0; ///< int32 [tokens * top-k] the runner requires
    std::size_t map_bytes     = 0;
    std::size_t total         = 0;
};

Runner& runner();

WorkspaceLayout workspace_layout(const Geometry& geometry, std::int32_t max_tokens) {
    WorkspaceLayout out;
    out.runner_bytes = runner().getWorkspaceSize(
        max_tokens, geometry.hidden, geometry.intermediate, geometry.experts,
        geometry.experts_per_token, tkc::ActivationType::Swiglu, tkc::MOEParallelismConfig{},
        /*use_lora*/ false, /*use_deepseek_fp8_block_scale*/ false, /*use_mxfp8_act_scaling*/ false,
        /*min_latency_mode*/ false, /*use_awq*/ false);
    out.output_bytes = static_cast<std::size_t>(max_tokens) * geometry.hidden * sizeof(__nv_bfloat16);
    out.map_bytes =
        static_cast<std::size_t>(max_tokens) * geometry.experts_per_token * sizeof(std::int32_t);
    out.runner_offset = 0;
    out.output_offset = align_up(out.runner_offset + out.runner_bytes, 256);
    out.map_offset    = align_up(out.output_offset + out.output_bytes, 256);
    out.total         = align_up(out.map_offset + out.map_bytes, 256);
    return out;
}

// -----------------------------------------------------------------------------------------------
// The runner, its tactics, and the on-disk cache
// -----------------------------------------------------------------------------------------------

struct TacticKey {
    std::int32_t bucket = 0;
    int gemm            = 0; ///< 1 or 2

    friend bool operator<(const TacticKey& a, const TacticKey& b) noexcept {
        return a.bucket != b.bucket ? a.bucket < b.bucket : a.gemm < b.gemm;
    }
};

struct State {
    std::mutex mutex;
    Runner runner;
    std::map<TacticKey, tce::CutlassGemmConfig> tactics;
    std::string cache_path;
    bool cache_loaded = false;
};

State& state() {
    static State instance;
    return instance;
}

Runner& runner() { return state().runner; }

std::string cache_directory() {
    if (const char* xdg = std::getenv("XDG_CACHE_HOME"); xdg != nullptr && xdg[0] != '\0') {
        return std::string(xdg) + "/surogate/trtllm_moe";
    }
    if (const char* home = std::getenv("HOME"); home != nullptr && home[0] != '\0') {
        return std::string(home) + "/.cache/surogate/trtllm_moe";
    }
    return {};
}

/// One file per device model and geometry: a tactic is only comparable within both.
std::string cache_path(const Geometry& geometry) {
    const std::string directory = cache_directory();
    if (directory.empty()) { return {}; }
    int device = 0;
    if (cudaGetDevice(&device) != cudaSuccess) { return {}; }
    cudaDeviceProp props{};
    if (cudaGetDeviceProperties(&props, device) != cudaSuccess) { return {}; }
    std::string name(props.name);
    for (char& character : name) {
        if (std::isalnum(static_cast<unsigned char>(character)) == 0) { character = '_'; }
    }
    std::ostringstream path;
    path << directory << "/v" << kTuningVersion << "_" << name << "_h" << geometry.hidden << "_e"
         << geometry.experts << "_k" << geometry.experts_per_token << "_i" << geometry.intermediate
         << ".tactics";
    return path.str();
}

bool make_cache_directory() {
    const std::string directory = cache_directory();
    if (directory.empty()) { return false; }
    std::error_code error;
    std::filesystem::create_directories(directory, error);
    return !error;
}

/// The tactic list is a property of the runner and the GEMM, not of the round, so one lookup
/// serves every bucket.
const std::vector<tce::CutlassGemmConfig>& tactics_for(int gemm) {
    static std::vector<tce::CutlassGemmConfig> gemm1 =
        state().runner.getTactics(tkc::MoeGemmId::GEMM_1);
    static std::vector<tce::CutlassGemmConfig> gemm2 =
        state().runner.getTactics(tkc::MoeGemmId::GEMM_2);
    return gemm == 1 ? gemm1 : gemm2;
}

/// `CutlassGemmConfig::toString` prints several lines; the cache is one record per line, so the
/// breaks are folded away and restored nowhere - the string is only ever compared with itself.
std::string flattened(const tce::CutlassGemmConfig& config) {
    std::string printed = config.toString();
    for (char& character : printed) {
        if (character == '\n' || character == '\r' || character == '\t') { character = ' '; }
    }
    // Collapse runs of spaces so the record is stable against incidental whitespace.
    std::string out;
    out.reserve(printed.size());
    bool space = false;
    for (const char character : printed) {
        if (character == ' ') {
            space = true;
            continue;
        }
        if (space && !out.empty()) { out.push_back(' '); }
        space = false;
        out.push_back(character);
    }
    return out;
}

/// A tactic is stored by its printed form, not its index: an index would silently select a
/// different kernel if the vendored tactic list ever changes order.
void load_cache(const Geometry& geometry) {
    State& s = state();
    if (s.cache_loaded) { return; }
    s.cache_loaded = true;
    s.cache_path   = cache_path(geometry);
    if (s.cache_path.empty()) { return; }
    std::ifstream file(s.cache_path);
    if (!file) { return; }
    std::string line;
    while (std::getline(file, line)) {
        std::istringstream parsed(line);
        int gemm            = 0;
        std::int32_t bucket = 0;
        if (!(parsed >> gemm >> bucket)) { continue; }
        std::string printed;
        std::getline(parsed, printed);
        if (!printed.empty() && printed.front() == ' ') { printed.erase(0, 1); }
        if (gemm != 1 && gemm != 2) { continue; }
        const std::vector<tce::CutlassGemmConfig>& candidates = tactics_for(gemm);
        for (const tce::CutlassGemmConfig& candidate : candidates) {
            if (flattened(candidate) == printed) {
                s.tactics[TacticKey{bucket, gemm}] = candidate;
                break;
            }
        }
    }
}

void store_cache() {
    State& s = state();
    if (s.cache_path.empty() || !make_cache_directory()) { return; }
    std::ofstream file(s.cache_path, std::ios::trunc);
    if (!file) { return; }
    for (const auto& [key, config] : s.tactics) {
        file << key.gemm << ' ' << key.bucket << ' ' << flattened(config) << '\n';
    }
}

// -----------------------------------------------------------------------------------------------
// Calling the runner
// -----------------------------------------------------------------------------------------------

tkc::QuantParams quant_params_of(const Nvfp4RoutedExperts& experts) {
    using ElementSF = tkc::TmaWarpSpecializedGroupedGemmInput::NVFP4ElementSF;
    return tkc::QuantParams::FP4(
        experts.gate_up_act_scale, static_cast<const ElementSF*>(experts.gate_up_block_scales),
        experts.gate_up_alpha, experts.down_act_scale,
        static_cast<const ElementSF*>(experts.down_block_scales), experts.down_alpha,
        /*fc1_use_per_expert_act_scale*/ true, /*fc2_use_per_expert_act_scale*/ true);
}

/// Runs the two grouped GEMMs with `gemm1`/`gemm2` already selected. Leaves the BF16 result in
/// `output`; the caller converts it.
void launch(const Geometry& geometry, const __nv_bfloat16* x, std::int32_t tokens,
            const std::int32_t* ids, const float* final_scales,
            const Nvfp4RoutedExperts& experts, char* runner_workspace, __nv_bfloat16* output,
            std::int32_t* permutation_map, cudaStream_t stream) {
    tk::LoraParams lora_params{};
    tkc::MoeMinLatencyParams min_latency_params{};
    state().runner.runMoe(
        x, /*input_sf*/ nullptr, /*swizzled_input_sf*/ false, ids, final_scales,
        experts.gate_up_codes, /*fc1_expert_biases*/ nullptr,
        tkc::ActivationParams(tkc::ActivationType::Swiglu), experts.down_codes,
        /*fc2_expert_biases*/ nullptr, quant_params_of(experts), tokens, geometry.hidden,
        /*unpadded_hidden_size*/ geometry.hidden, geometry.intermediate, geometry.experts,
        geometry.experts_per_token, runner_workspace, output, permutation_map,
        tkc::MOEParallelismConfig{}, /*enable_alltoall*/ false, /*use_lora*/ false, lora_params,
        /*use_deepseek_fp8_block_scale*/ false, /*use_mxfp8_act_scaling*/ false,
        /*min_latency_mode*/ false, min_latency_params, /*enable_pdl*/ false, stream);
}

__global__ void widen_kernel(const __nv_bfloat16* __restrict__ source, float* __restrict__ target,
                             std::int64_t count) {
    const std::int64_t index = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < count) { target[index] = __bfloat162float(source[index]); }
}

void widen(const __nv_bfloat16* source, float* target, std::int64_t count, cudaStream_t stream) {
    constexpr int kThreads = 256;
    const std::int64_t blocks = (count + kThreads - 1) / kThreads;
    widen_kernel<<<static_cast<unsigned>(blocks), kThreads, 0, stream>>>(source, target, count);
}

// -----------------------------------------------------------------------------------------------
// Tuning
// -----------------------------------------------------------------------------------------------

/// Fills a tuning round's activations and routing without curand: a cheap integer hash keeps the
/// timing reproducible and the values in the range a real round sees.
__global__ void tuning_input_kernel(__nv_bfloat16* __restrict__ x, std::int64_t count,
                                    std::uint32_t seed) {
    const std::int64_t index = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= count) { return; }
    std::uint32_t hash = static_cast<std::uint32_t>(index) * 2654435761U + seed;
    hash ^= hash >> 15;
    hash *= 2246822519U;
    hash ^= hash >> 13;
    const float value = static_cast<float>(hash & 0xffffU) / 32768.0f - 1.0f;
    x[index]          = __float2bfloat16(value);
}

/// Distinct experts per row, spread over the whole pool: a tactic tuned against a degenerate
/// routing would be tuned against the wrong per-expert row counts.
__global__ void tuning_route_kernel(std::int32_t* __restrict__ ids, float* __restrict__ scales,
                                    std::int32_t tokens, std::int32_t experts, std::int32_t top_k) {
    const std::int32_t token = blockIdx.x * blockDim.x + threadIdx.x;
    if (token >= tokens) { return; }
    std::uint32_t hash = static_cast<std::uint32_t>(token) * 2654435761U + 0x9e3779b9U;
    hash ^= hash >> 16;
    const std::int32_t stride = 1 + static_cast<std::int32_t>(hash % 7U);
    std::int32_t expert       = static_cast<std::int32_t>(hash % static_cast<std::uint32_t>(experts));
    for (std::int32_t slot = 0; slot < top_k; ++slot) {
        ids[token * top_k + slot]    = expert;
        scales[token * top_k + slot] = 1.0f / static_cast<float>(top_k);
        expert                       = (expert + stride) % experts;
    }
}

struct TuningBuffers {
    __nv_bfloat16* x        = nullptr;
    std::int32_t* ids       = nullptr;
    float* scales           = nullptr;
    void* workspace         = nullptr;
    std::size_t bytes       = 0;

    ~TuningBuffers() {
        (void)cudaFree(x);
        (void)cudaFree(ids);
        (void)cudaFree(scales);
        (void)cudaFree(workspace);
    }
};

/// Times one candidate pair. Returns nullopt when the tactic cannot run this shape — the vendored
/// heuristic lists tactics that exceed shared memory or alignment for some widths, and the runner
/// signals that by throwing or by leaving a sticky CUDA error.
std::optional<float> time_candidate(const Geometry& geometry, std::int32_t bucket,
                                    const Nvfp4RoutedExperts& experts, const TuningBuffers& buffers,
                                    const WorkspaceLayout& layout,
                                    const tce::CutlassGemmConfig& gemm1,
                                    const tce::CutlassGemmConfig& gemm2, cudaStream_t stream) {
    auto* base            = static_cast<char*>(buffers.workspace);
    auto* output          = reinterpret_cast<__nv_bfloat16*>(base + layout.output_offset);
    auto* permutation_map = reinterpret_cast<std::int32_t*>(base + layout.map_offset);

    const auto attempt = [&]() -> bool {
        try {
            state().runner.setTactic(gemm1, gemm2);
            launch(geometry, buffers.x, bucket, buffers.ids, buffers.scales, experts,
                   base + layout.runner_offset, output, permutation_map, stream);
        } catch (const std::exception&) {
            (void)cudaGetLastError();
            return false;
        }
        if (cudaStreamSynchronize(stream) != cudaSuccess) {
            (void)cudaGetLastError();
            return false;
        }
        return cudaGetLastError() == cudaSuccess;
    };

    constexpr int kWarmup = 3;
    constexpr int kTimed  = 10;
    for (int iteration = 0; iteration < kWarmup; ++iteration) {
        if (!attempt()) { return std::nullopt; }
    }

    cudaEvent_t start = nullptr;
    cudaEvent_t stop  = nullptr;
    if (cudaEventCreate(&start) != cudaSuccess) { return std::nullopt; }
    if (cudaEventCreate(&stop) != cudaSuccess) {
        (void)cudaEventDestroy(start);
        return std::nullopt;
    }
    bool ok = cudaEventRecord(start, stream) == cudaSuccess;
    for (int iteration = 0; ok && iteration < kTimed; ++iteration) { ok = attempt(); }
    ok = ok && cudaEventRecord(stop, stream) == cudaSuccess;
    ok = ok && cudaEventSynchronize(stop) == cudaSuccess;
    float milliseconds = 0.0f;
    ok = ok && cudaEventElapsedTime(&milliseconds, start, stop) == cudaSuccess;
    (void)cudaEventDestroy(start);
    (void)cudaEventDestroy(stop);
    if (!ok) {
        (void)cudaGetLastError();
        return std::nullopt;
    }
    return milliseconds / static_cast<float>(kTimed);
}

/// GEMM_1 is tuned against a fixed GEMM_2 and then GEMM_2 against the winner, the sweep
/// FlashInfer's autotuner performs: the two GEMMs do not share a tile, so a full cross product
/// would cost |t1| * |t2| timings to reach the same pair.
void tune_bucket(const Geometry& geometry, std::int32_t bucket, const Nvfp4RoutedExperts& experts,
                 const TuningBuffers& buffers, const WorkspaceLayout& layout, cudaStream_t stream) {
    State& s                                            = state();
    const std::vector<tce::CutlassGemmConfig>& gemm1_all = tactics_for(1);
    const std::vector<tce::CutlassGemmConfig>& gemm2_all = tactics_for(2);
    if (gemm1_all.empty() || gemm2_all.empty()) {
        throw std::runtime_error("trtllm_moe: the vendored runner offers no tactics");
    }

    // A tactic that cannot run this shape is skipped, so the sweep needs a partner that is known
    // to run before it can compare GEMM_1 candidates at all.
    std::optional<tce::CutlassGemmConfig> best_gemm1;
    std::optional<tce::CutlassGemmConfig> best_gemm2;
    float best = 0.0f;
    for (const tce::CutlassGemmConfig& candidate : gemm1_all) {
        for (const tce::CutlassGemmConfig& partner : gemm2_all) {
            const std::optional<float> milliseconds = time_candidate(
                geometry, bucket, experts, buffers, layout, candidate, partner, stream);
            if (!milliseconds.has_value()) { continue; }
            best       = *milliseconds;
            best_gemm1 = candidate;
            best_gemm2 = partner;
            break;
        }
        if (best_gemm1.has_value()) { break; }
    }
    if (!best_gemm1.has_value()) {
        throw std::runtime_error("trtllm_moe: no tactic pair ran at width " +
                                 std::to_string(bucket));
    }

    for (const tce::CutlassGemmConfig& candidate : gemm1_all) {
        const std::optional<float> milliseconds = time_candidate(
            geometry, bucket, experts, buffers, layout, candidate, *best_gemm2, stream);
        if (milliseconds.has_value() && *milliseconds < best) {
            best       = *milliseconds;
            best_gemm1 = candidate;
        }
    }
    for (const tce::CutlassGemmConfig& candidate : gemm2_all) {
        const std::optional<float> milliseconds = time_candidate(
            geometry, bucket, experts, buffers, layout, *best_gemm1, candidate, stream);
        if (milliseconds.has_value() && *milliseconds < best) {
            best       = *milliseconds;
            best_gemm2 = candidate;
        }
    }

    s.tactics[TacticKey{bucket, 1}] = *best_gemm1;
    s.tactics[TacticKey{bucket, 2}] = *best_gemm2;
}

void allocate_tuning_buffers(const Geometry& geometry, std::int32_t max_tokens,
                             const WorkspaceLayout& layout, TuningBuffers& buffers,
                             cudaStream_t stream) {
    const std::size_t rows = static_cast<std::size_t>(max_tokens);
    CUDA_CHECK(cudaMalloc(&buffers.x, rows * geometry.hidden * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&buffers.ids,
                          rows * geometry.experts_per_token * sizeof(std::int32_t)));
    CUDA_CHECK(cudaMalloc(&buffers.scales, rows * geometry.experts_per_token * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&buffers.workspace, layout.total));
    buffers.bytes = layout.total;

    const std::int64_t values = static_cast<std::int64_t>(rows) * geometry.hidden;
    tuning_input_kernel<<<static_cast<unsigned>((values + 255) / 256), 256, 0, stream>>>(
        buffers.x, values, 0x5eedU);
    tuning_route_kernel<<<static_cast<unsigned>((max_tokens + 127) / 128), 128, 0, stream>>>(
        buffers.ids, buffers.scales, max_tokens, geometry.experts, geometry.experts_per_token);
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

void require_geometry(const Geometry& geometry) {
    if (geometry.hidden <= 0 || geometry.experts <= 0 || geometry.experts_per_token <= 0 ||
        geometry.intermediate <= 0) {
        throw std::invalid_argument("trtllm_moe: invalid geometry");
    }
}

} // namespace

bool available() noexcept { return true; }

std::int32_t bucket_of(std::int32_t tokens) {
    if (tokens < 1 || tokens > kMaxTokens) {
        throw std::invalid_argument("trtllm_moe: token count " + std::to_string(tokens) +
                                    " is outside [1, " + std::to_string(kMaxTokens) + "]");
    }
    for (const std::int32_t bucket : ladder()) {
        if (tokens <= bucket) { return bucket; }
    }
    return kMaxTokens;
}

std::size_t workspace_bytes(const Geometry& geometry, std::int32_t max_tokens) {
    require_geometry(geometry);
    if (max_tokens < 1 || max_tokens > kMaxTokens) {
        throw std::invalid_argument("trtllm_moe: workspace width out of range");
    }
    std::lock_guard<std::mutex> guard(state().mutex);
    return workspace_layout(geometry, max_tokens).total;
}

void prepare(const Geometry& geometry, const Nvfp4RoutedExperts& sample, std::int32_t max_tokens,
             cudaStream_t stream) {
    require_geometry(geometry);
    if (max_tokens < 1) { return; }
    max_tokens = std::min(max_tokens, kMaxTokens);

    cudaStreamCaptureStatus capture = cudaStreamCaptureStatusNone;
    CUDA_CHECK(cudaStreamIsCapturing(stream, &capture));
    if (capture != cudaStreamCaptureStatusNone) {
        throw std::logic_error("trtllm_moe: prepare cannot run inside a stream capture");
    }

    std::lock_guard<std::mutex> guard(state().mutex);
    load_cache(geometry);

    // Every bucket a round of up to `max_tokens` rows can land in, which is the ladder up to and
    // including the bucket `max_tokens` itself falls in.
    const std::int32_t last = bucket_of(max_tokens);
    std::vector<std::int32_t> missing;
    for (const std::int32_t bucket : ladder()) {
        if (bucket > last) { break; }
        if (state().tactics.find(TacticKey{bucket, 1}) == state().tactics.end() ||
            state().tactics.find(TacticKey{bucket, 2}) == state().tactics.end()) {
            missing.push_back(bucket);
        }
    }
    if (missing.empty()) { return; }

    const WorkspaceLayout layout = workspace_layout(geometry, max_tokens);
    TuningBuffers buffers;
    allocate_tuning_buffers(geometry, max_tokens, layout, buffers, stream);
    for (const std::int32_t bucket : missing) {
        tune_bucket(geometry, bucket, sample, buffers, layout, stream);
    }
    store_cache();
}

void run(const Geometry& geometry, const __nv_bfloat16* x, std::int32_t tokens,
         const std::int32_t* ids, const float* final_scales, const Nvfp4RoutedExperts& experts,
         void* workspace, std::size_t workspace_capacity, float* routed_sum, cudaStream_t stream) {
    require_geometry(geometry);
    const std::int32_t bucket = bucket_of(tokens);

    std::lock_guard<std::mutex> guard(state().mutex);
    load_cache(geometry);

    const WorkspaceLayout layout = workspace_layout(geometry, tokens);
    if (workspace == nullptr || workspace_capacity < layout.total) {
        throw std::invalid_argument("trtllm_moe: insufficient workspace");
    }

    auto gemm1 = state().tactics.find(TacticKey{bucket, 1});
    auto gemm2 = state().tactics.find(TacticKey{bucket, 2});
    if (gemm1 == state().tactics.end() || gemm2 == state().tactics.end()) {
        cudaStreamCaptureStatus capture = cudaStreamCaptureStatusNone;
        CUDA_CHECK(cudaStreamIsCapturing(stream, &capture));
        if (capture != cudaStreamCaptureStatusNone) {
            // Tuning launches and synchronises; doing that mid-capture would corrupt the graph,
            // and silently running an arbitrary tactic would freeze it into the replay.
            throw std::logic_error("trtllm_moe: width bucket " + std::to_string(bucket) +
                                   " was not prepared before capture");
        }
        const WorkspaceLayout tuning_layout = workspace_layout(geometry, bucket);
        TuningBuffers buffers;
        allocate_tuning_buffers(geometry, bucket, tuning_layout, buffers, stream);
        tune_bucket(geometry, bucket, experts, buffers, tuning_layout, stream);
        store_cache();
        gemm1 = state().tactics.find(TacticKey{bucket, 1});
        gemm2 = state().tactics.find(TacticKey{bucket, 2});
    }

    auto* base            = static_cast<char*>(workspace);
    auto* output          = reinterpret_cast<__nv_bfloat16*>(base + layout.output_offset);
    auto* permutation_map = reinterpret_cast<std::int32_t*>(base + layout.map_offset);
    state().runner.setTactic(gemm1->second, gemm2->second);
    launch(geometry, x, tokens, ids, final_scales, experts, base + layout.runner_offset, output,
           permutation_map, stream);
    widen(output, routed_sum, static_cast<std::int64_t>(tokens) * geometry.hidden, stream);
}

} // namespace sinfer::ops::detail::trtllm_moe
