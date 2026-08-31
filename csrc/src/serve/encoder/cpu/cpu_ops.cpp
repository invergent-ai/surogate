#include "encoder/cpu/cpu_ops.h"

#if defined(SINFER_WITH_ZENDNN)
#include <zendnnl.hpp>
#endif
#if defined(SINFER_WITH_ONEDNN)
#include <oneapi/dnnl/dnnl.hpp>
#endif

#include <algorithm>
#include <atomic>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <limits>
#include <stdexcept>
#include <cstdlib>
#include <unordered_map>
#include <unordered_set>

#if defined(__x86_64__)
#include <immintrin.h>
#include <sched.h>
#endif

namespace sinfer::encoder::cpu {
namespace {

constexpr int kLanes = 16; // one AVX-512 float vector

/// The CPUs this process may run on, which is the honest starting point: a
/// caller that has already pinned us with taskset or numactl has said what it
/// wants, and guessing past that is how a benchmark measures the wrong thing.
std::vector<int> allowed_cpus() {
    std::vector<int> cpus;
#if defined(__x86_64__)
    cpu_set_t set;
    CPU_ZERO(&set);
    if (sched_getaffinity(0, sizeof(set), &set) == 0) {
        for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu) {
            if (CPU_ISSET(cpu, &set)) { cpus.push_back(cpu); }
        }
    }
#endif
    if (cpus.empty()) {
        const unsigned hw = std::max(1U, std::thread::hardware_concurrency());
        for (unsigned cpu = 0; cpu < hw; ++cpu) { cpus.push_back(static_cast<int>(cpu)); }
    }
    return cpus;
}

/// One entry per *physical* core: sysfs lists a core's SMT siblings together,
/// so keeping the first of each distinct list deduplicates threads without
/// having to interpret the topology.
std::vector<int> physical_cores(const std::vector<int>& candidates) {
    std::unordered_set<std::string> seen;
    std::vector<int> cores;
    for (const int cpu : candidates) {
        const std::string path =
            "/sys/devices/system/cpu/cpu" + std::to_string(cpu) + "/topology/thread_siblings_list";
        std::ifstream file(path);
        std::string siblings;
        if (!file || !std::getline(file, siblings)) {
            cores.push_back(cpu); // no topology to read; take it at face value
            continue;
        }
        if (seen.insert(siblings).second) { cores.push_back(cpu); }
    }
    return cores.empty() ? candidates : cores;
}

/// Keep only the cores sharing one NUMA node. A 1.2 GB model is node-local, and
/// a barrier per matmul makes the far socket's latency everyone's latency.
std::vector<int> one_numa_node(const std::vector<int>& cores) {
    std::vector<std::pair<int, int>> by_node; // (node, cpu)
    for (const int cpu : cores) {
        int node = 0;
        for (int candidate = 0; candidate < 16; ++candidate) {
            const std::string path = "/sys/devices/system/node/node" + std::to_string(candidate) +
                                     "/cpu" + std::to_string(cpu);
            if (std::ifstream(path).good() || std::ifstream(path + "/online").good()) {
                node = candidate;
                break;
            }
        }
        by_node.emplace_back(node, cpu);
    }
    const int first = by_node.empty() ? 0 : by_node.front().first;
    std::vector<int> out;
    for (const auto& [node, cpu] : by_node) {
        if (node == first) { out.push_back(cpu); }
    }
    return out.empty() ? cores : out;
}


// --- dot products -----------------------------------------------------------
//
// The file is compiled generic; only these carry AVX-512, selected once at
// startup. `cpu_expert_compute.cpp` does the same, and for the same reason: a
// binary built here has to start on a host without the instructions.

float dot_scalar(const float* a, const float* b, std::int32_t n) {
    float sum = 0.0F;
    for (std::int32_t i = 0; i < n; ++i) { sum += a[i] * b[i]; }
    return sum;
}

void axpy_scalar(float weight, const float* x, float* out, std::int32_t n) {
    for (std::int32_t i = 0; i < n; ++i) { out[i] += weight * x[i]; }
}

#if defined(__x86_64__)
__attribute__((target("avx512f,avx512bw,avx512vl,avx512dq,fma")))
float dot_avx512(const float* a, const float* b, std::int32_t n) {
    __m512 acc0 = _mm512_setzero_ps();
    __m512 acc1 = _mm512_setzero_ps();
    std::int32_t i = 0;
    // Two accumulators: one FMA chain cannot cover the unit's latency.
    for (; i + 2 * kLanes <= n; i += 2 * kLanes) {
        acc0 = _mm512_fmadd_ps(_mm512_loadu_ps(a + i), _mm512_loadu_ps(b + i), acc0);
        acc1 = _mm512_fmadd_ps(_mm512_loadu_ps(a + i + kLanes), _mm512_loadu_ps(b + i + kLanes),
                               acc1);
    }
    for (; i + kLanes <= n; i += kLanes) {
        acc0 = _mm512_fmadd_ps(_mm512_loadu_ps(a + i), _mm512_loadu_ps(b + i), acc0);
    }
    float sum = _mm512_reduce_add_ps(_mm512_add_ps(acc0, acc1));
    for (; i < n; ++i) { sum += a[i] * b[i]; }
    return sum;
}

__attribute__((target("avx512f,avx512bw,avx512vl,avx512dq,fma")))
void axpy_avx512(float weight, const float* x, float* out, std::int32_t n) {
    const __m512 w = _mm512_set1_ps(weight);
    std::int32_t i = 0;
    for (; i + kLanes <= n; i += kLanes) {
        _mm512_storeu_ps(out + i,
                         _mm512_fmadd_ps(w, _mm512_loadu_ps(x + i), _mm512_loadu_ps(out + i)));
    }
    for (; i < n; ++i) { out[i] += weight * x[i]; }
}

bool have_avx512() {
    static const bool yes = __builtin_cpu_supports("avx512f") &&
                            __builtin_cpu_supports("avx512bw") &&
                            __builtin_cpu_supports("avx512vl") &&
                            __builtin_cpu_supports("avx512dq");
    return yes;
}
#endif

inline float dot(const float* a, const float* b, std::int32_t n) {
#if defined(__x86_64__)
    if (have_avx512()) { return dot_avx512(a, b, n); }
#endif
    return dot_scalar(a, b, n);
}

inline void axpy(float weight, const float* x, float* out, std::int32_t n) {
#if defined(__x86_64__)
    if (have_avx512()) { return axpy_avx512(weight, x, out, n); }
#endif
    axpy_scalar(weight, x, out, n);
}

inline float widen(std::uint16_t bits) {
    const std::uint32_t word = static_cast<std::uint32_t>(bits) << 16U;
    float value              = 0.0F;
    std::memcpy(&value, &word, sizeof(value));
    return value;
}

std::uint16_t narrow_one(float value) {
    std::uint32_t word = 0;
    std::memcpy(&word, &value, sizeof(word));
    // Round to nearest even, matching every producer of BF16 in this engine.
    word += 0x7FFFU + ((word >> 16U) & 1U);
    return static_cast<std::uint16_t>(word >> 16U);
}

inline float dot_bf16_scalar(const std::uint16_t* w, const std::uint16_t* x, std::int32_t n) {
    float sum = 0.0F;
    for (std::int32_t i = 0; i < n; ++i) { sum += widen(w[i]) * widen(x[i]); }
    return sum;
}

#if defined(__x86_64__)
/// Widen 16 BF16 to FP32: shift each into the high half of a 32-bit lane. Two
/// instructions, and it halves what the GEMM streams from memory.
__attribute__((target("avx512f,avx512bw,avx512vl,avx512dq,fma")))
inline __m512 widen16(const std::uint16_t* p) {
    const __m256i raw = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p));
    return _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(raw), 16));
}

__attribute__((target("avx512f,avx512bw,avx512vl,avx512dq,fma")))
float dot_bf16_avx512(const std::uint16_t* w, const std::uint16_t* x, std::int32_t n) {
    __m512 acc     = _mm512_setzero_ps();
    std::int32_t i = 0;
    for (; i + kLanes <= n; i += kLanes) {
        acc = _mm512_fmadd_ps(widen16(w + i), widen16(x + i), acc);
    }
    float sum = _mm512_reduce_add_ps(acc);
    for (; i < n; ++i) { sum += widen(w[i]) * widen(x[i]); }
    return sum;
}

/// 4 BF16 weight rows x 4 BF16 token columns, accumulated in FP32 registers.
///
/// A dot-product GEMM issues one FMA per two loads and pays a horizontal
/// reduction per output; blocking four by four turns eight loads per k-step into
/// sixteen FMAs and amortises the reductions sixteen ways. BF16 halves the bytes
/// each of those loads moves.
__attribute__((target("avx512f,avx512bw,avx512vl,avx512dq,fma")))
void gemm_bf16_4x4_avx512(const std::uint16_t* w0, const std::uint16_t* w1,
                          const std::uint16_t* w2, const std::uint16_t* w3,
                          const std::uint16_t* x0, const std::uint16_t* x1,
                          const std::uint16_t* x2, const std::uint16_t* x3, std::int32_t k,
                          float* out, std::int32_t out_stride) {
    __m512 acc[4][4];
    for (int r = 0; r < 4; ++r) {
        for (int c = 0; c < 4; ++c) { acc[r][c] = _mm512_setzero_ps(); }
    }
    std::int32_t i = 0;
    for (; i + kLanes <= k; i += kLanes) {
        const __m512 wv0 = widen16(w0 + i);
        const __m512 wv1 = widen16(w1 + i);
        const __m512 wv2 = widen16(w2 + i);
        const __m512 wv3 = widen16(w3 + i);
        const __m512 xv0 = widen16(x0 + i);
        const __m512 xv1 = widen16(x1 + i);
        const __m512 xv2 = widen16(x2 + i);
        const __m512 xv3 = widen16(x3 + i);
        acc[0][0] = _mm512_fmadd_ps(wv0, xv0, acc[0][0]);
        acc[0][1] = _mm512_fmadd_ps(wv0, xv1, acc[0][1]);
        acc[0][2] = _mm512_fmadd_ps(wv0, xv2, acc[0][2]);
        acc[0][3] = _mm512_fmadd_ps(wv0, xv3, acc[0][3]);
        acc[1][0] = _mm512_fmadd_ps(wv1, xv0, acc[1][0]);
        acc[1][1] = _mm512_fmadd_ps(wv1, xv1, acc[1][1]);
        acc[1][2] = _mm512_fmadd_ps(wv1, xv2, acc[1][2]);
        acc[1][3] = _mm512_fmadd_ps(wv1, xv3, acc[1][3]);
        acc[2][0] = _mm512_fmadd_ps(wv2, xv0, acc[2][0]);
        acc[2][1] = _mm512_fmadd_ps(wv2, xv1, acc[2][1]);
        acc[2][2] = _mm512_fmadd_ps(wv2, xv2, acc[2][2]);
        acc[2][3] = _mm512_fmadd_ps(wv2, xv3, acc[2][3]);
        acc[3][0] = _mm512_fmadd_ps(wv3, xv0, acc[3][0]);
        acc[3][1] = _mm512_fmadd_ps(wv3, xv1, acc[3][1]);
        acc[3][2] = _mm512_fmadd_ps(wv3, xv2, acc[3][2]);
        acc[3][3] = _mm512_fmadd_ps(wv3, xv3, acc[3][3]);
    }
    const std::uint16_t* w[4] = {w0, w1, w2, w3};
    const std::uint16_t* x[4] = {x0, x1, x2, x3};
    for (int r = 0; r < 4; ++r) {
        for (int c = 0; c < 4; ++c) {
            float sum = _mm512_reduce_add_ps(acc[r][c]);
            for (std::int32_t t = i; t < k; ++t) { sum += widen(w[r][t]) * widen(x[c][t]); }
            out[static_cast<std::int64_t>(c) * out_stride + r] = sum;
        }
    }
}
#endif

inline float dot_bf16(const std::uint16_t* w, const std::uint16_t* x, std::int32_t n) {
#if defined(__x86_64__)
    if (have_avx512()) { return dot_bf16_avx512(w, x, n); }
#endif
    return dot_bf16_scalar(w, x, n);
}

inline float gelu_tanh(float z) {
    constexpr float kSqrt2Pi = 0.79788456080286535588F;
    return 0.5F * z * (1.0F + std::tanh(kSqrt2Pi * (z + 0.044715F * z * z * z)));
}

} // namespace

ThreadPlan ThreadPlan::detect() {
    ThreadPlan plan;
    plan.cpus   = one_numa_node(physical_cores(allowed_cpus()));
    plan.threads = static_cast<int>(plan.cpus.size());
    return plan;
}

// --- pool -------------------------------------------------------------------

struct ThreadPool::Impl {
    std::vector<std::thread> workers;
    std::mutex mutex;
    std::condition_variable wake;
    std::condition_variable done;

    const std::function<void(std::int64_t, std::int64_t)>* body = nullptr;
    std::int64_t count    = 0;
    std::uint64_t epoch   = 0;
    int outstanding       = 0;
    bool stopping         = false;
    int threads           = 1;
    std::vector<int> cpus;

    void worker(int index) {
        if (!cpus.empty()) {
#if defined(__x86_64__)
            cpu_set_t set;
            CPU_ZERO(&set);
            CPU_SET(cpus[static_cast<std::size_t>(index) % cpus.size()], &set);
            sched_setaffinity(0, sizeof(set), &set);
#endif
        }
        std::uint64_t seen = 0;
        while (true) {
            const std::function<void(std::int64_t, std::int64_t)>* job = nullptr;
            std::int64_t total = 0;
            {
                std::unique_lock<std::mutex> lock(mutex);
                wake.wait(lock, [&] { return stopping || epoch != seen; });
                if (stopping) { return; }
                seen  = epoch;
                job   = body;
                total = count;
            }
            // The worker's own slice. Index 0 is the calling thread's share.
            const std::int64_t per   = (total + threads - 1) / threads;
            const std::int64_t begin = std::min(total, per * (index + 1));
            const std::int64_t end   = std::min(total, begin + per);
            if (begin < end) { (*job)(begin, end); }
            {
                std::lock_guard<std::mutex> lock(mutex);
                if (--outstanding == 0) { done.notify_one(); }
            }
        }
    }
};

ThreadPool::ThreadPool(ThreadPlan plan) : impl_(std::make_unique<Impl>()) {
    impl_->threads = std::max(1, plan.threads);
    impl_->cpus    = std::move(plan.cpus);
    // One fewer worker than threads: the caller runs a slice too, which keeps a
    // single-threaded plan free of any handoff at all.
    for (int index = 0; index + 1 < impl_->threads; ++index) {
        impl_->workers.emplace_back([impl = impl_.get(), index] { impl->worker(index); });
    }
}

ThreadPool::~ThreadPool() {
    {
        std::lock_guard<std::mutex> lock(impl_->mutex);
        impl_->stopping = true;
    }
    impl_->wake.notify_all();
    for (std::thread& worker : impl_->workers) {
        if (worker.joinable()) { worker.join(); }
    }
}

int ThreadPool::threads() const noexcept { return impl_->threads; }

void ThreadPool::parallel_for(std::int64_t count,
                              const std::function<void(std::int64_t, std::int64_t)>& body) {
    if (count <= 0) { return; }
    if (impl_->workers.empty()) {
        body(0, count);
        return;
    }
    {
        std::lock_guard<std::mutex> lock(impl_->mutex);
        impl_->body        = &body;
        impl_->count       = count;
        impl_->outstanding = static_cast<int>(impl_->workers.size());
        ++impl_->epoch;
    }
    impl_->wake.notify_all();

    const std::int64_t per = (count + impl_->threads - 1) / impl_->threads;
    if (per > 0) { body(0, std::min(count, per)); }

    std::unique_lock<std::mutex> lock(impl_->mutex);
    impl_->done.wait(lock, [&] { return impl_->outstanding == 0; });
}

// --- kernels ----------------------------------------------------------------

#if defined(SINFER_WITH_ZENDNN)
namespace {

/// out[n, tokens] = w[n, k] . w x[k, tokens], through AMD's tuned kernels.
///
/// The shapes line up without a copy. Activations and outputs are
/// [rows, tokens] with rows contiguous, which reads as row-major
/// [tokens, rows]; so C[tokens, n] = x[tokens, k] * w[n, k]^T is the call, and
/// `transB` takes the weight exactly as the artifact stores it.
/// `is_weights_const` lets the library prepack a weight once and keep the
/// packing -- something an inference workload can promise and a training one
/// cannot.
bool gemm_zendnn(const std::uint16_t* w, const std::uint16_t* x, float* out, std::int32_t n,
                 std::int32_t k, std::int32_t tokens, int threads) {
    using namespace zendnnl::lowoha::matmul;
    matmul_data_types dtypes;
    dtypes.src     = zendnnl::data_type_t::bf16;
    dtypes.wei     = zendnnl::data_type_t::bf16;
    dtypes.dst     = zendnnl::data_type_t::f32;
    dtypes.bias    = zendnnl::data_type_t::none;
    dtypes.compute = zendnnl::data_type_t::none;

    matmul_params params;
    params.dtypes      = dtypes;
    params.num_threads = threads;

    matmul_batch_params_t batch;
    batch.Batch_A = 1;
    batch.Batch_B = 1;

    const auto status = matmul_direct('r', /*transA*/ false, /*transB*/ true, tokens, n, k, 1.0F,
                                      x, k, w, k, nullptr, 0.0F, out, n,
                                      /*is_weights_const*/ true, batch, params);
    return status == zendnnl::status_t::success;
}

} // namespace
#endif


#if defined(SINFER_WITH_ONEDNN)
namespace {

/// The same product through oneDNN, which is what Intel's stack -- OpenVINO's
/// CPU plugin included -- runs underneath.
///
/// oneDNN caches primitives internally, so the descriptor work here is paid
/// once per distinct shape rather than per call. The weight is described as
/// [n, k] with an `ab` (row-major) layout and the product asks for its
/// transpose by giving the destination `[tokens, n]`, exactly as the ZenDNN
/// path does.
/// One cached primitive per (n, k, tokens, fused-activation) shape.
///
/// A forward runs 168 of these, and a stack-built primitive_desc pays kernel
/// selection every time. The encoder only ever sees a handful of distinct
/// shapes, so they are built once and looked up.
struct MatmulKey {
    std::int32_t n, k, tokens;
    bool fused_gelu_mul;
    bool operator==(const MatmulKey& other) const noexcept {
        return n == other.n && k == other.k && tokens == other.tokens &&
               fused_gelu_mul == other.fused_gelu_mul;
    }
};
struct MatmulKeyHash {
    std::size_t operator()(const MatmulKey& key) const noexcept {
        return (static_cast<std::size_t>(key.n) * 0x9E3779B1u) ^
               (static_cast<std::size_t>(key.k) * 0x85EBCA77u) ^
               (static_cast<std::size_t>(key.tokens) * 0xC2B2AE3Du) ^
               (key.fused_gelu_mul ? 0x27D4EB2Fu : 0u);
    }
};
struct CachedMatmul {
    dnnl::memory::desc src, weight, dst, binary;
    dnnl::matmul primitive;
};

const CachedMatmul& cached_matmul(const dnnl::engine& cpu_engine, std::int32_t n, std::int32_t k,
                                  std::int32_t tokens, bool fused_gelu_mul) {
    using namespace dnnl;
    static std::unordered_map<MatmulKey, CachedMatmul, MatmulKeyHash> cache;
    static std::mutex cache_mutex;
    const MatmulKey key{n, k, tokens, fused_gelu_mul};
    const std::lock_guard<std::mutex> lock(cache_mutex);
    const auto found = cache.find(key);
    if (found != cache.end()) { return found->second; }

    // `ba` on the weight is the transpose: the buffer is [n, k] row-major, which
    // is [k, n] with the strides swapped. No copy.
    CachedMatmul entry{
        memory::desc({tokens, k}, memory::data_type::bf16, memory::format_tag::ab),
        memory::desc({k, n}, memory::data_type::bf16, memory::format_tag::ba),
        memory::desc({tokens, n}, memory::data_type::f32, memory::format_tag::ab),
        memory::desc({tokens, n}, memory::data_type::f32, memory::format_tag::ab),
        matmul{}};

    primitive_attr attributes;
    // BF16 compute over FP32 storage. Zen 4 has avx512_bf16, and AMD's own
    // guidance is that BF16 is the fastest datatype there; the weights came from
    // int8 codes with an fp16 scale, so their precision was never near FP32 to
    // begin with and rounding the multiply to BF16 costs nothing real. Off by
    // default because it is a numerical change, and a numerical change should be
    // asked for.
    if (std::getenv("SINFER_CPU_BF16") != nullptr) {
        attributes.set_fpmath_mode(fpmath_mode::bf16, /*apply_to_int*/ true);
    }
    if (fused_gelu_mul) {
        // Gemma's MLP is gelu(gate) * up. Both halves are post-ops here, so the
        // activation and the gating multiply ride the gate projection instead of
        // costing two more passes over an [intermediate, tokens] buffer.
        post_ops ops;
        ops.append_eltwise(algorithm::eltwise_gelu_tanh, 0.0F, 0.0F);
        ops.append_binary(algorithm::binary_mul, entry.binary);
        attributes.set_post_ops(ops);
    }
    const matmul::primitive_desc pd(cpu_engine, entry.src, entry.weight, entry.dst, attributes);
    entry.primitive = matmul(pd);
    return cache.emplace(key, std::move(entry)).first->second;
}

bool gemm_onednn(const std::uint16_t* w, const std::uint16_t* x, float* out, std::int32_t n,
                 std::int32_t k, std::int32_t tokens, const float* gate_by) {
    using namespace dnnl;
    static engine cpu_engine(engine::kind::cpu, 0);
    static stream cpu_stream(cpu_engine);

    const CachedMatmul& entry = cached_matmul(cpu_engine, n, k, tokens, gate_by != nullptr);
    memory src_mem(entry.src, cpu_engine, const_cast<std::uint16_t*>(x));
    memory weight_mem(entry.weight, cpu_engine, const_cast<std::uint16_t*>(w));
    memory dst_mem(entry.dst, cpu_engine, out);

    std::unordered_map<int, memory> args{{DNNL_ARG_SRC, src_mem},
                                         {DNNL_ARG_WEIGHTS, weight_mem},
                                         {DNNL_ARG_DST, dst_mem}};
    memory binary_mem;
    if (gate_by != nullptr) {
        binary_mem = memory(entry.binary, cpu_engine, const_cast<float*>(gate_by));
        args[DNNL_ARG_ATTR_MULTIPLE_POST_OP(1) | DNNL_ARG_SRC_1] = binary_mem;
    }
    entry.primitive.execute(cpu_stream, args);
    // No wait: a oneDNN CPU stream is synchronous, so execute() has already
    // returned with the result written; waiting only cycles the OpenMP team.
    return true;
}

} // namespace
#endif

GemmBackend gemm_backend() {
    static const GemmBackend chosen = [] {
        // An override, so either vendor path can be exercised on whatever
        // machine is to hand -- correctness does not depend on running the
        // matching silicon, and being unable to test the other vendor's path is
        // how it rots.
        if (const char* forced = std::getenv("SINFER_CPU_GEMM")) {
            const std::string name(forced);
#if defined(SINFER_WITH_ZENDNN)
            if (name == "zendnn") { return GemmBackend::ZenDnn; }
#endif
#if defined(SINFER_WITH_ONEDNN)
            if (name == "onednn") { return GemmBackend::OneDnn; }
#endif
            if (name == "builtin") { return GemmBackend::Builtin; }
        }
#if defined(__x86_64__)
        __builtin_cpu_init();
#if defined(SINFER_WITH_ZENDNN)
        if (__builtin_cpu_is("amd")) { return GemmBackend::ZenDnn; }
#endif
#if defined(SINFER_WITH_ONEDNN)
        // Intel, and anything not recognised as AMD: oneDNN is what Intel's own
        // stack runs underneath, OpenVINO's CPU plugin included.
        return GemmBackend::OneDnn;
#endif
#endif
        return GemmBackend::Builtin;
    }();
    return chosen;
}

const char* gemm_backend_name() {
    switch (gemm_backend()) {
    case GemmBackend::ZenDnn:
        return "zendnn";
    case GemmBackend::OneDnn:
        return "onednn";
    case GemmBackend::Builtin:
        break;
    }
    return "builtin-avx512";
}


void gemm(const std::uint16_t* w, const std::uint16_t* x, float* out, std::int32_t n,
          std::int32_t k, std::int32_t tokens, ThreadPool& pool) {
    // A vendor backend that declines a shape falls through to the builtin, so a
    // build with one is never *less* capable than a build without.
    switch (gemm_backend()) {
    case GemmBackend::ZenDnn:
#if defined(SINFER_WITH_ZENDNN)
        if (gemm_zendnn(w, x, out, n, k, tokens, pool.threads())) { return; }
#endif
        break;
    case GemmBackend::OneDnn:
#if defined(SINFER_WITH_ONEDNN)
        if (gemm_onednn(w, x, out, n, k, tokens, nullptr)) { return; }
#endif
        break;
    case GemmBackend::Builtin:
        break;
    }

    // One barrier for the whole GEMM: partition the output rows once, and let
    // each thread walk its own rows against every token. An earlier version
    // tiled tokens in an outer loop and paid a barrier per tile -- 1,512 of them
    // per forward -- which cost more than the tiling saved.
    constexpr std::int32_t kBlock = 4;
    const std::int64_t blocks     = (n + kBlock - 1) / kBlock;

    pool.parallel_for(blocks, [&](std::int64_t begin, std::int64_t end) {
        for (std::int64_t block = begin; block < end; ++block) {
            const std::int32_t row0 = static_cast<std::int32_t>(block) * kBlock;
            const std::int32_t rows = std::min(kBlock, n - row0);
#if defined(__x86_64__)
            if (have_avx512() && rows == kBlock) {
                const std::uint16_t* w0 = w + static_cast<std::int64_t>(row0 + 0) * k;
                const std::uint16_t* w1 = w + static_cast<std::int64_t>(row0 + 1) * k;
                const std::uint16_t* w2 = w + static_cast<std::int64_t>(row0 + 2) * k;
                const std::uint16_t* w3 = w + static_cast<std::int64_t>(row0 + 3) * k;
                std::int32_t t  = 0;
                for (; t + kBlock <= tokens; t += kBlock) {
                    gemm_bf16_4x4_avx512(w0, w1, w2, w3, x + static_cast<std::int64_t>(t + 0) * k,
                                    x + static_cast<std::int64_t>(t + 1) * k,
                                    x + static_cast<std::int64_t>(t + 2) * k,
                                    x + static_cast<std::int64_t>(t + 3) * k, k,
                                    out + static_cast<std::int64_t>(t) * n + row0, n);
                }
                for (; t < tokens; ++t) { // ragged tail of tokens
                    for (std::int32_t r = 0; r < rows; ++r) {
                        out[static_cast<std::int64_t>(t) * n + row0 + r] =
                            dot_bf16(w + static_cast<std::int64_t>(row0 + r) * k,
                                     x + static_cast<std::int64_t>(t) * k, k);
                    }
                }
                continue;
            }
#endif
            for (std::int32_t r = 0; r < rows; ++r) {
                const std::uint16_t* w_row = w + static_cast<std::int64_t>(row0 + r) * k;
                for (std::int32_t t = 0; t < tokens; ++t) {
                    out[static_cast<std::int64_t>(t) * n + row0 + r] =
                        dot_bf16(w_row, x + static_cast<std::int64_t>(t) * k, k);
                }
            }
        }
    });
}

bool gemm_gelu_mul(const std::uint16_t* w, const std::uint16_t* x, const float* up, float* out,
                   std::int32_t n, std::int32_t k, std::int32_t tokens) {
#if defined(SINFER_WITH_ONEDNN)
    if (gemm_backend() == GemmBackend::OneDnn) {
        return gemm_onednn(w, x, out, n, k, tokens, up);
    }
#endif
    return false; // caller falls back to gemm + gelu_mul
}

void embed(const std::uint16_t* table, const std::int32_t* ids, float* out, std::int32_t hidden,
           std::int32_t tokens, std::int32_t vocab) {
    for (std::int32_t t = 0; t < tokens; ++t) {
        const std::int32_t id = ids[t];
        if (id < 0 || id >= vocab) { throw std::out_of_range("embed: token id out of range"); }
        const std::uint16_t* row = table + static_cast<std::int64_t>(id) * hidden;
        float* target            = out + static_cast<std::int64_t>(t) * hidden;
        for (std::int32_t d = 0; d < hidden; ++d) { target[d] = widen(row[d]); }
    }
}

void rmsnorm(const float* x, const float* weight, float epsilon, bool unit_offset, float* out,
             std::int32_t rows, std::int32_t tokens, ThreadPool& pool) {
    // Every token's row is independent, and there are enough of them to be worth
    // spreading: at 543 tokens this ran serially and cost more than the GEMM it
    // feeds.
    pool.parallel_for(tokens, [&](std::int64_t begin, std::int64_t end) {
        for (std::int64_t t = begin; t < end; ++t) {
            const float* column = x + t * rows;
            float* target       = out + t * rows;
            double squares      = 0.0;
            for (std::int32_t i = 0; i < rows; ++i) {
                squares += static_cast<double>(column[i]) * column[i];
            }
            const float inverse =
                static_cast<float>(1.0 / std::sqrt(squares / static_cast<double>(rows) + epsilon));
            for (std::int32_t i = 0; i < rows; ++i) {
                const float gain = unit_offset ? 1.0F + weight[i] : weight[i];
                target[i]        = column[i] * inverse * gain;
            }
        }
    });
}

RopeTable::RopeTable(std::int32_t head_dim, std::int32_t max_tokens, float theta)
    : half_(head_dim / 2),
      cos_(static_cast<std::size_t>(max_tokens) * (head_dim / 2)),
      sin_(static_cast<std::size_t>(max_tokens) * (head_dim / 2)) {
    for (std::int32_t position = 0; position < max_tokens; ++position) {
        for (std::int32_t i = 0; i < half_; ++i) {
            const double inverse =
                std::pow(static_cast<double>(theta),
                         -2.0 * static_cast<double>(i) / static_cast<double>(head_dim));
            const double angle = static_cast<double>(position) * inverse;
            const std::size_t at = static_cast<std::size_t>(position) * half_ + i;
            cos_[at] = static_cast<float>(std::cos(angle));
            sin_[at] = static_cast<float>(std::sin(angle));
        }
    }
}

const float* RopeTable::cosines(std::int32_t position) const noexcept {
    return cos_.data() + static_cast<std::size_t>(position) * half_;
}
const float* RopeTable::sines(std::int32_t position) const noexcept {
    return sin_.data() + static_cast<std::size_t>(position) * half_;
}

void rope(float* x, const std::int32_t* positions, std::int32_t head_dim, std::int32_t heads,
          std::int32_t tokens, const RopeTable& table, ThreadPool& pool) {
    const std::int32_t half = head_dim / 2;
    pool.parallel_for(tokens, [&](std::int64_t begin, std::int64_t end) {
    for (std::int32_t t = static_cast<std::int32_t>(begin); t < end; ++t) {
        const float* cosines = table.cosines(positions[t]);
        const float* sines   = table.sines(positions[t]);
        for (std::int32_t head = 0; head < heads; ++head) {
            float* row = x + (static_cast<std::int64_t>(t) * heads + head) * head_dim;
            for (std::int32_t i = 0; i < half; ++i) {
                const float lo = row[i];
                const float hi = row[i + half];
                row[i]         = lo * cosines[i] - hi * sines[i];
                row[i + half]  = hi * cosines[i] + lo * sines[i];
            }
        }
    }
    });
}


#if defined(SINFER_WITH_ONEDNN)
namespace {

/// Attention's two products through oneDNN, with only the masked softmax left
/// to us.
///
/// Both are batched matmuls over the query heads, and both read Q, K and V
/// exactly where they already are -- oneDNN takes explicit strides, so the head
/// slices of a [tokens, heads*head_dim] buffer are described rather than copied,
/// and K's transpose is a stride swap. The single key/value head is a batch of
/// one and broadcasts across the query heads.
///
/// The softmax stays here because oneDNN's takes no mask, and a bidirectional
/// sliding layer is nothing but a mask. It is O(T^2) of cheap arithmetic against
/// the products' O(T^2 * d), so it is not where the time is.
bool attention_onednn(const std::uint16_t* q, const std::uint16_t* k, const std::uint16_t* v,
                      float* out, std::int32_t q_heads, std::int32_t head_dim,
                      std::int32_t tokens, std::int32_t window, float scale, float* scores,
                      ThreadPool& pool) {
    using namespace dnnl;
    static engine cpu_engine(engine::kind::cpu, 0);
    static stream cpu_stream(cpu_engine);

    const memory::dim heads = q_heads, T = tokens, D = head_dim;
    const memory::dim query_rows = q_heads * head_dim;

    // Q as [heads, T, D] over a [T, heads*D] buffer.
    const memory::desc q_md({heads, T, D}, memory::data_type::bf16, {D, query_rows, 1});
    // K^T as [1, D, T] over a [T, D] buffer: the transpose is the stride swap.
    const memory::desc kt_md({1, D, T}, memory::data_type::bf16, {1, 1, D});
    const memory::desc s_md({heads, T, T}, memory::data_type::f32, {T * T, T, 1});

    primitive_attr attributes;
    if (std::getenv("SINFER_CPU_BF16") != nullptr) {
        attributes.set_fpmath_mode(fpmath_mode::bf16, true);
    }

    {
        const matmul::primitive_desc pd(cpu_engine, q_md, kt_md, s_md, attributes);
        memory qm(q_md, cpu_engine, const_cast<std::uint16_t*>(q));
        memory km(kt_md, cpu_engine, const_cast<std::uint16_t*>(k));
        memory sm(s_md, cpu_engine, scores);
        matmul(pd).execute(cpu_stream,
                           {{DNNL_ARG_SRC, qm}, {DNNL_ARG_WEIGHTS, km}, {DNNL_ARG_DST, sm}});
        cpu_stream.wait();
    }

    // Masked softmax, in place over the last axis.
    pool.parallel_for(static_cast<std::int64_t>(heads) * T, [&](std::int64_t begin,
                                                               std::int64_t end) {
        for (std::int64_t unit = begin; unit < end; ++unit) {
            const auto query = static_cast<std::int32_t>(unit % T);
            float* row       = scores + unit * T;
            const std::int32_t lo = window > 0 ? std::max(0, query - window + 1) : 0;
            const std::int32_t hi = window > 0 ? std::min(tokens, query + window) : tokens;

            float maximum = -std::numeric_limits<float>::infinity();
            for (std::int32_t j = lo; j < hi; ++j) { maximum = std::max(maximum, row[j] * scale); }
            float sum = 0.0F;
            for (std::int32_t j = lo; j < hi; ++j) {
                row[j] = std::exp(row[j] * scale - maximum);
                sum += row[j];
            }
            const float inverse = sum > 0.0F ? 1.0F / sum : 0.0F;
            for (std::int32_t j = 0; j < tokens; ++j) {
                row[j] = (j >= lo && j < hi) ? row[j] * inverse : 0.0F;
            }
        }
    });

    {
        // V as [1, T, D], broadcast across heads; out as [heads, T, D] in place.
        // P stays FP32 (the softmax wrote it); only V narrows.
        const memory::desc v_md({1, T, D}, memory::data_type::bf16, {1, D, 1});
        const memory::desc o_md({heads, T, D}, memory::data_type::f32, {D, query_rows, 1});
        const matmul::primitive_desc pd(cpu_engine, s_md, v_md, o_md, attributes);
        memory sm(s_md, cpu_engine, scores);
        memory vm(v_md, cpu_engine, const_cast<std::uint16_t*>(v));
        memory om(o_md, cpu_engine, out);
        matmul(pd).execute(cpu_stream,
                           {{DNNL_ARG_SRC, sm}, {DNNL_ARG_WEIGHTS, vm}, {DNNL_ARG_DST, om}});
        cpu_stream.wait();
    }
    return true;
}

} // namespace
#endif

std::size_t attention_scratch(std::int32_t q_heads, std::int32_t tokens) {
    // The full score matrix: the library path materialises it, and the builtin
    // path needs nothing, so one size covers both.
    return static_cast<std::size_t>(q_heads) * static_cast<std::size_t>(tokens) *
           static_cast<std::size_t>(tokens);
}

void attention(const std::uint16_t* q, const std::uint16_t* k, const std::uint16_t* v, float* out,
               std::int32_t q_heads, std::int32_t head_dim, std::int32_t tokens,
               std::int32_t window, float scale, float* scratch, ThreadPool& pool) {
#if defined(SINFER_WITH_ONEDNN)
    if (gemm_backend() == GemmBackend::OneDnn &&
        attention_onednn(q, k, v, out, q_heads, head_dim, tokens, window, scale, scratch, pool)) {
        return;
    }
#endif
    (void)scratch; // the builtin path's per-query weights fit a thread-local buffer
    const std::int32_t query_rows = q_heads * head_dim;

    // One (head, query) pair per unit of work. Each writes its own slice of the
    // output and reads nothing another unit writes, so the only shared state is
    // the inputs.
    pool.parallel_for(static_cast<std::int64_t>(q_heads) * tokens,
                      [&](std::int64_t begin, std::int64_t end) {
        // At most `tokens` floats -- 8 KB at the 2048-token limit. Thread-local
        // rather than an argument so the pool needs no thread index, and reused
        // across units so the hot loop allocates nothing.
        static thread_local std::vector<float> weights;
        weights.resize(static_cast<std::size_t>(tokens));

        for (std::int64_t unit = begin; unit < end; ++unit) {
            const auto head  = static_cast<std::int32_t>(unit / tokens);
            const auto query = static_cast<std::int32_t>(unit % tokens);

            const std::int32_t lo = window > 0 ? std::max(0, query - window + 1) : 0;
            const std::int32_t hi = window > 0 ? std::min(tokens, query + window) : tokens;
            const std::uint16_t* q_row =
                q + static_cast<std::int64_t>(query) * query_rows + head * head_dim;

            float maximum = -std::numeric_limits<float>::infinity();
            for (std::int32_t key = lo; key < hi; ++key) {
                const std::uint16_t* k_col = k + static_cast<std::int64_t>(key) * head_dim;
                const float dot            = dot_bf16(q_row, k_col, head_dim);
                const float value                            = dot * scale;
                weights[static_cast<std::size_t>(key - lo)]  = value;
                maximum                                      = std::max(maximum, value);
            }

            float sum = 0.0F;
            for (std::int32_t key = lo; key < hi; ++key) {
                float& value = weights[static_cast<std::size_t>(key - lo)];
                value        = std::exp(value - maximum);
                sum += value;
            }
            const float inverse = sum > 0.0F ? 1.0F / sum : 0.0F;

            float* target = out + static_cast<std::int64_t>(query) * query_rows + head * head_dim;
            std::fill(target, target + head_dim, 0.0F);
            for (std::int32_t key = lo; key < hi; ++key) {
                const float weight = weights[static_cast<std::size_t>(key - lo)] * inverse;
                const std::uint16_t* v_col = v + static_cast<std::int64_t>(key) * head_dim;
                for (std::int32_t d = 0; d < head_dim; ++d) { target[d] += weight * widen(v_col[d]); }
            }
        }
    });
}

void narrow(const float* x, std::uint16_t* out, std::int64_t count, ThreadPool& pool) {
    pool.parallel_for(count, [&](std::int64_t begin, std::int64_t end) {
        for (std::int64_t i = begin; i < end; ++i) { out[i] = narrow_one(x[i]); }
    });
}

void gelu_mul(const float* gate, const float* up, float* out, std::int64_t count,
              ThreadPool& pool) {
    // A tanh per element, and at 543 tokens that is 15 million of them per
    // forward. Serially this was the largest single cost after the projections.
    pool.parallel_for(count, [&](std::int64_t begin, std::int64_t end) {
        for (std::int64_t i = begin; i < end; ++i) { out[i] = gelu_tanh(gate[i]) * up[i]; }
    });
}

void add(const float* y, float* x, std::int64_t count, ThreadPool& pool) {
    pool.parallel_for(count, [&](std::int64_t begin, std::int64_t end) {
        for (std::int64_t i = begin; i < end; ++i) { x[i] += y[i]; }
    });
}

void scale(float* x, float factor, std::int64_t count) {
    for (std::int64_t i = 0; i < count; ++i) { x[i] *= factor; }
}

void mean_pool(const float* x, float* out, std::int32_t hidden, std::int32_t count) {
    std::vector<double> sums(static_cast<std::size_t>(hidden), 0.0);
    for (std::int32_t t = 0; t < count; ++t) {
        const float* column = x + static_cast<std::int64_t>(t) * hidden;
        for (std::int32_t h = 0; h < hidden; ++h) { sums[static_cast<std::size_t>(h)] += column[h]; }
    }
    for (std::int32_t h = 0; h < hidden; ++h) {
        out[h] = static_cast<float>(sums[static_cast<std::size_t>(h)] / count);
    }
}

void l2norm(float* x, std::int32_t rows, std::int32_t columns, float epsilon) {
    for (std::int32_t c = 0; c < columns; ++c) {
        float* column  = x + static_cast<std::int64_t>(c) * rows;
        double squares = 0.0;
        for (std::int32_t i = 0; i < rows; ++i) { squares += static_cast<double>(column[i]) * column[i]; }
        const auto inverse = static_cast<float>(1.0 / std::sqrt(squares + epsilon));
        for (std::int32_t i = 0; i < rows; ++i) { column[i] *= inverse; }
    }
}

} // namespace sinfer::encoder::cpu
