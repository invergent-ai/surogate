#include "encoder/cpu/cpu_ops.h"

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


#if defined(__x86_64__)
/// 4 weight rows x 4 token columns, accumulated in registers.
///
/// A dot-product GEMM issues one FMA per two loads and then pays a horizontal
/// reduction for every single output, which leaves it load-bound well below the
/// FMA units. Blocking four by four turns eight loads per k-step into sixteen
/// FMAs and amortises the reductions sixteen ways -- the ratio Zen4's two loads
/// and two FMAs per cycle actually want.
__attribute__((target("avx512f,avx512bw,avx512vl,avx512dq,fma")))
void gemm_4x4_avx512(const float* w0, const float* w1, const float* w2, const float* w3,
                     const float* x0, const float* x1, const float* x2, const float* x3,
                     std::int32_t k, float* out, std::int32_t out_stride) {
    __m512 acc[4][4];
#pragma unroll
    for (int r = 0; r < 4; ++r) {
        for (int c = 0; c < 4; ++c) { acc[r][c] = _mm512_setzero_ps(); }
    }
    std::int32_t i = 0;
    for (; i + kLanes <= k; i += kLanes) {
        const __m512 wv0 = _mm512_loadu_ps(w0 + i);
        const __m512 wv1 = _mm512_loadu_ps(w1 + i);
        const __m512 wv2 = _mm512_loadu_ps(w2 + i);
        const __m512 wv3 = _mm512_loadu_ps(w3 + i);
        const __m512 xv0 = _mm512_loadu_ps(x0 + i);
        const __m512 xv1 = _mm512_loadu_ps(x1 + i);
        const __m512 xv2 = _mm512_loadu_ps(x2 + i);
        const __m512 xv3 = _mm512_loadu_ps(x3 + i);
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
    const float* w[4] = {w0, w1, w2, w3};
    const float* x[4] = {x0, x1, x2, x3};
    for (int r = 0; r < 4; ++r) {
        for (int c = 0; c < 4; ++c) {
            float sum = _mm512_reduce_add_ps(acc[r][c]);
            for (std::int32_t t = i; t < k; ++t) { sum += w[r][t] * x[c][t]; }
            out[static_cast<std::int64_t>(c) * out_stride + r] = sum;
        }
    }
}
#endif

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

void gemm(const float* w, const float* x, float* out, std::int32_t n, std::int32_t k,
          std::int32_t tokens, ThreadPool& pool) {
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
                const float* w0 = w + static_cast<std::int64_t>(row0 + 0) * k;
                const float* w1 = w + static_cast<std::int64_t>(row0 + 1) * k;
                const float* w2 = w + static_cast<std::int64_t>(row0 + 2) * k;
                const float* w3 = w + static_cast<std::int64_t>(row0 + 3) * k;
                std::int32_t t  = 0;
                for (; t + kBlock <= tokens; t += kBlock) {
                    gemm_4x4_avx512(w0, w1, w2, w3, x + static_cast<std::int64_t>(t + 0) * k,
                                    x + static_cast<std::int64_t>(t + 1) * k,
                                    x + static_cast<std::int64_t>(t + 2) * k,
                                    x + static_cast<std::int64_t>(t + 3) * k, k,
                                    out + static_cast<std::int64_t>(t) * n + row0, n);
                }
                for (; t < tokens; ++t) { // ragged tail of tokens
                    for (std::int32_t r = 0; r < rows; ++r) {
                        out[static_cast<std::int64_t>(t) * n + row0 + r] =
                            dot(w + static_cast<std::int64_t>(row0 + r) * k,
                                x + static_cast<std::int64_t>(t) * k, k);
                    }
                }
                continue;
            }
#endif
            for (std::int32_t r = 0; r < rows; ++r) {
                const float* w_row = w + static_cast<std::int64_t>(row0 + r) * k;
                for (std::int32_t t = 0; t < tokens; ++t) {
                    out[static_cast<std::int64_t>(t) * n + row0 + r] =
                        dot(w_row, x + static_cast<std::int64_t>(t) * k, k);
                }
            }
        }
    });
}

void embed(const float* table, const std::int32_t* ids, float* out, std::int32_t hidden,
           std::int32_t tokens, std::int32_t vocab) {
    for (std::int32_t t = 0; t < tokens; ++t) {
        const std::int32_t id = ids[t];
        if (id < 0 || id >= vocab) { throw std::out_of_range("embed: token id out of range"); }
        std::memcpy(out + static_cast<std::int64_t>(t) * hidden,
                    table + static_cast<std::int64_t>(id) * hidden,
                    static_cast<std::size_t>(hidden) * sizeof(float));
    }
}

void rmsnorm(const float* x, const float* weight, float epsilon, bool unit_offset, float* out,
             std::int32_t rows, std::int32_t tokens) {
    for (std::int32_t t = 0; t < tokens; ++t) {
        const float* column = x + static_cast<std::int64_t>(t) * rows;
        float* target       = out + static_cast<std::int64_t>(t) * rows;
        double squares      = 0.0;
        for (std::int32_t i = 0; i < rows; ++i) { squares += static_cast<double>(column[i]) * column[i]; }
        const float inverse =
            static_cast<float>(1.0 / std::sqrt(squares / static_cast<double>(rows) + epsilon));
        for (std::int32_t i = 0; i < rows; ++i) {
            const float gain = unit_offset ? 1.0F + weight[i] : weight[i];
            target[i]        = column[i] * inverse * gain;
        }
    }
}

void rope(float* x, const std::int32_t* positions, std::int32_t head_dim, std::int32_t heads,
          std::int32_t tokens, float theta) {
    const std::int32_t half = head_dim / 2;
    for (std::int32_t t = 0; t < tokens; ++t) {
        const auto position = static_cast<float>(positions[t]);
        for (std::int32_t head = 0; head < heads; ++head) {
            float* row = x + (static_cast<std::int64_t>(t) * heads + head) * head_dim;
            for (std::int32_t i = 0; i < half; ++i) {
                const float frequency =
                    position * std::pow(theta, -2.0F * static_cast<float>(i) /
                                                   static_cast<float>(head_dim));
                const float cosine = std::cos(frequency);
                const float sine   = std::sin(frequency);
                const float lo     = row[i];
                const float hi     = row[i + half];
                row[i]             = lo * cosine - hi * sine;
                row[i + half]      = hi * cosine + lo * sine;
            }
        }
    }
}

std::size_t attention_scratch(std::int32_t q_heads, std::int32_t tokens) {
    return static_cast<std::size_t>(q_heads) * static_cast<std::size_t>(tokens);
}

void attention(const float* q, const float* k, const float* v, float* out, std::int32_t q_heads,
               std::int32_t head_dim, std::int32_t tokens, std::int32_t window, float scale,
               float* scratch, ThreadPool& pool) {
    (void)scratch; // the per-query weights fit a thread-local buffer; see below
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
            const float* q_row =
                q + static_cast<std::int64_t>(query) * query_rows + head * head_dim;

            float maximum = -std::numeric_limits<float>::infinity();
            for (std::int32_t key = lo; key < hi; ++key) {
                const float* k_col = k + static_cast<std::int64_t>(key) * head_dim;
                const float dot = ::sinfer::encoder::cpu::dot(q_row, k_col, head_dim);
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
                const float* v_col = v + static_cast<std::int64_t>(key) * head_dim;
                axpy(weight, v_col, target, head_dim);
            }
        }
    });
}

void gelu_mul(const float* gate, const float* up, float* out, std::int64_t count) {
    for (std::int64_t i = 0; i < count; ++i) { out[i] = gelu_tanh(gate[i]) * up[i]; }
}

void add(const float* y, float* x, std::int64_t count) {
    for (std::int64_t i = 0; i < count; ++i) { x[i] += y[i]; }
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
