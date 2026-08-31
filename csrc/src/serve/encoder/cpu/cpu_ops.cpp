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
    // Partition over output rows: each row of w is read once per token block and
    // every thread writes a disjoint stripe of `out`, so there is nothing shared
    // to contend on.
    pool.parallel_for(n, [&](std::int64_t begin, std::int64_t end) {
        for (std::int64_t row = begin; row < end; ++row) {
            const float* w_row = w + row * k;
            for (std::int32_t t = 0; t < tokens; ++t) {
                const float* x_col = x + static_cast<std::int64_t>(t) * k;
#if defined(__AVX512F__)
                __m512 acc = _mm512_setzero_ps();
                std::int32_t i = 0;
                for (; i + kLanes <= k; i += kLanes) {
                    acc = _mm512_fmadd_ps(_mm512_loadu_ps(w_row + i), _mm512_loadu_ps(x_col + i),
                                          acc);
                }
                float sum = _mm512_reduce_add_ps(acc);
                for (; i < k; ++i) { sum += w_row[i] * x_col[i]; }
#else
                float sum = 0.0F;
                for (std::int32_t i = 0; i < k; ++i) { sum += w_row[i] * x_col[i]; }
#endif
                out[static_cast<std::int64_t>(t) * n + row] = sum;
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
                float dot          = 0.0F;
#if defined(__AVX512F__)
                __m512 acc     = _mm512_setzero_ps();
                std::int32_t d = 0;
                for (; d + kLanes <= head_dim; d += kLanes) {
                    acc = _mm512_fmadd_ps(_mm512_loadu_ps(q_row + d), _mm512_loadu_ps(k_col + d),
                                          acc);
                }
                dot = _mm512_reduce_add_ps(acc);
                for (; d < head_dim; ++d) { dot += q_row[d] * k_col[d]; }
#else
                for (std::int32_t d = 0; d < head_dim; ++d) { dot += q_row[d] * k_col[d]; }
#endif
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
#if defined(__AVX512F__)
                const __m512 w = _mm512_set1_ps(weight);
                std::int32_t d = 0;
                for (; d + kLanes <= head_dim; d += kLanes) {
                    _mm512_storeu_ps(target + d, _mm512_fmadd_ps(w, _mm512_loadu_ps(v_col + d),
                                                                _mm512_loadu_ps(target + d)));
                }
                for (; d < head_dim; ++d) { target[d] += weight * v_col[d]; }
#else
                for (std::int32_t d = 0; d < head_dim; ++d) { target[d] += weight * v_col[d]; }
#endif
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
