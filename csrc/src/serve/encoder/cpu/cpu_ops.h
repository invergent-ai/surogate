#pragma once

// The encoder's arithmetic, on host cores.
//
// A small, closed set: an encoder is one forward, so there is no KV cache, no
// sampler and no decode round to serve. What is left is eleven kernels, and
// they are here rather than behind the engine's op contract because that
// contract takes a `cudaStream_t` in 102 of its 104 entry points.
//
// Weights arrive dequantised to FP32. The artifact stores W8G32_F16S -- int8
// codes with one binary16 scale per 32 values -- and decoding it once at load
// costs 1.2 GB of the host's 504 and buys a dense GEMM instead of a quantised
// one. That is the right trade here and would not be on a 100 GB model.
//
// Everything is FP32. The GPU path is BF16 because its tensor cores are, and
// the reference is FP32; on CPU the cheapest thing is also the most accurate.

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <span>
#include <vector>

namespace sinfer::encoder::cpu {

/// How many worker threads to use, and which CPUs they may run on.
///
/// Sized on *physical cores of one NUMA node*, not on
/// `hardware_concurrency()`. Measured on this host (2x EPYC 9124, 32 physical /
/// 64 logical / 2 nodes) with llama.cpp on the same model at 512 tokens:
///
///     64 threads, SMT, both sockets      390.7 ms
///     32 threads, physical, both sockets 209.6 ms   1.86x
///     16 threads, physical, one node     181.9 ms   2.15x
///
/// Two effects, and both are the same cause: every matmul ends in a barrier, so
/// the slowest thread sets the pace. An SMT sibling contends for the same
/// execution ports, and a thread on the far socket waits on interconnect. A
/// 1.2 GB model is comfortably node-local, so there is nothing to gain by
/// spanning nodes -- unlike a large interleaved expert bank, where spanning
/// them is what buys the bandwidth.
struct ThreadPlan {
    int threads = 0;             ///< 0 = choose from the topology.
    std::vector<int> cpus;       ///< empty = do not pin.

    /// Physical cores of the node this process is already bound to, or of node
    /// 0 when it is unbound.
    static ThreadPlan detect();
};

class ThreadPool {
public:
    explicit ThreadPool(ThreadPlan plan);
    ~ThreadPool();
    ThreadPool(const ThreadPool&)            = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;

    /// Runs `body(begin, end)` over a partition of [0, count) and returns once
    /// every part is done.
    void parallel_for(std::int64_t count, const std::function<void(std::int64_t, std::int64_t)>& body);

    [[nodiscard]] int threads() const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// --- the kernels ------------------------------------------------------------
//
// Activations are [rows, tokens] with rows contiguous, matching the GPU path,
// so the two implementations can be read side by side.

/// out[n, T] = w[n, k] . x[k, T]
void gemm(const float* w, const float* x, float* out, std::int32_t n, std::int32_t k,
          std::int32_t tokens, ThreadPool& pool);

/// out[d, T] = table[ids[t], d]
void embed(const float* table, const std::int32_t* ids, float* out, std::int32_t hidden,
           std::int32_t tokens, std::int32_t vocab);

/// RMSNorm over the fastest axis. `unit_offset` applies gain = 1 + weight,
/// which is Gemma's zero-centred convention.
void rmsnorm(const float* x, const float* weight, float epsilon, bool unit_offset, float* out,
             std::int32_t rows, std::int32_t tokens);

/// Rotates [0, head_dim) of every head in place, positions[t] per column.
void rope(float* x, const std::int32_t* positions, std::int32_t head_dim, std::int32_t heads,
          std::int32_t tokens, float theta);

/// Non-causal GQA over one sequence. `window` 0 admits every key; positive W
/// admits abs(i - j) < W, symmetric -- what a window means once attention is
/// bidirectional.
void attention(const float* q, const float* k, const float* v, float* out, std::int32_t q_heads,
               std::int32_t head_dim, std::int32_t tokens, std::int32_t window, float scale,
               float* scratch, ThreadPool& pool);

/// out = gelu_tanh(gate) * up, elementwise.
void gelu_mul(const float* gate, const float* up, float* out, std::int64_t count);

/// x += y, elementwise.
void add(const float* y, float* x, std::int64_t count);

/// x *= factor, elementwise.
void scale(float* x, float factor, std::int64_t count);

/// out[h] = mean over the first `count` columns of x[h, .]
void mean_pool(const float* x, float* out, std::int32_t hidden, std::int32_t count);

/// Divides each column by its own L2 norm.
void l2norm(float* x, std::int32_t rows, std::int32_t columns, float epsilon);

/// Scratch floats `attention` needs for one sequence.
[[nodiscard]] std::size_t attention_scratch(std::int32_t q_heads, std::int32_t tokens);

} // namespace sinfer::encoder::cpu
