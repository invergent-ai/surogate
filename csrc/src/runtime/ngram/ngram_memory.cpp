// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0

#include "runtime/ngram/ngram_memory.h"

#include <algorithm>
#include <cerrno>
#include <cstring>
#include <filesystem>
#include <map>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>

#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

#include <cuda_runtime.h>
#include <fmt/format.h>
#include <nlohmann/json.hpp>

#include "kernels/kernels.h"
#include "utilities/utils.h"

namespace ngram_memory {

Spec Spec::parse(const std::string& text) {
    Spec s;
    const nlohmann::json j = nlohmann::json::parse(text);
    s.layer = j.at("layer").get<int>();
    s.hash.ngram = j.at("ngram_size").get<int>();
    s.hash.heads = j.at("heads").get<int>();
    s.hash.eos_token = j.at("eos_token_id").get<int>();
    s.head_dim = j.at("head_dim").get<int>();
    s.row_count = j.at("row_count").get<std::int64_t>();
    const std::string mode = j.value("table_mode", std::string("file"));
    if (mode != "file" && mode != "trainable") {
        throw std::invalid_argument("ngram_memory: table_mode must be 'file' or 'trainable'");
    }
    s.trainable = mode == "trainable";
    if (s.trainable) {
        // No file: an FP32 adapter tensor on the device (W_v random, the table zero).
        s.row_bytes = 0;
        s.table_path.clear();
        s.host_table = false;
    } else {
        s.row_bytes = j.at("row_bytes").get<int>();
        s.table_path = j.at("table").get<std::string>();
        s.table_offset = j.value("table_offset", static_cast<std::int64_t>(0));
        s.host_table = j.value("table_residency", std::string("host")) != "device";
    }
    s.eps = j.value("eps", 1e-6F);
    s.init_std = j.value("init_std", 0.02F);
    s.seed = j.value("seed", static_cast<std::uint64_t>(0));
    s.lr_scale = j.value("lr_scale", 1.0F);
    s.table_lr_scale = j.value("table_lr_scale", s.lr_scale);
    const auto mult = j.at("multipliers").get<std::vector<std::uint64_t>>();
    const auto offsets = j.at("head_offsets").get<std::vector<std::int64_t>>();
    const auto sizes = j.at("head_vocab_sizes").get<std::vector<std::int64_t>>();
    const auto zero = j.value("zero_token_ids", std::vector<int>{});

    if (s.layer < 0) throw std::invalid_argument("ngram_memory: layer must be >= 0");
    if (s.hash.ngram < 2 || s.hash.ngram > kMaxNgram) throw std::invalid_argument("ngram_memory: ngram_size must be 2 or 3");
    if (s.hash.heads <= 0 || s.hash.heads > kMaxHeads || s.hash.heads % (s.hash.ngram - 1) != 0)
        throw std::invalid_argument("ngram_memory: heads must be a multiple of ngram_size - 1, at most 16");
    if (static_cast<int>(mult.size()) < s.hash.ngram) throw std::invalid_argument("ngram_memory: too few multipliers");
    if (static_cast<int>(offsets.size()) != s.hash.heads || static_cast<int>(sizes.size()) != s.hash.heads)
        throw std::invalid_argument("ngram_memory: head_offsets / head_vocab_sizes must have one entry per head");
    if (s.head_dim <= 0 || s.head_dim % 32 != 0 || (!s.trainable && s.row_bytes != s.head_dim / 32 * 18))
        throw std::invalid_argument("ngram_memory: head_dim must be a multiple of 32 and row_bytes head_dim/32*18");
    if (s.row_count <= 0 || s.row_count > (std::int64_t{1} << 31) - 1)
        throw std::invalid_argument("ngram_memory: row_count out of range");
    // The trainable table is one adapter tensor: the optimisers index its elements with int.
    if (s.trainable && s.row_count * static_cast<std::int64_t>(s.head_dim) > (std::int64_t{1} << 31) - 1)
        throw std::invalid_argument("ngram_memory: a trainable table holds at most 2^31 - 1 values");
    if (s.trainable && j.contains("row_shuffle_a"))
        throw std::invalid_argument("ngram_memory: row_shuffle is a control of the file table, not of a trainable one");
    if (!(s.table_lr_scale > 0.0F)) throw std::invalid_argument("ngram_memory: table_lr_scale must be positive");
    for (int i = 0; i < s.hash.ngram; ++i) s.hash.multipliers[i] = mult[static_cast<std::size_t>(i)];
    for (int h = 0; h < s.hash.heads; ++h) {
        const auto off = offsets[static_cast<std::size_t>(h)];
        const auto size = sizes[static_cast<std::size_t>(h)];
        if (off < 0 || size <= 0 || off + size > s.row_count || size > (std::int64_t{1} << 31) - 1)
            throw std::invalid_argument(fmt::format("ngram_memory: head {} range exceeds the table", h));
        s.hash.head_offsets[h] = static_cast<std::int32_t>(off);
        s.hash.head_vocab_sizes[h] = static_cast<std::int32_t>(size);
    }
    if (static_cast<int>(zero.size()) > kMaxZeroTokens) throw std::invalid_argument("ngram_memory: at most 4 zero tokens");
    s.zero_tokens.count = static_cast<int>(zero.size());
    for (int i = 0; i < s.zero_tokens.count; ++i) s.zero_tokens.ids[i] = zero[static_cast<std::size_t>(i)];
    if (j.contains("row_shuffle_a")) {
        const auto a = j.at("row_shuffle_a").get<std::uint64_t>();
        const auto b = j.value("row_shuffle_b", static_cast<std::uint64_t>(0));
        if (a == 0 || (a & 1) == 0 || a >= (std::uint64_t{1} << 32) || b >= (std::uint64_t{1} << 32)) {
            throw std::invalid_argument("ngram_memory: row_shuffle needs an odd a in (0, 2^32) and b in [0, 2^32)");
        }
        s.row_shuffle = RowShuffle{a, b, s.row_count};
    }
    return s;
}

// ------------------------------------------------------------------------------------------------
// Table residency
namespace {

struct HostTable {
    void* host = nullptr;
    std::size_t bytes = 0;
    std::map<int, const std::uint8_t*> device_ptrs;
};
struct DeviceTable {
    void* device = nullptr;
};

std::mutex& registry_mutex() {
    static std::mutex m;
    return m;
}
std::map<std::string, HostTable>& host_tables() {
    static std::map<std::string, HostTable> m;
    return m;
}
std::map<std::pair<std::string, int>, DeviceTable>& device_tables() {
    static std::map<std::pair<std::string, int>, DeviceTable> m;
    return m;
}

std::size_t table_bytes(const Spec& spec) {
    const std::size_t bytes = static_cast<std::size_t>(spec.row_count) * static_cast<std::size_t>(spec.row_bytes);
    std::error_code ec;
    const auto file_size = std::filesystem::file_size(spec.table_path, ec);
    if (ec) throw std::runtime_error(fmt::format("ngram_memory: cannot stat table '{}': {}", spec.table_path, ec.message()));
    if (file_size < static_cast<std::uintmax_t>(spec.table_offset) + bytes)
        throw std::runtime_error(fmt::format("ngram_memory: table '{}' holds {} bytes, the spec needs {} + offset {}",
                                             spec.table_path, file_size, bytes, spec.table_offset));
    return bytes;
}

// Reads [offset, offset + bytes) of the file into dst with several threads (pread).
void read_file_parallel(const std::string& path, std::int64_t offset, std::size_t bytes, std::byte* dst) {
    const int fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0) throw std::runtime_error(fmt::format("ngram_memory: cannot open '{}': {}", path, std::strerror(errno)));
    const unsigned threads = std::max(1U, std::min(16U, std::thread::hardware_concurrency()));
    const std::size_t per = (bytes + threads - 1) / threads;
    std::vector<std::thread> pool;
    std::vector<std::string> errors(threads);
    for (unsigned i = 0; i < threads; ++i) {
        pool.emplace_back([&, i]() {
            std::size_t begin = std::min(bytes, per * i);
            const std::size_t end = std::min(bytes, begin + per);
            while (begin < end) {
                const std::size_t n = std::min<std::size_t>(end - begin, std::size_t{1} << 26);
                const ssize_t got = ::pread(fd, dst + begin, n, offset + static_cast<off_t>(begin));
                if (got <= 0) {
                    errors[i] = got == 0 ? "unexpected end of file" : std::strerror(errno);
                    return;
                }
                begin += static_cast<std::size_t>(got);
            }
        });
    }
    for (auto& t : pool) t.join();
    ::close(fd);
    for (const auto& e : errors)
        if (!e.empty()) throw std::runtime_error(fmt::format("ngram_memory: reading '{}': {}", path, e));
}

}  // namespace

const std::uint8_t* acquire_table(const Spec& spec, int device) {
    std::lock_guard<std::mutex> lock(registry_mutex());
    const std::string key = fmt::format("{}@{}", spec.table_path, spec.table_offset);
    if (!spec.host_table) {
        auto& entry = device_tables()[{key, device}];
        if (!entry.device) {
            const std::size_t bytes = table_bytes(spec);
            CUDA_CHECK(cudaMalloc(&entry.device, bytes));
            constexpr std::size_t kStage = std::size_t{1} << 28;
            std::vector<std::byte> stage(std::min(bytes, kStage));
            for (std::size_t off = 0; off < bytes; off += kStage) {
                const std::size_t n = std::min(kStage, bytes - off);
                read_file_parallel(spec.table_path, spec.table_offset + static_cast<std::int64_t>(off), n, stage.data());
                CUDA_CHECK(cudaMemcpy(static_cast<std::byte*>(entry.device) + off, stage.data(), n, cudaMemcpyHostToDevice));
            }
            fprintf(stderr, "[ngram_memory] table %s: %.2f GB copied to device %d\n", spec.table_path.c_str(),
                    bytes / 1e9, device);
        }
        return static_cast<const std::uint8_t*>(entry.device);
    }
    auto& entry = host_tables()[key];
    if (!entry.host) {
        const std::size_t bytes = table_bytes(spec);
        void* host = ::mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (host == MAP_FAILED)
            throw std::runtime_error(fmt::format("ngram_memory: cannot map {} bytes of host memory", bytes));
        read_file_parallel(spec.table_path, spec.table_offset, bytes, static_cast<std::byte*>(host));
        CUDA_CHECK(cudaHostRegister(host, bytes, cudaHostRegisterMapped | cudaHostRegisterPortable));
        entry.host = host;
        entry.bytes = bytes;
        fprintf(stderr, "[ngram_memory] table %s: %.2f GB pinned in host memory (mapped)\n", spec.table_path.c_str(),
                bytes / 1e9);
    }
    auto it = entry.device_ptrs.find(device);
    if (it == entry.device_ptrs.end()) {
        void* dptr = nullptr;
        CUDA_CHECK(cudaHostGetDevicePointer(&dptr, entry.host, 0));
        it = entry.device_ptrs.emplace(device, static_cast<const std::uint8_t*>(dptr)).first;
    }
    return it->second;
}

// ------------------------------------------------------------------------------------------------
// Forward / backward
namespace {

void require(bool ok, const char* what) {
    if (!ok) throw std::invalid_argument(std::string("ngram_memory: ") + what);
}

}  // namespace

void forward_core(Tensor& k, Tensor& v, Tensor& stats, Tensor& u, float* global_stats, const Tensor& r,
                  const Tensor& emb, const Tensor& Wk, const Tensor& Wv, const float* gk, const float* gq,
                  const int* ids, const ZeroTokens& zero, float eps, cublasLtHandle_t handle, Tensor& workspace,
                  cudaStream_t stream) {
    const int N = static_cast<int>(emb.Sizes[0]);
    const int E = static_cast<int>(emb.Sizes[1]);
    const int C = static_cast<int>(Wk.Sizes[0]);
    require(Wk.Sizes[1] == E && Wv.Sizes[0] == C && Wv.Sizes[1] == E, "W_k / W_v must be [C, E]");
    require(Wk.DType == emb.DType && Wv.DType == emb.DType, "W_k / W_v must have the dtype of the embedding");
    require(k.DType == r.DType && v.DType == r.DType && u.DType == r.DType, "k, v, u must have the dtype of r");
    require(static_cast<long>(r.nelem()) == static_cast<long>(N) * C, "r must be [N, C]");
    // k = e W_k^T, v = e W_v^T
    matmul(k, Wk, emb, std::nullopt, nullptr, nullptr, handle, workspace, C, N, E, EMMTranspose::TN, false, stream);
    matmul(v, Wv, emb, std::nullopt, nullptr, nullptr, handle, workspace, C, N, E, EMMTranspose::TN, false, stream);
    if (r.DType == ETensorDType::BF16) {
        gate_forward<nv_bfloat16>(u.get<nv_bfloat16>(), stats.get<float>(), global_stats, r.get<nv_bfloat16>(),
                                  k.get<nv_bfloat16>(), v.get<nv_bfloat16>(), gk, gq, ids, zero, N, C, eps, stream);
    } else if (r.DType == ETensorDType::FP32) {
        gate_forward<float>(u.get<float>(), stats.get<float>(), global_stats, r.get<float>(), k.get<float>(),
                            v.get<float>(), gk, gq, ids, zero, N, C, eps, stream);
    } else {
        throw std::invalid_argument("ngram_memory: r must be BF16 or FP32");
    }
}

std::size_t table_backward_scratch_bytes(long entries) { return encoder_backward_scratch_bytes(entries); }

void table_backward(Tensor& dtable, Tensor& d_emb, Tensor& scratch, const Tensor& dk, const Tensor& dv,
                    const Tensor& Wk, const Tensor& Wv, const int* rows, int heads, int head_dim,
                    cublasLtHandle_t handle, Tensor& workspace, cudaStream_t stream) {
    const int N = static_cast<int>(dk.Sizes[0]);
    const int C = static_cast<int>(Wk.Sizes[0]);
    const int E = static_cast<int>(Wk.Sizes[1]);
    require(E == heads * head_dim, "W_k must be [C, heads * head_dim]");
    require(Wv.Sizes[0] == C && Wv.Sizes[1] == E, "W_v must be [C, E]");
    require(dk.DType == Wk.DType && dv.DType == Wk.DType && Wv.DType == Wk.DType,
            "dk, dv, W_k and W_v must share a dtype");
    require(static_cast<long>(dk.nelem()) == static_cast<long>(N) * C && dv.nelem() == dk.nelem(),
            "dk / dv must be [N, C]");
    require(d_emb.DType == ETensorDType::FP32 && static_cast<long>(d_emb.nelem()) >= static_cast<long>(N) * E,
            "d_emb must be FP32 [N, E]");
    require(dtable.DType == ETensorDType::FP32 && dtable.Rank == 2 && dtable.Sizes[1] == head_dim,
            "the table gradient must be FP32 [rows, head_dim]");
    if (N <= 0) return;
    // d_emb = dk W_k + dv W_v   (row-major [N, E]: column-major d_emb^T [E, N] = W^T [E, C] d^T [C, N])
    Tensor de = d_emb;
    de.Rank = 2;
    de.Sizes[0] = N;
    de.Sizes[1] = E;
    matmul(de, Wk, dk, std::nullopt, nullptr, nullptr, handle, workspace, E, N, C, EMMTranspose::NN, false, stream);
    matmul(de, Wv, dv, std::nullopt, nullptr, nullptr, handle, workspace, E, N, C, EMMTranspose::NN, true, stream);
    // dtable[row] += d_emb row slices, in row order: the N * heads (row, slice) pairs of the gather.
    const long entries = static_cast<long>(N) * heads;
    const std::size_t need = encoder_backward_scratch_bytes(entries);
    require(scratch.bytes() >= need, "table scratch too small");
    encoder_backward(dtable.get<float>(), reinterpret_cast<std::byte*>(scratch.Data), scratch.bytes(),
                     de.get<float>(), rows, 1, static_cast<int>(entries), head_dim, 0U, stream);
}

void backward_core(Tensor& d_res, Tensor& dWk, Tensor& dWv, Tensor& dgk, Tensor& dgq, Tensor& dk, Tensor& dv,
                   Tensor& w, Tensor& partials, float* abs_max, const Tensor& r, const Tensor& emb, const Tensor& k,
                   const Tensor& v, const Tensor& stats, const float* gk, const float* gq, const int* ids,
                   const ZeroTokens& zero, cublasLtHandle_t handle, Tensor& workspace, cudaStream_t stream) {
    const int N = static_cast<int>(emb.Sizes[0]);
    const int E = static_cast<int>(emb.Sizes[1]);
    const int C = static_cast<int>(dWk.Sizes[0]);
    require(dWk.Sizes[1] == E && dWv.Sizes[0] == C && dWv.Sizes[1] == E, "dW_k / dW_v must be [C, E]");
    require(d_res.DType == r.DType && dk.DType == r.DType && dv.DType == r.DType, "d_res, dk, dv must have the dtype of r");
    require(static_cast<long>(partials.nelem()) >= static_cast<long>(gain_grad_parts(N)) * C, "partials too small");
    const bool bf16 = r.DType == ETensorDType::BF16;
    if (bf16) {
        gate_backward<nv_bfloat16>(dk.get<nv_bfloat16>(), dv.get<nv_bfloat16>(), w.get<float>(),
                                   d_res.get<nv_bfloat16>(), abs_max, r.get<nv_bfloat16>(), k.get<nv_bfloat16>(),
                                   v.get<nv_bfloat16>(), stats.get<float>(), gk, gq, ids, zero, N, C, stream);
    } else {
        gate_backward<float>(dk.get<float>(), dv.get<float>(), w.get<float>(), d_res.get<float>(), abs_max,
                             r.get<float>(), k.get<float>(), v.get<float>(), stats.get<float>(), gk, gq, ids, zero, N,
                             C, stream);
    }
    // The GEMM operands must share a dtype: the embedding's. In the bf16 path dk/dv are bf16 already.
    require(dk.DType == emb.DType, "dk / dv must have the dtype of the embedding");
    // dW = d^T e  ([C, E]), accumulated
    matmul(dWv, emb, dv, std::nullopt, nullptr, nullptr, handle, workspace, E, C, N, EMMTranspose::NT, true, stream);
    matmul(dWk, emb, dk, std::nullopt, nullptr, nullptr, handle, workspace, E, C, N, EMMTranspose::NT, true, stream);
    const bool g_bf16 = dgk.DType == ETensorDType::BF16;
    require(dgq.DType == dgk.DType, "dgk / dgq dtypes differ");
    if (bf16 && g_bf16) {
        gain_grads<nv_bfloat16, nv_bfloat16>(dgk.get<nv_bfloat16>(), dgq.get<nv_bfloat16>(), partials.get<float>(),
                                             w.get<float>(), r.get<nv_bfloat16>(), k.get<nv_bfloat16>(), gk, gq, N, C,
                                             stream);
    } else if (bf16) {
        gain_grads<nv_bfloat16, float>(dgk.get<float>(), dgq.get<float>(), partials.get<float>(), w.get<float>(),
                                       r.get<nv_bfloat16>(), k.get<nv_bfloat16>(), gk, gq, N, C, stream);
    } else if (g_bf16) {
        gain_grads<float, nv_bfloat16>(dgk.get<nv_bfloat16>(), dgq.get<nv_bfloat16>(), partials.get<float>(),
                                       w.get<float>(), r.get<float>(), k.get<float>(), gk, gq, N, C, stream);
    } else {
        gain_grads<float, float>(dgk.get<float>(), dgq.get<float>(), partials.get<float>(), w.get<float>(),
                                 r.get<float>(), k.get<float>(), gk, gq, N, C, stream);
    }
}

}  // namespace ngram_memory
