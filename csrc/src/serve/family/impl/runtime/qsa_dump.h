#pragma once

// QSA parity dumps (a debugging tool, off unless asked for): SUROGATE_SERVE_QSA_DUMP=<dir> with
// SUROGATE_SERVE_QSA_DUMP_LAYER=<l>[,<l>...] writes, for every eagerly executed forward through
// those full-attention layers, the inputs and outputs a reference indexer + attention needs:
// the indexer's input hidden states, the columns' cache and rope positions, the engine's block
// selection (one bit per 4-key block per query row), the attention's query, key and value after
// their norms and rope, and its output before the output gate. One `.npy` per tensor, named
// c<call>_L<layer>_<name>.npy, and one JSON line per call in index.jsonl.
// SUROGATE_SERVE_QSA_DUMP_CALLS=<n> caps the calls written (default 256). CUDA-graph replays
// run no host code, so serve with --enforce-eager to dump decode steps.
// (surogate/serve/tools/parity/qwen4exp/qsa_parity.py reads them.)

#include "api/ops/gqa_attention.h"
#include "core/device.h"
#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <set>
#include <sstream>
#include <string>
#include <vector>

namespace sinfer::family::detail {

struct QsaDumpConfig {
    std::string dir;
    std::set<int> layers;
    int max_calls = 256;
    int calls     = 0;
    std::mutex mutex;
};

inline QsaDumpConfig& qsa_dump_config() {
    static QsaDumpConfig config;
    static const bool initialized = [] {
        QsaDumpConfig& c = config;
        if (const char* dir = std::getenv("SUROGATE_SERVE_QSA_DUMP"); dir != nullptr && *dir) {
            c.dir = dir;
            if (const char* raw = std::getenv("SUROGATE_SERVE_QSA_DUMP_LAYER"); raw != nullptr) {
                std::stringstream list(raw);
                for (std::string item; std::getline(list, item, ',');) {
                    if (!item.empty()) { c.layers.insert(std::stoi(item)); }
                }
            }
            if (const char* raw = std::getenv("SUROGATE_SERVE_QSA_DUMP_CALLS"); raw != nullptr && *raw) {
                c.max_calls = std::atoi(raw);
            }
            std::filesystem::create_directories(c.dir);
        }
        return true;
    }();
    (void)initialized;
    return config;
}

/// The call number to dump this layer's forward under, or -1 (off, another layer, a capture, or
/// the budget spent).
inline int qsa_dump_begin(int layer, cudaStream_t stream) {
    QsaDumpConfig& c = qsa_dump_config();
    if (c.dir.empty() || c.layers.count(layer) == 0) { return -1; }
    cudaStreamCaptureStatus capturing = cudaStreamCaptureStatusNone;
    CUDA_CHECK(cudaStreamIsCapturing(stream, &capturing));
    if (capturing != cudaStreamCaptureStatusNone) { return -1; }
    const std::lock_guard<std::mutex> lock(c.mutex);
    if (c.calls >= c.max_calls) { return -1; }
    return c.calls++;
}

/// One host copy of `bytes` from device memory at `data`, written as a C-order .npy of `shape`.
inline void qsa_dump_raw(int call, int layer, const char* name, const void* data, std::size_t bytes,
                         const char* descr, const std::vector<std::int64_t>& shape, cudaStream_t stream) {
    std::vector<char> host(bytes);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaMemcpy(host.data(), data, bytes, cudaMemcpyDeviceToHost));
    std::ostringstream header;
    header << "{'descr': '" << descr << "', 'fortran_order': False, 'shape': (";
    for (std::size_t i = 0; i < shape.size(); ++i) { header << shape[i] << (shape.size() == 1 ? "," : (i + 1 < shape.size() ? ", " : "")); }
    header << "), }";
    std::string text = header.str();
    const std::size_t total = 10 + text.size() + 1;
    text.append((64 - total % 64) % 64, ' ');
    text.push_back('\n');
    const auto path = std::filesystem::path(qsa_dump_config().dir) /
                      ("c" + std::to_string(call) + "_L" + std::to_string(layer) + "_" + name + ".npy");
    std::ofstream out(path, std::ios::binary);
    const unsigned char magic[8] = {0x93, 'N', 'U', 'M', 'P', 'Y', 1, 0};
    out.write(reinterpret_cast<const char*>(magic), 8);
    const std::uint16_t length = static_cast<std::uint16_t>(text.size());
    out.write(reinterpret_cast<const char*>(&length), 2);
    out.write(text.data(), static_cast<std::streamsize>(text.size()));
    out.write(host.data(), static_cast<std::streamsize>(host.size()));
}

/// A contiguous BF16 / I32 tensor, its dimensions reversed into a C-order shape ([hidden, T] is
/// (T, hidden)). Non-contiguous views are skipped and named in the call's JSON line.
inline bool qsa_dump_tensor(int call, int layer, const char* name, const Tensor& t, cudaStream_t stream) {
    if (t.data == nullptr || !t.is_contiguous()) { return false; }
    const char* descr = t.dtype == DType::BF16 ? "<u2" : t.dtype == DType::I32 ? "<i4" : t.dtype == DType::FP32 ? "<f4" : nullptr;
    if (descr == nullptr) { return false; }
    std::vector<std::int64_t> shape;
    int dims = 4;
    while (dims > 1 && t.ne[dims - 1] == 1) { --dims; }
    for (int d = dims - 1; d >= 0; --d) { shape.push_back(t.ne[d]); }
    qsa_dump_raw(call, layer, name, t.data, t.bytes(), descr, shape, stream);
    return true;
}

inline void qsa_dump_meta(const std::string& json_line) {
    QsaDumpConfig& c = qsa_dump_config();
    const std::lock_guard<std::mutex> lock(c.mutex);
    std::ofstream(std::filesystem::path(c.dir) / "index.jsonl", std::ios::app) << json_line << "\n";
}


/// One attention call's inputs and outputs: `h` [hidden, T], the columns' cache and rope positions,
/// the selection's mask (when sparse), q/k/v after norm and rope, and the output before the gate.
struct QsaDumpCall {
    const char* path = "";
    int layer = 0;
    int columns = 0;
    int sequence_batch = 0;
    int sequence_width = 0;
    std::uint64_t max_visible_keys = 0;
    bool owns_kv = true;
    int head_dim = 0, q_heads = 0, kv_heads = 0;
    float attention_scale = 0.0F;
};

inline void qsa_dump_attention(const QsaDumpCall& m, const Tensor& h, const Tensor& cache_positions,
                               const Tensor& rope, const Tensor& rows, const sinfer::ops::GqaBlockMask& selection,
                               const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& a,
                               cudaStream_t s) {
    const int call = qsa_dump_begin(m.layer, s);
    if (call < 0) { return; }
    std::ostringstream meta;
    meta << "{\"call\": " << call << ", \"path\": \"" << m.path << "\", \"layer\": " << m.layer
         << ", \"columns\": " << m.columns << ", \"sequence_batch\": " << m.sequence_batch
         << ", \"sequence_width\": " << m.sequence_width << ", \"max_visible_keys\": " << m.max_visible_keys
         << ", \"owns_kv\": " << (m.owns_kv ? "true" : "false") << ", \"head_dim\": " << m.head_dim
         << ", \"q_heads\": " << m.q_heads << ", \"kv_heads\": " << m.kv_heads
         << ", \"attention_scale\": " << m.attention_scale << ", \"mask_stride\": " << selection.stride
         << ", \"mask_block\": " << selection.block << ", \"tile_lists\": "
         << (selection.tile_blocks != nullptr ? "true" : "false") << ", \"dumped\": [";
    bool first = true;
    const auto note = [&](const char* name, bool written) {
        if (!written) { return; }
        meta << (first ? "" : ", ") << "\"" << name << "\"";
        first = false;
    };
    note("h", qsa_dump_tensor(call, m.layer, "h", h, s));
    note("cache_positions", qsa_dump_tensor(call, m.layer, "cache_positions", cache_positions, s));
    note("rope_positions", qsa_dump_tensor(call, m.layer, "rope_positions", rope, s));
    note("kv_table_rows", qsa_dump_tensor(call, m.layer, "kv_table_rows", rows, s));
    if (selection.words != nullptr && selection.stride > 0) {
        qsa_dump_raw(call, m.layer, "mask", selection.words,
                     static_cast<std::size_t>(selection.stride) * static_cast<std::size_t>(m.columns) * sizeof(std::uint32_t),
                     "<u4", {m.columns, selection.stride}, s);
        note("mask", true);
    }
    note("q", qsa_dump_tensor(call, m.layer, "q", q, s));
    note("k", qsa_dump_tensor(call, m.layer, "k", k, s));
    note("v", qsa_dump_tensor(call, m.layer, "v", v, s));
    note("attention", qsa_dump_tensor(call, m.layer, "attention", a, s));
    meta << "]}";
    qsa_dump_meta(meta.str());
}

} // namespace sinfer::family::detail
