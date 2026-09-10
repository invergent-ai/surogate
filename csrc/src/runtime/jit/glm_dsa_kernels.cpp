// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
#include "runtime/jit/glm_dsa_kernels.h"
#include "kernels/decode.h"

#include "utilities/utils.h"

namespace {
constexpr std::array Names = {"dsa_norm",
                              "dsa_pool",
                              "dsa_score",
                              "dsa_select",
                              "dsa_indices",
                              "dsa_attn_fwd",
                              "dsa_attn_decode",
                              "dsa_attn_bwd",
                              "dsa_gather_latents",
                              "dsa_repack_kv",
                              "dsa_pool_paged",
                              "dsa_score_paged",
                              "dsa_select_paged",
                              "dsa_pool_batch",
                              "dsa_score_batch",
                              "dsa_select_batch",
                              "dsa_indices_batch"};
struct Workspace {
    std::size_t bytes = 0;
    void *keys, *pooled, *ends, *scores, *selected;
    Workspace(std::byte* data, int B, int T, int TK, int D, int P, int select) {
        auto alloc = [&](std::size_t count, int size) -> void* {
            auto offset = bytes;
            bytes += (count * size + 255) / 256 * 256;
            return data ? data + offset : nullptr;
        };
        const long rows = static_cast<long>(B) * T, pools = (TK + P - 1) / P;
        const long tile_rows = static_cast<long>(B) * std::min(T, GlmDsaKernels::IndexerQueryTile);
        keys = alloc(rows * D, 2);
        pooled = alloc(B * pools * D, 2);
        ends = alloc(B * pools, 4);
        scores = alloc(tile_rows * pools, 4);
        selected = alloc(tile_rows * select, 4);
    }
};
void require(bool ok, const char* message) {
    if (!ok) throw std::runtime_error(message);
}
}  // namespace

void GlmDsaKernels::load(const std::unordered_map<std::string, std::string>& manifests) {
    for (auto* name : Names)
        if (auto it = manifests.find(name); it != manifests.end())
            mKernels.insert_or_assign(name, JitKernel::load_manifest(it->second));
    require(mKernels.empty() || is_ready(), "Incomplete GLM DSA kernel manifests");
}

int GlmDsaKernels::constant(const char* kernel, const char* name) const {
    require(is_ready(), "GLM DSA requires compile_jit_kernels() and JitKernelManifests");
    return mKernels.at(kernel).meta().const_int(name, -1);
}

std::size_t GlmDsaKernels::indexer_workspace_bytes(int B, int T, int TK, int D, int P, int select) {
    return Workspace(nullptr, B, T, TK, D, P, select).bytes;
}

std::size_t GlmDsaKernels::workspace_bytes(int B, int T, int length) const {
    require(B > 0 && T > 0 && length >= 0, "Invalid DSA dimensions");
    return Workspace(nullptr,
                     B,
                     T,
                     T + length,
                     constant("dsa_pool", "D"),
                     constant("dsa_pool", "P"),
                     constant("dsa_select", "SELECT"))
        .bytes;
}

void GlmDsaKernels::indexer(const std::vector<Tensor>& x,
                            const Tensor& indices,
                            const Tensor& workspace,
                            cudaStream_t stream,
                            dsl::GlmDecodeState* cache,
                            int layer,
                            const dsl::ExecutionRequest* request) const {
    require(x.size() == 8 && x[0].Rank == 4, "Invalid DSA indexer inputs");
    int B = x[0].Sizes[0], T = x[0].Sizes[1], H = constant("dsa_score", "H"), D = constant("dsa_pool", "D");
    int P = constant("dsa_pool", "P"), select = constant("dsa_select", "SELECT");
    int offset = cache ? cache->length : 0, TK = T + offset, NP = (TK + P - 1) / P;
    if (request) {
        for (int row = 0; row < B; ++row)
            offset = std::max(offset, request->decode_state(row)->length);
        TK = T + offset;
        NP = (TK + P - 1) / P;
    }
    int capacity = cache ? cache->capacity : T, pcap = cache ? (capacity + P - 1) / P : NP;
    require(x[0].Sizes[2] == H && x[0].Sizes[3] == D && NP <= constant("dsa_select", "BLOCK"),
            "DSA geometry exceeds compiled manifests");
    for (int i = 0; i < 7; ++i)
        require(x[i].Data && x[i].DType == ETensorDType::BF16, "DSA indexer expects BF16 inputs");
    require(x[1].nelem() == static_cast<long>(B) * T * D && x[2].nelem() == x[1].nelem() &&
                x[3].nelem() == static_cast<long>(B) * T * H && x[4].nelem() == D && x[5].nelem() == D &&
                x[6].nelem() == P * D && x[7].DType == ETensorDType::INT32 && x[7].nelem() == B * T,
            "Invalid DSA projection or position shapes");
    require(indices.DType == ETensorDType::INT32 &&
                indices.nelem() == static_cast<long>(B) * T * constant("dsa_indices", "STRIDE"),
            "Invalid DSA indices output");
    require(!cache || (B == 1 && TK <= capacity), "GLM decode requires one sequence within cache capacity");
    Workspace s(workspace.Data, B, T, TK, D, P, select);
    require(workspace.Data && workspace.bytes() >= s.bytes, "DSA workspace is too small");
    launch("dsa_norm", dim3(B * T), stream, x[1].Data, x[4].Data, x[5].Data, s.keys, T);
    void *keys = s.keys, *gate = x[2].Data, *pos = x[7].Data, *pooled = s.pooled, *ends = s.ends;
    int start = 0;
    const bool paged = cache && cache->limit > 0;
    if (request) {
        const auto& kb = request->decode_binding(layer, "index_keys");
        auto normalized = Tensor::from_pointer(static_cast<std::byte*>(keys),
                                               x[1].Device,
                                               ETensorDType::BF16,
                                               std::vector<long>{B, T, D});
        decode_append_pages_batch(normalized, kb, B, T, D, stream);
        decode_append_pages_batch(x[2], request->decode_binding(layer, "index_gate"), B, T, D, stream);
        decode_append_pages_batch(x[7], request->decode_binding(layer, "index_positions"), B, T, 1, stream);
        keys = kb.Data;
        gate = request->decode_binding(layer, "index_gate").Data;
        pos = request->decode_binding(layer, "index_positions").Data;
        pooled = request->decode_binding(layer, "index_pooled").Data;
        ends = request->decode_binding(layer, "index_ends").Data;
    } else if (paged) {
        auto& k = cache->pages(layer, "index_keys", ETensorDType::BF16, D);
        auto& g = cache->pages(layer, "index_gate", ETensorDType::BF16, D);
        auto& p = cache->pages(layer, "index_positions", ETensorDType::INT32, 1);
        auto& pk = cache->pages(layer, "index_pooled", ETensorDType::BF16, D);
        auto& pe = cache->pages(layer, "index_ends", ETensorDType::INT32, 1);
        k.append(keys, offset, T, stream);
        g.append(gate, offset, T, stream);
        p.append(pos, offset, T, stream);
        pk.reserve(NP, stream);
        pe.reserve(NP, stream);
        keys = k.table().Data;
        gate = g.table().Data;
        pos = p.table().Data;
        pooled = pk.table().Data;
        ends = pe.table().Data;
        start = offset / P;
    } else if (cache) {
        auto k = cache->get(layer, "index_keys", ETensorDType::BF16, {B, capacity, D});
        auto g = cache->get(layer, "index_gate", ETensorDType::BF16, {B, capacity, D});
        auto p = cache->get(layer, "index_positions", ETensorDType::INT32, {B, capacity});
        auto pk = cache->get(layer, "index_pooled", ETensorDType::BF16, {B, pcap, D});
        auto pe = cache->get(layer, "index_ends", ETensorDType::INT32, {B, pcap});
        CUDA_CHECK(cudaMemcpyAsync(k.Data + static_cast<long>(offset) * D * 2,
                                   keys,
                                   static_cast<long>(T) * D * 2,
                                   cudaMemcpyDeviceToDevice,
                                   stream));
        CUDA_CHECK(cudaMemcpyAsync(g.Data + static_cast<long>(offset) * D * 2,
                                   gate,
                                   static_cast<long>(T) * D * 2,
                                   cudaMemcpyDeviceToDevice,
                                   stream));
        CUDA_CHECK(cudaMemcpyAsync(p.Data + static_cast<long>(offset) * 4,
                                   pos,
                                   static_cast<long>(T) * 4,
                                   cudaMemcpyDeviceToDevice,
                                   stream));
        keys = k.Data;
        gate = g.Data;
        pos = p.Data;
        pooled = pk.Data;
        ends = pe.Data;
        start = offset / P;
    }
    launch(request ? "dsa_pool_batch"
           : paged ? "dsa_pool_paged"
                   : "dsa_pool",
           dim3(request ? (T + 2 * P - 2) / P : NP - start, B),
           stream,
           keys,
           gate,
           x[6].Data,
           pos,
           pooled,
           ends,
           request ? T : TK,
           capacity,
           pcap,
           start);
    // Score, select and expand one query tile before reusing its workspace.
    // Keep the score arithmetic and deterministic tie order unchanged.
    // Keep the unfinished pool after all top-k slots, including empty ones.
    // Compacting it for short prefixes changes the attention reduction tree
    // relative to a packed training batch, which can change BF16 rounding.
    const int selected = select;
    for (int query_start = 0; query_start < T; query_start += IndexerQueryTile) {
        const int query_count = std::min(IndexerQueryTile, T - query_start);
        launch(request ? "dsa_score_batch"
               : paged ? "dsa_score_paged"
                       : "dsa_score",
               dim3(query_count, (NP + 31) / 32, B),
               stream,
               x[0].Data,
               x[3].Data,
               pooled,
               ends,
               x[7].Data,
               s.scores,
               T,
               TK,
               NP,
               pcap,
               offset,
               query_start,
               query_count);
        launch(request ? "dsa_select_batch"
               : paged ? "dsa_select_paged"
                       : "dsa_select",
               dim3(query_count, B),
               stream,
               s.scores,
               ends,
               s.selected,
               query_count,
               NP,
               pcap);
        if (request)
            launch("dsa_indices_batch",
                   dim3(query_count, B),
                   stream,
                   s.selected,
                   x[7].Data,
                   indices.Data,
                   T,
                   keys,
                   selected,
                   query_start,
                   query_count);
        else
            launch("dsa_indices",
                   dim3(query_count, B),
                   stream,
                   s.selected,
                   x[7].Data,
                   indices.Data,
                   T,
                   offset,
                   selected,
                   query_start,
                   query_count);
    }
}

int GlmDsaKernels::selection_slots() const {
    return constant("dsa_indices", "STRIDE");
}

void GlmDsaKernels::gather_latents(const Tensor& latent,
                                   const Tensor& indices,
                                   const Tensor& out,
                                   const Tensor& selected,
                                   int slots,
                                   cudaStream_t stream) const {
    const int rank = out.Sizes[1];
    launch("dsa_gather_latents",
           dim3(slots, (rank + 255) / 256),
           stream,
           latent.Data,
           indices.Data,
           out.Data,
           selected.Data,
           rank,
           slots);
}

void GlmDsaKernels::repack_kv(const Tensor& projected, const Tensor& out, cudaStream_t stream) const {
    launch("dsa_repack_kv",
           dim3((projected.nelem() + 255) / 256),
           stream,
           projected.Data,
           out.Data,
           static_cast<int>(projected.Sizes[0]));
}

void GlmDsaKernels::attention_selected(const Tensor& qkv,
                                       const Tensor& indices,
                                       const Tensor& kv,
                                       const Tensor& out,
                                       const Tensor& lse,
                                       int slots,
                                       cudaStream_t stream) const {
    launch("dsa_attn_decode",
           dim3(1, constant("dsa_attn_fwd", "H"), qkv.Sizes[0]),
           stream,
           qkv.Data,
           kv.Data,
           indices.Data,
           out.Data,
           lse.Data,
           1,
           slots,
           slots);
}

void GlmDsaKernels::attention(const Tensor& qkv,
                              const Tensor& indices,
                              const Tensor& out,
                              const Tensor& lse,
                              cudaStream_t stream,
                              dsl::GlmDecodeState* cache,
                              int layer) const {
    require(qkv.Rank == 4 && qkv.DType == ETensorDType::BF16, "DSA attention requires BF16 [B,T,3H,D]");
    int B = qkv.Sizes[0], T = qkv.Sizes[1], H = constant("dsa_attn_fwd", "H"), D = constant("dsa_attn_fwd", "D");
    int TK = T + (cache ? cache->length : 0), capacity = cache ? cache->capacity : T;
    int S = selection_slots();
    require(qkv.Sizes[2] == 3 * H && qkv.Sizes[3] == D && out.DType == ETensorDType::BF16 &&
                out.nelem() == static_cast<long>(B) * T * H * D && lse.DType == ETensorDType::FP32 &&
                lse.nelem() == static_cast<long>(B) * T * H && indices.DType == ETensorDType::INT32 &&
                indices.nelem() == static_cast<long>(B) * T * constant("dsa_indices", "STRIDE"),
            "Invalid DSA attention tensors");
    void* kv = nullptr;
    if (cache) {
        require(B == 1 && TK <= capacity, "Invalid DSA decode dimensions");
        auto history = cache->get(layer, "mla_kv", ETensorDType::BF16, {B, capacity, 2 * H, D});
        CUDA_CHECK(cudaMemcpy2DAsync(history.Data + static_cast<long>(cache->length) * 2 * H * D * 2,
                                     2L * H * D * 2,
                                     qkv.Data + H * D * 2,
                                     3L * H * D * 2,
                                     2L * H * D * 2,
                                     T,
                                     cudaMemcpyDeviceToDevice,
                                     stream));
        kv = history.Data;
    }
    launch(cache ? "dsa_attn_decode" : "dsa_attn_fwd",
           dim3(T, H, B),
           stream,
           qkv.Data,
           kv,
           indices.Data,
           out.Data,
           lse.Data,
           T,
           capacity,
           S);
}

void GlmDsaKernels::backward(const Tensor& dout,
                             const Tensor& qkv,
                             const Tensor& indices,
                             const Tensor& out,
                             const Tensor& lse,
                             const Tensor& dqkv,
                             cudaStream_t stream) const {
    int B = qkv.Sizes[0], T = qkv.Sizes[1], H = constant("dsa_attn_fwd", "H");
    int S = selection_slots();
    require(dout.DType == ETensorDType::FP32 && dqkv.DType == ETensorDType::FP32 && dqkv.nelem() == qkv.nelem(),
            "DSA backward requires FP32 gradients");
    CUDA_CHECK(cudaMemsetAsync(dqkv.Data, 0, dqkv.bytes(), stream));
    launch("dsa_attn_bwd",
           dim3(T, H, B),
           stream,
           dout.Data,
           qkv.Data,
           indices.Data,
           out.Data,
           lse.Data,
           dqkv.Data,
           T,
           S);
}
