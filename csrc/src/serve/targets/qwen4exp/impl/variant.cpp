#include "family/impl/lora_gdn.h"
#include "targets/qwen4exp/impl/variant.h"

#include "family/impl/lora_hook.h"
#include "family/impl/moe/expert_cache.h"
#include "api/ops/causal_conv1d_silu.h"
#include "api/ops/embedding.h"
#include "api/ops/gdn_gating.h"
#include "api/ops/hyper_connection.h"
#include "ops/gdn_input_proj/gdn_projected_conv.h"
#include "api/ops/linear.h"
#include "api/ops/ngram_ple.h"
#include "api/ops/scatter.h"
#include "api/ops/sparse_moe.h"
#include "core/device.h"
#include "core/engine_context.h"
#include "ops/linear/bf16/bf16_cublaslt.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <memory>
#include <string>
#include <stdexcept>
#include <unordered_map>
#include <vector>

#define SINFER_FAMILY_VARIANT    ::sinfer::targets::qwen4exp::detail::Variant
#define SINFER_FAMILY_RUNTIME_NS qwen4exp_runtime
#include "family/impl/runtime/instantiate.h"

namespace sinfer::targets::qwen4exp::detail {
ops::SparseMoeGeometry moe_geometry(const family::TextGeometry& g) {
    return {.hidden = g.hidden, .experts = g.experts, .experts_per_token = g.experts_per_token,
            .intermediate = g.intermediate, .gating = ops::SparseMoeGating::SoftmaxTopK,
            .routed_scale = g.routed_scale, .shared_gated = true,
            .shared_intermediate = g.shared_intermediate};
}

namespace {

constexpr auto kPolicy           = ops::LinearPolicy::A16Only;

// The mix hook of a block computes the inject gates its output projection scatters with. The
// family plans the hooks' scratch as transient, so the gates cannot live in the arena between
// the two calls; they use a small device buffer created before any graph capture, and travel
// through a thread-local handle rather than a family-visible parameter.
constexpr std::size_t kInjectScratchTokens = 65536;
thread_local Tensor t_inject;
thread_local family::TextGeometry t_inject_geometry;
// The inject gates are handed from the mixer to the combine through this thread (the host
// partial through the device's expert cache), while the buffers they name belong to a device. One thread drives every stage of a
// pipeline, so a hand-off that crosses a stage boundary would add one device's expert output
// into another's residual. SUROGATE_SERVE_CPU_MOE_VERIFY=1 checks that it never does.
thread_local int t_inject_device = -1;

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
                     "qwen4exp: %s was produced on device %d and is being consumed on device "
                     "%d\n",
                     what, produced_on, current);
    }
}

struct InjectScratch {
    std::unique_ptr<DeviceArena> storage;
    void* data        = nullptr;
    std::size_t bytes = 0;
};

struct InjectRegistry {
    std::mutex mutex;
    std::unordered_map<std::uint64_t, InjectScratch> buffers;
};

InjectScratch& inject_scratch_for_current_device(std::int32_t streams) {
    auto& registry = ops::engine_slot<InjectRegistry>();
    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    const std::lock_guard<std::mutex> lock(registry.mutex);
    return registry.buffers[(std::uint64_t(device) << 32U) | std::uint32_t(streams)];
}

std::vector<GraphExecutionProfile>
graph_profiles_through(std::uint32_t max_frontier, const std::vector<std::uint32_t>& ends) {
    std::vector<GraphExecutionProfile> out;
    std::uint32_t begin = 0;
    for (const std::uint32_t preferred_end : ends) {
        if (begin > max_frontier) { break; }
        const std::uint32_t end = std::min(preferred_end, max_frontier);
        out.push_back({begin, end});
        if (end == max_frontier) { return out; }
        begin = end + 1;
    }
    if (begin <= max_frontier) { out.push_back({begin, max_frontier}); }
    return out;
}

std::size_t round_up(std::size_t bytes) { return (bytes + 255) / 256 * 256; }

// Debug parity dumps: SUROGATE_SERVE_DUMP_RESIDUAL=<dir> writes the residual streams before
// every layer and the final mixed hidden state of the first forwards with at most 64 columns,
// as raw BF16 with a 16-byte header {magic, rows, columns, forward}. Synchronises the stream.
struct ResidualDump {
    std::string dir;
    int forward = 0;
    int limit   = 0;
};

ResidualDump& residual_dump() {
    static ResidualDump dump = [] {
        ResidualDump out;
        if (const char* raw = std::getenv("SUROGATE_SERVE_DUMP_RESIDUAL"); raw != nullptr && *raw) {
            out.dir   = raw;
            out.limit = 2;
        }
        return out;
    }();
    return dump;
}

void dump_tensor(const Tensor& tensor, const std::string& name, int forward, cudaStream_t stream) {
    const std::size_t bytes = tensor.bytes();
    std::vector<std::byte> host(bytes);
    CUDA_CHECK(cudaMemcpyAsync(host.data(), tensor.data, bytes, cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    const std::string path = residual_dump().dir + "/" + name + ".bin";
    FILE* file             = std::fopen(path.c_str(), "wb");
    if (file == nullptr) { return; }
    const std::int32_t header[4] = {0x52455344, tensor.ne[0], tensor.ne[1], forward};
    std::fwrite(header, sizeof(header), 1, file);
    std::fwrite(host.data(), 1, bytes, file);
    std::fclose(file);
}

// Layer 0's prologue marks the start of a forward; the layer index is kept for the
// per-block intermediate dumps of the first two layers.
int g_dump_layer = -1;
int g_dump_block = 0; // mixes seen in the current layer: 0 = mixer block, 1 = MLP block

void maybe_dump_layer(int layer, const Tensor& residual, cudaStream_t stream) {
    ResidualDump& dump = residual_dump();
    if (dump.dir.empty() || residual.ne[1] > 64) { return; }
    if (layer == 0) { dump.forward += 1; }
    g_dump_layer = layer;
    g_dump_block = 0;
    if (dump.forward > dump.limit) { return; }
    dump_tensor(residual, "f" + std::to_string(dump.forward) + "_layer" + std::to_string(layer),
                dump.forward, stream);
}

void maybe_dump_block(const char* tag, const Tensor& tensor, cudaStream_t stream) {
    ResidualDump& dump = residual_dump();
    if (dump.dir.empty() || tensor.ne[1] > 64 || dump.forward > dump.limit || g_dump_layer < 0 ||
        g_dump_layer > 1) {
        return;
    }
    dump_tensor(tensor,
                "f" + std::to_string(dump.forward) + "_L" + std::to_string(g_dump_layer) +
                    (g_dump_block == 0 ? "_mixer_" : "_mlp_") + tag,
                dump.forward, stream);
}

void maybe_dump_final(const Tensor& hidden, cudaStream_t stream) {
    ResidualDump& dump = residual_dump();
    if (dump.dir.empty() || hidden.ne[1] > 64 || dump.forward > dump.limit) { return; }
    dump_tensor(hidden, "f" + std::to_string(dump.forward) + "_final", dump.forward, stream);
}

std::size_t plane_bytes(std::int32_t rows, std::int32_t tokens, DType dtype) {
    return round_up(static_cast<std::size_t>(rows) * static_cast<std::size_t>(tokens) *
                    dtype_size(dtype));
}

std::size_t mix_capacity(const family::TextGeometry& g, std::int32_t first, std::int32_t last) {
    return ops::hyper_connection_mix_workspace_capacity_bytes(g.hc_streams, g.hidden, g.hc_low_rank, first,
                                                              last);
}

std::size_t w8_capacity(std::int32_t rows, std::int32_t columns, std::int32_t first,
                        std::int32_t last) {
    return ops::linear_workspace_capacity_bytes(QType::W8G32_F16S, rows, columns, kPolicy, first,
                                                last);
}

// Mixes the residual streams into `hidden` and keeps the inject gates for the combine.
void mix_into(const family::TextGeometry& g, const Tensor& residual, const ops::HyperConnectionWeights& weights, Tensor& hidden,
              WorkspaceArena& workspace, cudaStream_t stream) {
    const std::int32_t tokens = residual.ne[1];
    const InjectScratch& scratch = inject_scratch_for_current_device(g.hc_streams);
    const std::size_t needed =
        static_cast<std::size_t>(g.hc_streams) * static_cast<std::size_t>(tokens) * sizeof(float);
    if (scratch.data == nullptr || needed > scratch.bytes) {
        throw std::logic_error("qwen4exp: inject scratch is missing or too small for this forward");
    }
    Tensor inject(scratch.data, DType::FP32, {g.hc_streams, tokens});
    ops::hyper_connection_mix(residual, weights, g.hc_streams, g.rms_epsilon, hidden, &inject, workspace,
                              stream);
    t_inject = inject;
    t_inject_geometry = g;
    CUDA_CHECK(cudaGetDevice(&t_inject_device));
    maybe_dump_block("mixed", hidden, stream);
    maybe_dump_block("inject", inject, stream);
}

// Scatters a block output into the residual streams with the gates the mix kept.
void combine_into(const Tensor& block_output, Tensor& residual, cudaStream_t stream) {
    const auto& g = t_inject_geometry;
    if (t_inject.data == nullptr || t_inject.ne[1] != residual.ne[1]) {
        throw std::logic_error("qwen4exp: combine without a matching mix");
    }
    if (family::ExpertCache::pool_floor() == 0) {
        ops::hyper_connection_combine(block_output, t_inject, residual, stream);
        g_dump_block += 1;
        check_device_handoff("the inject gates", t_inject_device);
        t_inject_device = -1;
        t_inject = Tensor{};
        return;
    }
    family::ExpertCache& cache = family::ExpertCache::for_current_device(
        moe_geometry(g), (g.layers + g.mtp_layers));
    cache.tick_combine();
    maybe_dump_block("blockout", block_output, stream);
    if (cache.has_pending_partial()) {
        // The host round's partial is added in the same pass that recombines the streams: the
        // cache joins the round (its done flag or the side stream's event) before the kernel.
        const float* partial = cache.wait_pending_partial(stream);
        ops::hyper_connection_combine(block_output, partial, t_inject, residual, stream);
        cache.finish_pending_partial(block_output, stream);
    } else {
        ops::hyper_connection_combine(block_output, t_inject, residual, stream);
    }
    maybe_dump_block("combined", residual, stream);
    g_dump_block += 1;
    check_device_handoff("the inject gates", t_inject_device);
    t_inject_device = -1;
    t_inject = Tensor{};
}

Tensor rows_of(const Tensor& fused, std::int32_t begin, std::int32_t count, WorkspaceArena& work,
               cudaStream_t stream) {
    Tensor out = work.alloc(DType::BF16, {count, fused.ne[1]});
    ops::extract_bf16_columns(fused, begin, out, stream);
    return out;
}

} // namespace

std::vector<GraphExecutionProfile> Variant::ordinary_graph_profiles(std::uint32_t capacity) {
    return graph_profiles_through(capacity - 1, {127, 511, 2047});
}

std::vector<GraphExecutionProfile> Variant::mtp_graph_profiles(std::uint32_t capacity,
                                                               std::uint32_t draft_window) {
    // The draft head's rounds are captured over the same visible-key bands the trunk uses,
    // shifted by the draft window: a verify round writes the window's keys before the head
    // reads them, so the head's envelope trails the trunk's by that much.
    if (draft_window == 0 || capacity == 0) { return {}; }
    std::vector<std::uint32_t> ends;
    for (const std::uint32_t visible_end : {128U, 512U, 2048U, 4096U, 8198U, 16390U, 32768U}) {
        if (visible_end >= 2 * draft_window) { ends.push_back(visible_end - 2 * draft_window); }
    }
    std::sort(ends.begin(), ends.end());
    ends.erase(std::unique(ends.begin(), ends.end()), ends.end());
    return graph_profiles_through(capacity - 1, ends);
}

std::vector<GraphExecutionProfile> Variant::dflash_graph_profiles(std::uint32_t, std::uint32_t,
                                                                  std::uint32_t) {
    return {};
}

// --- residual hooks ---------------------------------------------------------------------------

void Variant::embed_residual(const ModelView& model, const Tensor& ids, Tensor& residual,
                             WorkspaceArena& workspace, cudaStream_t stream) {
    const auto& g = model.geometry;
    // Transient: the broadcast consumes the embedding before anything else runs.
    auto scope                = workspace.scope();
    const std::int32_t tokens = residual.ne[1];
    Tensor embedded           = workspace.alloc(DType::BF16, {g.hidden, tokens});
    ops::embedding(ids, model.token_embedding, embedded, stream);
    ops::broadcast_streams(embedded, g.hc_streams, residual, stream);
}

void Variant::prepare_expert_split(const ModelView& model) {
    if (family::ExpertCache::pool_floor() == 0) { return; }
    const auto& g = model.geometry;
    family::ExpertCache& cache = family::ExpertCache::for_current_device(
        moe_geometry(g), (g.layers + g.mtp_layers));
    const auto prepare = [&](const auto& layers) {
        for (const auto& layer : layers) {
            const auto& payload = layer.post_mixer;
            if (payload.layer < 0 || payload.host_gate_up == nullptr) { continue; }
            cache.prepare_split(family::BankedMixture{payload.layer, g.layers + g.mtp_layers,
                &payload.op, payload.gate_up_planes, payload.down_planes,
                payload.host_gate_up, payload.host_down});
            return true;
        }
        return false;
    };
    if (!prepare(model.gdn_layers)) { (void)prepare(model.full_layers); }

}

void Variant::prewarm_device_scratch(const family::TextGeometry& g) {
    InjectScratch& scratch = inject_scratch_for_current_device(g.hc_streams);
    if (scratch.data == nullptr) {
        scratch.bytes = kInjectScratchTokens * g.hc_streams * sizeof(float);
        scratch.storage = std::make_unique<DeviceArena>(scratch.bytes);
        scratch.data = scratch.storage->base();
    }
    if (family::ExpertCache::pool_floor() != 0) {
        (void)family::ExpertCache::for_current_device(moe_geometry(g), g.layers + g.mtp_layers);
    }
}

void Variant::final_residual_mix(const ModelView& model, const Tensor& residual, Tensor& hidden,
                                 WorkspaceArena& workspace, cudaStream_t stream) {
    const auto& g = model.geometry;
    ops::hyper_connection_mix(residual, model.output_mix, g.hc_streams, g.rms_epsilon, hidden, nullptr,
                              workspace, stream);
    maybe_dump_final(hidden, stream);
}

void Variant::attention_norm(const Tensor& residual, const FullAttentionProjectionWeights& weights,
                             Tensor& hidden, WorkspaceArena& workspace, cudaStream_t stream) {
    const auto& g = weights.geometry;
    mix_into(g, residual, weights.mix, hidden, workspace, stream);
}

void Variant::post_mixer_norm(const Tensor& residual, const PostMixerWeights& weights,
                              Tensor& hidden, WorkspaceArena& workspace, cudaStream_t stream) {
    const auto& g = weights.geometry;
    mix_into(g, residual, weights.mix, hidden, workspace, stream);
}

void Variant::layer_prologue(const ModelView& model, int layer, Tensor& residual,
                             const family::detail::PrologueColumns& columns,
                             NgramPleStatePool* ple_state, WorkspaceArena& workspace,
                             cudaStream_t stream) {
    const auto& g = model.geometry;
    maybe_dump_layer(layer, residual, stream);
    if (layer != model.ple.layer) { return; }
    if (ple_state == nullptr || ple_state->empty()) {
        throw std::logic_error("qwen4exp: the PLE layer needs its state pool");
    }
    ops::NgramPleColumns ple_columns{columns.ids, columns.segment_begin, columns.slots,
                                     columns.segment_last};
    ops::NgramPleState state{ple_state->history, ple_state->conv_state};
    maybe_dump_block("ple_in", residual, stream);
    ops::ngram_ple_forward(residual, ple_columns, model.ple.hash, model.ple.table, model.ple.op,
                           state, g.hc_streams, g.ple_conv_kernel,
                           g.ple_ngram, g.rms_epsilon, workspace, stream);
    maybe_dump_block("ple_out", residual, stream);
}

void Variant::debug_probe(const char* tag, const Tensor& tensor, std::int32_t /*layer_count*/, cudaStream_t stream) {
    ResidualDump& dump = residual_dump();
    if (dump.dir.empty() || dump.forward > dump.limit || g_dump_layer < 0 || g_dump_layer > 1) {
        return;
    }
    // Flatten to [ne0, rest] so the dump header describes the payload.
    const std::int64_t rest = tensor.numel() / tensor.ne[0];
    if (rest > 8192) { return; }
    Tensor flat = tensor.view({tensor.ne[0], static_cast<std::int32_t>(rest)});
    dump_tensor(flat,
                "f" + std::to_string(dump.forward) + "_L" + std::to_string(g_dump_layer) +
                    (g_dump_block == 0 ? "_mixer_" : "_mlp_") + tag,
                dump.forward, stream);
}

NgramPleStatePoolSpec Variant::ple_state_spec(const family::TextGeometry& g, std::int32_t slot_count) {
    return NgramPleStatePoolSpec{
        .history_tokens = g.ple_ngram - 1,
        .conv_history   = g.ple_conv_history(),
        .channels       = g.residual,
        .slot_count     = slot_count,
        .eos_token      = g.ple_eos_token,
    };
}

std::size_t Variant::layer_prologue_workspace_capacity_bytes(const family::TextGeometry& g, std::int32_t first,
                                                             std::int32_t last) {
    // The PLE forward and the transient embedding never overlap; reserve the larger.
    return std::max(g.ple_ngram ? ops::ngram_ple_workspace_capacity_bytes(g.hc_streams, g.hidden,
                                                            g.ple_embed(),
                                                            g.ple_heads(), first, last) : std::size_t{0},
                    plane_bytes(g.hidden, last, DType::BF16));
}

// --- projections -------------------------------------------------------------------------------

void Variant::attention_projection(const Tensor& hidden,
                                   const FullAttentionProjectionWeights& weights, Tensor& query,
                                   Tensor& gate, Tensor& key, Tensor& value, family::TextPhase,
                                   WorkspaceArena& workspace, cudaStream_t stream) {
    const auto& g = weights.geometry;
    auto scope                = workspace.scope();
    const std::int32_t tokens = hidden.ne[1];
    Tensor fused = workspace.alloc(DType::BF16, {(2 * g.query_size() + 2 * g.kv_size()), tokens});
    ops::linear(hidden, weights.query_key_gate_value, fused, kPolicy, workspace, stream);
    // Row order from the converter: q | k | gate | v.
    ops::extract_bf16_columns(fused, 0, query, stream);
    ops::extract_bf16_columns(fused, g.query_size(), key, stream);
    ops::extract_bf16_columns(fused, g.query_size() + g.kv_size(), gate, stream);
    ops::extract_bf16_columns(fused, 2 * g.query_size() + g.kv_size(), value,
                              stream);
    family::apply_lora_qkv(weights.query_key_gate_value, hidden, query, key, value, stream);
    family::apply_lora(weights.query_key_gate_value, family::kAttentionGatePort, hidden, gate, stream);
}

void Variant::attention_output_projection(const Tensor& attention, const Weight& weight,
                                          Tensor& residual, family::TextPhase,
                                          WorkspaceArena& workspace, cudaStream_t stream) {
    auto scope     = workspace.scope();
    Tensor output  = workspace.alloc(DType::BF16, {weight.n, attention.ne[1]});
    ops::linear(attention, weight, output, kPolicy, workspace, stream);
    // Before the hyper-connection combine: the delta belongs to o_proj's output,
    // and the combine is what distributes it across the residual streams.
    family::apply_lora(weight, 3, attention, output, stream);
    combine_into(output, residual, stream);
}

// ---------------------------------------------------------------------------------------------
// NextN draft head
//
// The head folds the next token's embedding into the trunk's wide residual, runs one trunk
// block over it, and collapses the result with its own mixer before reusing the trunk's LM
// head. The block is the family's business -- it is an ordinary layer and the family runs it
// with the trunk's own mixer. These two are what is the head's own.
// ---------------------------------------------------------------------------------------------

const FullAttentionWeights& Variant::mtp_block(const ModelView& model) {
    if (!model.mtp_head.present) {
        throw std::logic_error("qwen4exp: the draft head's block is not resident");
    }
    return model.mtp_block;
}

void Variant::mtp_fold(const ModelView& model, const Tensor& embedding, const Tensor& hidden,
                       Tensor& residual, WorkspaceArena& workspace, cudaStream_t stream) {
    const auto& g = model.geometry;
    const MtpHeadWeights& head = model.mtp_head;
    if (!head.present) { throw std::logic_error("qwen4exp: the draft head is not resident"); }
    const std::int32_t tokens = embedding.ne[1];
    auto scope = workspace.scope();

    // The two inputs, each under its own norm. `hidden` is the wide residual, so its norm is
    // per stream -- the same first line the layer mixers run.
    Tensor e = workspace.alloc(DType::BF16, {g.hidden, tokens});
    Tensor h = workspace.alloc(DType::BF16, {g.residual, tokens});
    ops::rmsnorm(embedding, head.embedding_norm, g.rms_epsilon, false, e, stream);
    ops::hyper_connection_norm(hidden, head.hidden_norm, g.hc_streams, g.rms_epsilon, h, stream);

    // One matmul over [e; h_s] per stream. The pack puts the stream fastest, so the result
    // read as [kHcWidth, tokens] is already stream-major within each column.
    Tensor packed = workspace.alloc(DType::BF16, {2 * g.hidden, g.hc_streams * tokens});
    ops::mtp_pack_fc_input_streams(e, h, g.hc_streams, packed, stream);
    Tensor folded = residual.view({g.hidden, g.hc_streams * tokens});
    ops::linear(packed, head.input_projection, folded, kPolicy, workspace, stream);
}

void Variant::mtp_collapse(const ModelView& model, const Tensor& residual, Tensor& hidden,
                           WorkspaceArena& workspace, cudaStream_t stream) {
    const auto& g = model.geometry;
    const MtpHeadWeights& head = model.mtp_head;
    if (!head.present) { throw std::logic_error("qwen4exp: the draft head is not resident"); }
    ops::hyper_connection_mix(residual, head.head_mix, g.hc_streams, g.rms_epsilon, hidden, nullptr, workspace,
                              stream);
}

std::size_t Variant::mtp_fold_workspace_capacity_bytes(const family::TextGeometry& g,
                                                       std::int32_t first, std::int32_t last) {
    if (first <= 0 || last < first) {
        throw std::invalid_argument("qwen4exp: draft-head token interval is empty");
    }
    const auto plane = [&](std::int32_t rows, std::int32_t columns) {
        return plane_bytes(rows, columns, DType::BF16);
    };
    return plane(g.hidden, last) + plane(g.residual, last) + plane(2 * g.hidden, g.hc_streams * last) +
           w8_capacity(g.hidden, 2 * g.hidden, g.hc_streams * first, g.hc_streams * last) +
           mix_capacity(g, first, last);
}

void Variant::mtp_attention_projection(const Tensor&, const MtpAttentionProjectionWeights&,
                                       Tensor&, Tensor&, Tensor&, Tensor&, WorkspaceArena&,
                                       cudaStream_t) {
    // The head's block is a trunk block; the family runs it through the trunk's own mixer, so
    // the fixed draft-tail projections are never reached. See `mtp_block_is_trunk_layer`.
    throw std::logic_error("qwen4exp: the draft head runs the trunk's block, not this path");
}

void Variant::mtp_kv_projection(const Tensor&, const MtpAttentionProjectionWeights&, Tensor&,
                                Tensor&, WorkspaceArena&, cudaStream_t) {
    throw std::logic_error("qwen4exp: MTP is not served");
}

void Variant::mtp_q_gate_projection(const Tensor&, const MtpAttentionProjectionWeights&, Tensor&,
                                    Tensor&, WorkspaceArena&, cudaStream_t) {
    throw std::logic_error("qwen4exp: MTP is not served");
}

void Variant::gdn_input_projection(const Tensor& hidden, const GdnProjectionWeights& weights,
                                   Tensor& qkv, Tensor& output_gate, family::TextPhase,
                                   WorkspaceArena& workspace, cudaStream_t stream) {
    const auto& g = weights.geometry;
    auto scope                = workspace.scope();
    const std::int32_t tokens = static_cast<std::int32_t>(hidden.ne[1] * hidden.ne[2]);
    Tensor flat_hidden        = hidden.view({g.hidden, tokens});
    Tensor fused = workspace.alloc(DType::BF16, {(g.convolution_dim() + g.value_dim()), tokens});
    ops::linear(flat_hidden, weights.query_key_value_z, fused, kPolicy, workspace, stream);
    maybe_dump_block("gdn_fused", fused, stream);
    Tensor qkv_flat  = qkv.view({g.convolution_dim(), tokens});
    Tensor gate_flat = output_gate.view({g.value_dim(), tokens});
    ops::extract_bf16_columns(fused, 0, qkv_flat, stream);
    ops::extract_bf16_columns(fused, g.convolution_dim(), gate_flat, stream);
    family::apply_lora_gdn_input(weights, flat_hidden, qkv_flat, gate_flat, stream);
}

void Variant::gdn_input_projection_snapshot(
    const Tensor& hidden, const GdnProjectionWeights& weights, const Tensor& conv_weight,
    Tensor& conv_states, const Tensor& valid_columns, const Tensor& initial_slot,
    const Tensor& snapshot_base_slot, Tensor& query, Tensor& key, Tensor& value,
    Tensor& output_gate, family::TextPhase phase, WorkspaceArena& workspace,
    cudaStream_t stream) {
    const auto& g = weights.geometry;
    auto scope               = workspace.scope();
    const std::int32_t width = hidden.ne[1];
    const std::int32_t batch = hidden.ne[2];
    const std::int32_t tokens = width * batch;
    Tensor projected = workspace.alloc(DType::BF16, {g.convolution_dim(), width, batch});
    gdn_input_projection(hidden, weights, projected, output_gate, phase, workspace, stream);
    Tensor convolved = workspace.alloc(DType::BF16, {g.convolution_dim(), width, batch});
    ops::causal_conv1d_silu_snapshot(projected, conv_weight, conv_states, valid_columns,
                                     initial_slot, snapshot_base_slot, convolved, stream);
    Tensor convolved_flat = convolved.view({g.convolution_dim(), tokens});
    maybe_dump_block("gdn_conv", convolved_flat, stream);
    Tensor query_flat     = query.view({g.key_dim(), tokens});
    Tensor key_flat       = key.view({g.key_dim(), tokens});
    Tensor value_flat     = value.view({g.value_dim(), tokens});
    ops::extract_bf16_columns(convolved_flat, 0, query_flat, stream);
    ops::extract_bf16_columns(convolved_flat, g.key_dim(), key_flat, stream);
    ops::extract_bf16_columns(convolved_flat, 2 * g.key_dim(), value_flat, stream);
}

void Variant::gdn_input_projection_record(const Tensor& hidden,
                                          const GdnProjectionWeights& weights,
                                          const Tensor& conv_weight, const Tensor& conv_states,
                                          const Tensor& valid_columns, const Tensor& initial_slots,
                                          Tensor& conv_record, Tensor& query, Tensor& key,
                                          Tensor& value, Tensor& output_gate,
                                          family::TextPhase phase, WorkspaceArena& workspace,
                                          cudaStream_t stream) {
    const auto& g = weights.geometry;
    // A speculative round records the pre-convolution projection instead of advancing the
    // convolution state: a rejected draft replays the scan from the record, so the recurrent
    // state is never rolled back, only re-derived. The projection is the layer's own; the
    // convolution tail is the family's, the same one the fused profiles compose with.
    gdn_input_projection(hidden, weights, conv_record, output_gate, phase, workspace, stream);
    ops::detail::gdn_projected_conv_record_launch(conv_record, conv_weight, conv_states,
                                                  valid_columns, initial_slots, query, key, value,
                                                  stream);
}

void Variant::gdn_output_projection(const Tensor& hidden, const Weight& weight, Tensor& residual,
                                    family::TextPhase, WorkspaceArena& workspace,
                                    cudaStream_t stream) {
    auto scope    = workspace.scope();
    Tensor output = workspace.alloc(DType::BF16, {weight.n, hidden.ne[1]});
    maybe_dump_block("gdn_final", hidden, stream);
    ops::linear(hidden, weight, output, kPolicy, workspace, stream);
    maybe_dump_block("gdn_out", output, stream);
    family::apply_lora(weight, family::kGdnOutputPort, hidden, output, stream);
    combine_into(output, residual, stream);
}

void Variant::gdn_norm_control_projection(const Tensor& residual, const Tensor&, float,
                                          const GdnProjectionWeights& weights, Tensor& hidden,
                                          Tensor& gates, Tensor& beta, WorkspaceArena& workspace,
                                          cudaStream_t stream) {
    const auto& g = weights.geometry;
    mix_into(g, residual, weights.mix, hidden, workspace, stream);
    auto scope                = workspace.scope();
    const std::int32_t tokens = hidden.ne[1];
    const std::int32_t heads  = g.gdn_value_heads;
    Tensor ab                 = workspace.alloc(DType::BF16, {2 * heads, tokens});
    ops::detail::bf16_cublaslt_gemm(weights.a_b_projection, hidden, ab, stream);
    Tensor a = rows_of(ab, 0, heads, workspace, stream);
    Tensor b = rows_of(ab, heads, heads, workspace, stream);
    family::apply_lora(weights.a_b_projection, family::kGdnAPort, hidden, a, stream);
    family::apply_lora(weights.a_b_projection, family::kGdnBPort, hidden, b, stream);
    ops::lora_auto_bias(weights.dt_bias, a, stream);
    ops::gdn_gating(a, b, weights.a_log, weights.dt_bias, gates, beta, stream);
    maybe_dump_block("gdn_ab", ab, stream);
    maybe_dump_block("gdn_g", gates, stream);
    maybe_dump_block("gdn_beta", beta, stream);
}

void Variant::post_mixer(const Tensor& hidden, const PostMixerWeights& weights, Tensor& residual,
                         family::TextPhase, WorkspaceArena& workspace, cudaStream_t stream) {
    const auto& g = weights.geometry;
    auto scope                = workspace.scope();
    const std::int32_t tokens = hidden.ne[1];
    // The MoE op adds into its destination; a zeroed plane turns that into a plain store.
    Tensor output = workspace.alloc(DType::BF16, {g.hidden, tokens});
    CUDA_CHECK(cudaMemsetAsync(output.data, 0, output.bytes(), stream));
    if (weights.host_gate_up != nullptr) {
        auto& cache = family::ExpertCache::for_current_device(moe_geometry(g), g.layers + g.mtp_layers);
        // Routed experts come from the device slot pool, and the misses the split hands the
        // host come back as a partial the combine adds (family/impl/moe/expert_cache.h).
        cache.run(family::BankedMixture{weights.layer, (g.layers + g.mtp_layers), &weights.op,
                                        weights.gate_up_planes, weights.down_planes,
                                        weights.host_gate_up, weights.host_down},
                  hidden, output, workspace, stream);
    } else {
        const DeviceSpan storage = workspace.alloc_bytes(ops::sparse_moe_workspace_capacity_bytes(
            ops::sparse_moe_geometry(weights.op), weights.op.routed_gate_up.qtype,
            weights.op.routed_down.qtype, tokens, tokens));
        WorkspaceArena leaf(storage);
        ops::sparse_moe(hidden, weights.op, ops::SparseMoeEpilogue::AddResidual, output, leaf,
                        stream);
    }
    combine_into(output, residual, stream);
}

void Variant::mtp_post_mixer(const Tensor&, const MtpPostMixerWeights&, Tensor&, WorkspaceArena&,
                             cudaStream_t) {
    throw std::logic_error("qwen4exp: MTP is not served");
}

// --- workspace capacities ----------------------------------------------------------------------

std::size_t Variant::mtp_attention_projection_workspace_capacity_bytes(const family::TextGeometry& g, std::int32_t, std::int32_t) {
    return 0;
}

std::size_t Variant::mtp_kv_projection_workspace_capacity_bytes(const family::TextGeometry& g, std::int32_t, std::int32_t) {
    return 0;
}

std::size_t Variant::mtp_q_gate_projection_workspace_capacity_bytes(const family::TextGeometry& g, std::int32_t, std::int32_t) {
    return 0;
}

std::size_t Variant::attention_projection_workspace_capacity_bytes(const family::TextGeometry& g, WeightsProfile,
                                                                   family::TextPhase,
                                                                   std::int32_t first,
                                                                   std::int32_t last) {
    // The mix hook's planes live in the same mixer scope as the projection.
    return mix_capacity(g, first, last) +
           plane_bytes((2 * g.query_size() + 2 * g.kv_size()), last, DType::BF16) +
           w8_capacity((2 * g.query_size() + 2 * g.kv_size()), g.hidden, first, last);
}

std::size_t Variant::attention_output_projection_workspace_capacity_bytes(const family::TextGeometry& g, WeightsProfile,
                                                                          family::TextPhase,
                                                                          std::int32_t first,
                                                                          std::int32_t last) {
    return plane_bytes(g.hidden, last, DType::BF16) +
           w8_capacity(g.hidden, g.query_size(), first, last);
}

std::size_t Variant::gdn_input_projection_workspace_capacity_bytes(const family::TextGeometry& g, WeightsProfile,
                                                                   family::TextPhase,
                                                                   std::int32_t first,
                                                                   std::int32_t last) {
    return plane_bytes((g.convolution_dim() + g.value_dim()), last, DType::BF16) +
           w8_capacity((g.convolution_dim() + g.value_dim()), g.hidden, first, last);
}

std::size_t Variant::gdn_input_projection_snapshot_workspace_capacity_bytes(const family::TextGeometry& g, WeightsProfile profile, family::TextPhase phase, std::int32_t batch_size,
    std::int32_t min_width, std::int32_t max_width) {
    const std::int32_t tokens = batch_size * max_width;
    return 2 * plane_bytes(g.convolution_dim(), tokens, DType::BF16) +
           gdn_input_projection_workspace_capacity_bytes(g, profile, phase, batch_size * min_width,
                                                         tokens);
}

std::size_t Variant::gdn_input_projection_record_workspace_capacity_bytes(
    const family::TextGeometry& g, WeightsProfile profile, family::TextPhase phase,
    std::int32_t batch, std::int32_t first, std::int32_t last) {
    return gdn_input_projection_workspace_capacity_bytes(g, profile, phase, batch * first, batch * last);
}

std::size_t Variant::gdn_output_projection_workspace_capacity_bytes(const family::TextGeometry& g, WeightsProfile,
                                                                    family::TextPhase,
                                                                    std::int32_t first,
                                                                    std::int32_t last) {
    return plane_bytes(g.hidden, last, DType::BF16) +
           w8_capacity(g.hidden, g.value_dim(), first, last);
}

std::size_t Variant::gdn_norm_control_projection_workspace_capacity_bytes(const family::TextGeometry& g, std::int32_t first,
                                                                          std::int32_t last) {
    const std::int32_t heads = g.gdn_value_heads;
    return mix_capacity(g, first, last) + plane_bytes(2 * heads, last, DType::BF16) +
           2 * plane_bytes(heads, last, DType::BF16);
}

std::size_t Variant::post_mixer_workspace_capacity_bytes(const family::TextGeometry& g, WeightsProfile, family::TextPhase,
                                                         std::int32_t first, std::int32_t last) {
    return mix_capacity(g, first, last) + plane_bytes(g.hidden, last, DType::BF16) +
           round_up(ops::sparse_moe_workspace_capacity_bytes(moe_geometry(g),
                                                             QType::W8G32_F16S, QType::W8G32_F16S,
                                                             first, last));
}

std::size_t Variant::mtp_post_mixer_workspace_capacity_bytes(const family::TextGeometry& g, std::int32_t, std::int32_t) {
    return 0;
}


// This family's non-attending layers run a gated delta net, not a short convolution. The leaf
// is declared because the runtime is a template over the whole Variant interface, and it
// refuses rather than returning quietly: a mixer that contributed nothing to the residual reads
// as a model that merely answers badly.
void Variant::short_conv_projection(const Tensor&, const Tensor&, float,
                                    const GdnProjectionWeights&, Tensor&, family::TextPhase,
                                    WorkspaceArena&, cudaStream_t) {
    throw std::logic_error("short_conv_projection: this target's linear mixer is a gated delta net");
}

std::size_t Variant::short_conv_projection_workspace_capacity_bytes(const family::TextGeometry&,
                                                                    WeightsProfile,
                                                                    family::TextPhase,
                                                                    std::int32_t, std::int32_t) {
    return 0;
}

} // namespace sinfer::targets::qwen4exp::detail
