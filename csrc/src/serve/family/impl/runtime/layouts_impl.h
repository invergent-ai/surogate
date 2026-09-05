#include "family/impl/runtime/instance.h"
#include "family/impl/runtime/layouts.h"
#include "family/impl/runtime/residual_policy.h"
#include "ops/linear/marlin/marlin_plane.h"
#include "family/impl/runtime/linear_state_slots.h"
#include "family/impl/runtime/vision_context.h"
#include "api/ops/qsa_indexer.h"
#include "family/impl/runtime/workspace_recipe.h"

#include "core/device.h"
#include "api/ops/gated_delta_net.h"
#include "api/ops/gdn_gating_proj.h"
#include "api/ops/gdn_input_proj.h"
#include "api/ops/linear_add.h"
#include "api/ops/linear_swiglu.h"
#include "api/ops/sampling.h"
#include "api/ops/speculative_round.h"
#include "api/ops/gqa_attention.h"
#include "api/ops/bidirectional_gqa_attention.h"
#include "api/ops/swa.h"

#include <algorithm>
#include <initializer_list>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace sinfer::family::detail::SINFER_FAMILY_RUNTIME_NS {
namespace {

/// How many of a checkpoint's layers attend, and how many are linear.
///
/// The same question `ModelConfig` answers, and it has to be answered the same way: from the
/// artifact's schedule where it declares one, and from the target's compiled period otherwise.
/// Counting these two differently is how a state pool comes to be sized for a layer inventory
/// the runtime does not have.
[[nodiscard]] inline std::int32_t geometry_full_attention_layers(const family::TextGeometry& g) {
    std::int32_t count = 0;
    for (std::int32_t layer = 0; layer < g.layers; ++layer) {
        count += (g.attention_schedule_declared ? g.layer_attends(layer)
                                                : TextConfig::is_full_attention(static_cast<int>(layer)))
                     ? 1
                     : 0;
    }
    return count;
}

[[nodiscard]] inline std::int32_t geometry_gdn_layers(const family::TextGeometry& g) {
    return g.layers - geometry_full_attention_layers(g);
}

} // namespace

namespace {

[[nodiscard]] constexpr std::size_t round_up_256(std::size_t bytes) noexcept {
    return (bytes + 255U) & ~static_cast<std::size_t>(255U);
}

// QSA indexer width of the target (design/INFERENCE.md, phase 4), or 0 when the model has no
// indexer: the KV cache then carries one extra BF16 plane per full-attention layer.
// Transient bytes one full-attention layer's QSA indexer needs for `tokens` columns over a
// history of `keys` cells; zero for a target without an indexer.
template <class V>
[[nodiscard]] std::size_t variant_indexer_workspace_bytes(std::int32_t tokens,
                                                          std::int32_t keys) noexcept {
    if constexpr (!requires { V::indexer_head_dim; }) {
        (void)tokens;
        (void)keys;
        return 0;
    } else {
        const auto columns = static_cast<std::size_t>(std::max(tokens, 1));
        const auto width   = static_cast<std::size_t>(V::indexer_head_dim);
        const auto heads   = static_cast<std::size_t>(V::indexer_heads);
        const std::size_t raw_keys  = round_up_256(columns * width * sizeof(std::uint16_t));
        const std::size_t queries   = round_up_256(columns * width * heads * sizeof(std::uint16_t));
        const ops::QsaIndexerGeometry geometry{.head_dim   = V::indexer_head_dim,
                                               .heads      = V::indexer_heads,
                                               .block      = V::indexer_block,
                                               .top_k      = V::indexer_top_k,
                                               .rotary_dim = 0,
                                               .rope_theta = 0.0F,
                                               .rms_eps    = 0.0F};
        const std::size_t mask = round_up_256(
            static_cast<std::size_t>(ops::qsa_block_mask_words(keys, V::indexer_block)) * columns *
            sizeof(std::int32_t));
        const std::size_t scores =
            ops::qsa_indexer_select_workspace_capacity_bytes(static_cast<std::int32_t>(columns),
                                                             keys, geometry);
        return raw_keys + 2 * queries + mask + scores;
    }
}

template <class V>
[[nodiscard]] constexpr std::int32_t variant_indexer_head_dim() noexcept {
    if constexpr (requires { V::indexer_head_dim; }) {
        return static_cast<std::int32_t>(V::indexer_head_dim);
    } else {
        return 0;
    }
}

constexpr std::size_t kMiB        = 1024ULL * 1024ULL;
constexpr std::size_t kArenaAlign = 256ULL;

enum class GdnWorkspacePath : std::uint8_t {
    Prefill,
    Snapshot,
    ReplayRecord,
};

std::size_t checked_add(std::size_t a, std::size_t b, const char* label) {
    if (b > std::numeric_limits<std::size_t>::max() - a) { throw std::overflow_error(label); }
    return a + b;
}

std::size_t checked_mul(std::size_t a, std::size_t b, const char* label) {
    if (b != 0 && a > std::numeric_limits<std::size_t>::max() / b) {
        throw std::overflow_error(label);
    }
    return a * b;
}

std::int32_t checked_i32(std::uint64_t value, const char* label) {
    if (value == 0 ||
        value > static_cast<std::uint64_t>(std::numeric_limits<std::int32_t>::max())) {
        throw std::overflow_error(label);
    }
    return static_cast<std::int32_t>(value);
}

/// The storage a request for `Auto` resolves to for this geometry: BF16 when every layer
/// is attention, e4m3 when linear-attention layers carry the stack. KvCacheStorage::Auto
/// documents the measurements behind the split.
KvCacheStorage resolve_kv_storage(KvCacheStorage storage, const family::TextGeometry& geometry) {
    if (storage != KvCacheStorage::Auto) { return storage; }
    return geometry_gdn_layers(geometry) == 0 ? KvCacheStorage::BFloat16 : KvCacheStorage::Fp8E4M3;
}

DType kv_storage_dtype(KvCacheStorage storage) {
    switch (storage) {
    case KvCacheStorage::BFloat16: return DType::BF16;
    case KvCacheStorage::Int8Group64: return DType::I8;
    case KvCacheStorage::Fp8E4M3: return DType::FP8_E4M3FN;
    case KvCacheStorage::Auto: break; // resolved against the geometry before this
    }
    throw std::invalid_argument("KV cache storage must be resolved before it is a dtype");
}

std::uint32_t page_count(std::uint32_t capacity) {
    if (capacity == 0) { throw std::invalid_argument("Paged KV capacity must be positive"); }
    return 1U + (capacity - 1U) / static_cast<std::uint32_t>(kPagedKVPageSize);
}

template <class ProfileAllowance>
std::size_t graph_topology_allowance(const std::vector<GraphExecutionProfile>& profiles,
                                     ProfileAllowance&& profile_allowance, const char* label) {
    std::vector<std::pair<std::uint32_t, std::size_t>> classes;
    for (const GraphExecutionProfile profile : profiles) {
        const std::size_t allowance = profile_allowance(profile);
        const auto existing = std::find_if(classes.begin(), classes.end(), [&](const auto& entry) {
            return entry.first == profile.topology_class;
        });
        if (existing == classes.end()) {
            classes.emplace_back(profile.topology_class, allowance);
        } else {
            existing->second = std::max(existing->second, allowance);
        }
    }

    std::size_t total = 0;
    for (const auto& [topology_class, allowance] : classes) {
        (void)topology_class;
        total = checked_add(total, allowance, label);
    }
    return total;
}

TensorLayout add_tensor(LayoutBuilder& builder, DType dtype,
                        std::initializer_list<std::int32_t> shape, const char* label) {
    return builder.add_tensor(dtype, shape, kArenaAlign, label);
}

PersistentLayout persistent_layout(const SequencePlanImpl& plan) {
    const std::int32_t linear_state_slots =
        LinearStateSlots::state_slot_count(plan.max_concurrency, plan.rewrite_checkpoints);
    const auto effective_prefill_chunk =
        static_cast<std::int32_t>(std::min(plan.prefill_chunk, plan.capacity));
    const std::uint32_t logical_pages  = page_count(plan.capacity);
    // The Main pool's laid-out span; every other consumer of the physical count (MTP, DFlash,
    // the reservation) uses plan.main_page_groups, the cap.
    const std::uint32_t physical_pages = plan.main_page_virtual != 0 ? plan.main_page_virtual
                                                                     : plan.main_page_groups;
    const std::uint64_t mtp_extra_pages =
        plan.features.mtp()
            ? static_cast<std::uint64_t>(plan.max_concurrency) *
                  ((static_cast<std::uint64_t>(plan.draft_window - 1U) + kPagedKVPageSize - 1U) /
                   static_cast<std::uint32_t>(kPagedKVPageSize))
            : 0ULL;
    const std::uint32_t mtp_physical_pages = static_cast<std::uint32_t>(
        checked_i32(static_cast<std::uint64_t>(plan.main_page_groups) + mtp_extra_pages,
                    "MTP Paged KV physical pages exceed int32"));
    LayoutBuilder builder;
    PersistentLayout out;
    out.decoder = family::plan_decoder_state(
        builder, family::DecoderStateSpec{
                     .full_attention_layers     = geometry_full_attention_layers(plan.geometry),
                     .mtp_layers                = plan.geometry.mtp_layers,
                     .capacity                  = plan.capacity,
                     .kv_heads                  = plan.geometry.kv_heads,
                     .attention_head_dim        = plan.geometry.head_dim,
                     .kv_dtype                  = plan.kv_dtype,
                     .kv_quant_group            = plan.kv_quant_group,
                     .kv_skip_layers            = plan.kv_skip_layers,
                     .enable_mtp                = plan.features.mtp(),
                     .elastic_kv                = plan.elastic_kv,
                     .kv_table_rows             = static_cast<std::int32_t>(plan.max_concurrency),
                     .text_physical_page_groups = physical_pages,
                     .text_physical_page_cap    = plan.elastic_kv ? plan.main_page_groups : 0U,
                     .elastic_kv_overcommit     = plan.elastic_kv && plan.elastic_kv_overcommit,
                     .mtp_physical_page_groups  = mtp_physical_pages,
                     .linear_attention =
                         {
                             .layers         = geometry_gdn_layers(plan.geometry),
                             // A short convolution runs over the residual width itself and
                             // carries no recurrent image; a delta net convolves its own fused
                             // q|k|v and the recurrent matrix is the mixer. One pool, two
                             // shapes, and the mixer kind is what tells them apart.
                             .conv_channels  = schedule::kLinearMixer == family::LinearMixer::ShortConv
                                                   ? plan.geometry.hidden
                                                   : plan.geometry.convolution_dim(),
                             .conv_width     = plan.geometry.gdn_conv_state_width(),
                             .value_heads    = schedule::kLinearMixer == family::LinearMixer::ShortConv
                                                   ? 0
                                                   : plan.geometry.gdn_value_heads,
                             .value_head_dim = schedule::kLinearMixer == family::LinearMixer::ShortConv
                                                   ? 0
                                                   : plan.geometry.gdn_value_head_dim,
                             .key_head_dim   = schedule::kLinearMixer == family::LinearMixer::ShortConv
                                                   ? 0
                                                   : plan.geometry.gdn_key_head_dim,
                             .slot_count     = linear_state_slots,
                             .conv_dtype     = DType::BF16,
                         },
                     .ple = ResidualHooks<Variant>::ple_state_spec(linear_state_slots),
                 });
    if (plan.speculative_backend != SpeculativeBackend::None) {
        out.replay_records = plan_gdn_replay_records(
            builder, GdnReplayRecordSpec{
                         .layers          = geometry_gdn_layers(plan.geometry),
                         .record_capacity = static_cast<std::int32_t>(plan.max_concurrency),
                         .width           = static_cast<std::int32_t>(plan.draft_window + 1U),
                         .conv_channels   = plan.geometry.convolution_dim(),
                         .qk_heads        = plan.geometry.gdn_key_heads,
                         .value_heads     = plan.geometry.gdn_value_heads,
                         .key_dim         = plan.geometry.gdn_key_head_dim,
                         .value_dim       = plan.geometry.gdn_value_head_dim,
                     });
    }
    if constexpr (Variant::supports_dflash) {
        if (plan.features.dflash()) {
            DFlashPersistentLayout& dflash = out.dflash.emplace();
            dflash.local = plan_cyclic_kv_cache(builder, DFlashConfig::local_layers,
                                                DFlashConfig::local_capacity,
                                                DFlashConfig::kv_heads, DFlashConfig::head_dim,
                                                static_cast<std::int32_t>(plan.max_concurrency));
            dflash.rewrite_checkpoint_local = plan_cyclic_kv_cache(
                builder, DFlashConfig::local_layers, DFlashConfig::local_capacity,
                DFlashConfig::kv_heads, DFlashConfig::head_dim,
                static_cast<std::int32_t>(plan.max_concurrency));
            PagedKVPoolSpec full_pool{
                .page_group_count      = plan.main_page_groups,
                .logical_page_capacity = logical_pages,
                .table_rows            = static_cast<std::int32_t>(plan.max_concurrency),
                .plane_order           = PagedKVPlaneOrder::HeadMajor,
                .planes =
                    {
                        {DType::BF16, DFlashConfig::head_dim, DFlashConfig::kv_heads, 256},
                        {DType::BF16, DFlashConfig::head_dim, DFlashConfig::kv_heads, 256},
                    },
            };
            dflash.full = family::PagedKVCacheLayout{
                .pool        = plan_paged_kv_pool(builder, full_pool),
                .layers      = 1,
                .max_context = plan.capacity,
                .kv_heads    = DFlashConfig::kv_heads,
                .head_dim    = DFlashConfig::head_dim,
                .dtype       = DType::BF16,
                .quant_group = 0,
            };
            dflash.prefill_features = add_tensor(
                builder, DType::BF16, {DFlashConfig::feature_rows, effective_prefill_chunk},
                "DFlash prefill target features");
            dflash.prefill_positions = add_tensor(builder, DType::I32, {effective_prefill_chunk},
                                                  "DFlash prefill target positions");
            dflash.pending_features  = add_tensor(builder, DType::BF16,
                                                  {DFlashConfig::feature_rows,
                                                   static_cast<std::int32_t>(plan.draft_window + 1U),
                                                   static_cast<std::int32_t>(plan.max_concurrency)},
                                                  "DFlash pending target features");
        }
    }

    // A trunk-block draft head folds into the wide residual and hands it back, so the hidden
    // that crosses a round boundary -- into the head, into the continuation store, out of the
    // verify -- is the residual, not the model width. Identical for every other family, whose
    // residual is its hidden.
    const std::int32_t round_hidden =
        plan.features.mtp() && mtp_block_is_trunk_layer<Variant>() ? plan.geometry.residual
                                                                   : plan.geometry.hidden;
    out.round = family::begin_round_state_layout(
        builder, family::RoundStateSpec{.hidden         = round_hidden,
                                         .output_rows    = plan.geometry.output_rows,
                                         .batch_capacity = plan.max_concurrency,
                                         .draft_window   = plan.draft_window,
                                         .enable_mtp     = plan.features.mtp(),
                                         .enable_dflash  = plan.features.dflash()});
    out.prefill_hidden = add_tensor(
        builder, DType::BF16, {round_hidden, effective_prefill_chunk}, "step prefill hidden");
    family::complete_round_state_layout(builder, out.round);
    const auto i32 = [&](std::size_t n, const char* label) {
        return add_tensor(builder, DType::I32, {static_cast<std::int32_t>(n)}, label);
    };
    out.token_counts =
        add_tensor(builder, DType::I32,
                   {plan.geometry.token_domain, static_cast<std::int32_t>(plan.max_concurrency)},
                   "sampling token counts");
    const auto config_words = static_cast<std::int32_t>(
        (sizeof(ops::SamplingConfig) + sizeof(std::int32_t) - 1) / sizeof(std::int32_t));
    out.sampling_config = add_tensor(
        builder, DType::I32, {config_words, static_cast<std::int32_t>(plan.max_concurrency)},
        "sampling config");
    // These carry a lane's hidden from one round to the next, which is where a trunk-block
    // draft head reads its `h` from -- so they widen with the round's hidden.
    const std::int32_t persistent_hidden =
        plan.features.mtp() && mtp_block_is_trunk_layer<Variant>() ? plan.geometry.residual
                                                                   : plan.geometry.hidden;
    out.tail_hidden = add_tensor(
        builder, DType::BF16, {persistent_hidden, static_cast<std::int32_t>(plan.max_concurrency)},
        "tail hidden");
    out.rewrite_checkpoint_hidden = add_tensor(
        builder, DType::BF16, {persistent_hidden, static_cast<std::int32_t>(plan.max_concurrency)},
        "rewrite checkpoint hidden");
    out.bytes = builder.finish(kArenaAlign, "persistent layout");
    out.kv_payload_bytes =
        out.decoder.kv_payload_bytes() + (out.dflash ? out.dflash->kv_payload_bytes() : 0);
    // The reservation charges the planes' exact bytes, which grow linearly with the page
    // count as the capacity curve requires; the padded span the region reserves (each plane
    // on a mapping quantum) is virtual and at most one granule wider.
    if (out.decoder.text_kv.pool.spec.elastic) {
        const std::size_t per_page = out.decoder.text_kv.payload_bytes() / physical_pages;
        out.elastic_plane_bytes    = per_page * plan.main_page_groups;
    }
    return out;
}

WorkspacePlan build_workspace_plan(const SequencePlanImpl& plan) {
    const std::uint32_t chunk_u32 = std::min(plan.prefill_chunk, plan.capacity);
    if (chunk_u32 == 0 ||
        chunk_u32 > static_cast<std::uint32_t>(std::numeric_limits<std::int32_t>::max()) ||
        plan.draft_window >= static_cast<std::uint32_t>(std::numeric_limits<std::int32_t>::max())) {
        throw std::invalid_argument("sequence workspace dimensions are invalid");
    }
    const auto chunk  = static_cast<std::int32_t>(chunk_u32);
    const auto drafts = static_cast<std::int32_t>(plan.draft_window);
    const auto verify = drafts + 1;
    const ops::GqaExecutionEnvelope text_envelope{1, plan.capacity};

    const auto matrix  = [](WorkspaceLayoutBuilder& layout, DType dtype, std::int32_t rows,
                           std::int32_t tokens) { (void)layout.alloc(dtype, {rows, tokens}); };
    const auto scratch = [](WorkspaceLayoutBuilder& layout, std::size_t bytes) {
        if (bytes == 0) { return; }
        auto scope = layout.scope();
        (void)layout.alloc_bytes(bytes);
    };
    const auto finish = [](const WorkspaceLayoutBuilder& layout) { return layout.peak_bytes(1); };

    const auto text_common_root = [&](WorkspaceLayoutBuilder& layout, std::int32_t tokens) {
        (void)workspace_recipe::text_prefill_roots(
            layout, plan.geometry, tokens, plan.features.vision ? 3 : 0, plan.features.vision ? tokens : 0);
    };
    const auto attention_stage = [&](WorkspaceLayoutBuilder& layout, std::int32_t first,
                                     std::int32_t last, family::TextPhase phase,
                                     std::int32_t batch_size, std::int32_t min_width,
                                     std::int32_t max_width, ops::GqaExecutionEnvelope envelope) {
        auto stage = layout.scope();
        (void)workspace_recipe::text_attention_projection(layout, plan.geometry, last);
        scratch(layout, Variant::attention_projection_workspace_capacity_bytes(plan.geometry, plan.weights_profile,
                                                                               phase, first, last));
        // Marlin-tile residency (PATCHES.md #60) produces the fused parent whole
        // and splits it afterwards, so the plan must carry a [parent_rows, T]
        // staging buffer. Reserved whenever adoption is enabled, because an
        // adopted weight has no other route and a short workspace would be a
        // hard failure rather than a fallback.
        scratch(layout, ops::detail::marlin_fused_parent_bytes(
                            plan.geometry.query_size() * 2 + plan.geometry.kv_size() * 2, last));
        (void)workspace_recipe::text_attention_results(layout, plan.geometry, last);
        // QSA indexer (design/INFERENCE.md, phase 4): raw keys, queries and their norm, the
        // per-column block mask and the selection's score scratch. Reserved whenever the target
        // has an indexer — the selection engages only past its budget, but the append runs for
        // every column of every round, and a short workspace would be a hard failure.
        scratch(layout, variant_indexer_workspace_bytes<Variant>(
                            last, static_cast<std::int32_t>(envelope.max_visible_keys)));
        scratch(layout, ops::gqa_attention_workspace_capacity_bytes(
                            plan.geometry.head_dim, plan.geometry.query_heads,
                            plan.geometry.kv_heads, plan.kv_dtype, envelope,
                            batch_size, min_width, max_width));
        scratch(layout, Variant::attention_output_projection_workspace_capacity_bytes(plan.geometry, plan.weights_profile, phase, first, last));
    };
    const auto gdn_stage = [&](WorkspaceLayoutBuilder& layout, std::int32_t first,
                               std::int32_t last, family::TextPhase phase, GdnWorkspacePath path,
                               std::int32_t batch_size, std::int32_t min_width,
                               std::int32_t max_width) {
        auto stage = layout.scope();
        if constexpr (schedule::kLinearMixer == family::LinearMixer::ShortConv) {
            // The short-convolution mixer's whole round: the projection's three stacked parts,
            // the convolved value, and the two linears at either end. None of the delta net's
            // roots exist here -- their widths are the recurrent state's, which is zero.
            (void)workspace_recipe::short_conv(layout, plan.geometry, last);
            scratch(layout, Variant::short_conv_projection_workspace_capacity_bytes(
                                plan.geometry, plan.weights_profile, phase, first, last));
            scratch(layout, Variant::gdn_output_projection_workspace_capacity_bytes(
                                plan.geometry, plan.weights_profile, phase, first, last));
            return;
        }
        (void)workspace_recipe::gdn_control(layout, plan.geometry, last, schedule::kLinearMixer);
        scratch(layout, Variant::gdn_norm_control_projection_workspace_capacity_bytes(plan.geometry, first, last));
        (void)workspace_recipe::gdn_projection(layout, plan.geometry, last);
        if (path == GdnWorkspacePath::Snapshot) {
            scratch(layout, Variant::gdn_input_projection_snapshot_workspace_capacity_bytes(plan.geometry, plan.weights_profile, phase, batch_size, min_width, max_width));
        } else if (path == GdnWorkspacePath::ReplayRecord) {
            scratch(layout, Variant::gdn_input_projection_record_workspace_capacity_bytes(plan.geometry, plan.weights_profile, phase, batch_size, min_width, max_width));
        } else {
            (void)workspace_recipe::gdn_prefill_conv(layout, plan.geometry, last);
            scratch(layout, Variant::gdn_input_projection_workspace_capacity_bytes(plan.geometry, plan.weights_profile, phase, first, last));
            scratch(layout, ops::detail::marlin_fused_parent_bytes(
                                plan.geometry.key_dim() * 2 + plan.geometry.value_dim() * 2, last));
        }
        (void)workspace_recipe::gdn_recurrent_output(layout, plan.geometry, last);
        if (path == GdnWorkspacePath::Prefill) {
            scratch(layout,
                    ops::gated_delta_net_workspace_capacity_bytes(
                        plan.geometry.gdn_key_heads, plan.geometry.gdn_value_heads, true, first, last));
        }
        (void)workspace_recipe::gdn_normalized_output(layout, plan.geometry, last);
        scratch(layout, Variant::gdn_output_projection_workspace_capacity_bytes(plan.geometry, plan.weights_profile, phase, first, last));
    };
    const auto post_mixer_stage = [&](WorkspaceLayoutBuilder& layout, std::int32_t first,
                                      std::int32_t last, family::TextPhase phase) {
        auto stage = layout.scope();
        (void)workspace_recipe::post_mixer_hidden(layout, plan.geometry, last);
        scratch(layout, Variant::post_mixer_workspace_capacity_bytes(plan.geometry, plan.weights_profile, phase,
                                                                     first, last));
    };
    const auto target_body = [&](WorkspaceLayoutBuilder& layout, std::int32_t first,
                                 std::int32_t last, family::TextPhase phase, GdnWorkspacePath path,
                                 std::int32_t batch_size, std::int32_t min_width,
                                 std::int32_t max_width, ops::GqaExecutionEnvelope envelope) {
        if constexpr (ResidualHooks<Variant>::prologue) {
            // The staged column facts (ids are the caller's) and the prologue's own scratch.
            matrix(layout, DType::I32, 1, last);
            matrix(layout, DType::I32, 1, last);
            matrix(layout, DType::I32, 1, last);
            scratch(layout,
                    ResidualHooks<Variant>::layer_prologue_workspace_capacity_bytes(first, last));
        }
        attention_stage(layout, first, last, phase, batch_size, min_width, max_width, envelope);
        // A pure-attention target has no linear mixer, so it needs none of the
        // GDN scratch -- and cannot size it anyway: every extent below derives
        // from a GDN head count that is zero here, which the tensor constructor
        // rejects rather than silently allocating nothing.
        // Runtime, not `if constexpr`: how many layers are linear is the checkpoint's, and a
        // family whose schedule allows GDN may still be handed a checkpoint with none.
        if (geometry_gdn_layers(plan.geometry) > 0) {
            gdn_stage(layout, first, last, phase, path, batch_size, min_width, max_width);
        } else {
            (void)path;
        }
        post_mixer_stage(layout, first, last, phase);
    };
    const auto proposal_scratch = [&](WorkspaceLayoutBuilder& layout, std::int32_t columns) {
        if (plan.proposal_head == ProposalHead::Optimized) {
            matrix(layout, DType::BF16, Variant::draft_head_rows, columns);
        }
    };
    const auto mtp_stem = [&](WorkspaceLayoutBuilder& layout, std::int32_t tokens,
                              bool preembedded) {
        (void)workspace_recipe::mtp_stem(layout, plan.geometry, tokens, !preembedded);
    };
    const auto mtp_full_core = [&](WorkspaceLayoutBuilder& layout, std::int32_t tokens,
                                   ops::GqaExecutionEnvelope envelope) {
        auto core = layout.scope();
        mtp_stem(layout, tokens, false);
        (void)workspace_recipe::mtp_attention_projection(layout, plan.geometry, tokens);
        scratch(layout, Variant::mtp_attention_projection_workspace_capacity_bytes(plan.geometry, tokens, tokens));
        (void)workspace_recipe::mtp_attention_results(layout, plan.geometry, tokens);
        scratch(layout, ops::gqa_attention_workspace_capacity_bytes(
                            plan.geometry.head_dim, plan.geometry.query_heads,
                            plan.geometry.kv_heads, plan.kv_dtype, envelope,
                            1, tokens, tokens));
        (void)workspace_recipe::mtp_post_attention(layout, plan.geometry, tokens);
        scratch(layout, Variant::mtp_post_mixer_workspace_capacity_bytes(plan.geometry, tokens, tokens));
    };
    const auto mtp_full_call = [&](WorkspaceLayoutBuilder& layout, std::int32_t tokens,
                                   ops::GqaExecutionEnvelope envelope, bool build_proposal) {
        auto call = layout.scope();
        matrix(layout, DType::I32, 1, tokens);
        mtp_full_core(layout, tokens, envelope);
        if (build_proposal) {
            auto proposal = layout.scope();
            proposal_scratch(layout, 1);
        }
    };
    const auto mtp_prefill_chunk = [&](WorkspaceLayoutBuilder& layout, std::int32_t first,
                                       std::int32_t last, bool preembedded) {
        auto call = layout.scope();
        matrix(layout, DType::BF16, plan.geometry.hidden, 1);
        matrix(layout, DType::BF16, plan.geometry.hidden, 1);
        {
            auto bulk = layout.scope();
            mtp_stem(layout, last, preembedded);
            matrix(layout, DType::BF16, plan.geometry.kv_size(), last);
            matrix(layout, DType::BF16, plan.geometry.kv_size(), last);
            scratch(layout, Variant::mtp_kv_projection_workspace_capacity_bytes(plan.geometry, first, last));
            matrix(layout, DType::BF16, plan.geometry.kv_size(), last);
        }
        matrix(layout, DType::BF16, plan.geometry.query_size(), 1);
        matrix(layout, DType::BF16, plan.geometry.query_size(), 1);
        scratch(layout, Variant::mtp_q_gate_projection_workspace_capacity_bytes(plan.geometry, 1, 1));
        matrix(layout, DType::BF16, plan.geometry.query_size(), 1);
        matrix(layout, DType::I32, 3, 1);
        matrix(layout, DType::BF16, plan.geometry.query_size(), 1);
        scratch(layout, ops::gqa_attention_workspace_capacity_bytes(
                            plan.geometry.head_dim, plan.geometry.query_heads,
                            plan.geometry.kv_heads, plan.kv_dtype,
                            text_envelope, 1, 1, 1));
        matrix(layout, DType::BF16, plan.geometry.hidden, 1);
        matrix(layout, DType::BF16, plan.geometry.hidden, 1);
        scratch(layout, Variant::mtp_post_mixer_workspace_capacity_bytes(plan.geometry, 1, 1));
        proposal_scratch(layout, 1);
    };

    WorkspacePlan out;
    WorkspaceLayoutBuilder text_prefill;
    text_common_root(text_prefill, chunk);
    target_body(text_prefill, 1, chunk, family::TextPhase::Prefill, GdnWorkspacePath::Prefill, 1,
                1, chunk, text_envelope);
    scratch(text_prefill, ops::sampling_workspace_capacity_bytes(plan.geometry.token_domain, 1, 1));
    out.text_prefill = finish(text_prefill);

    for (std::int32_t batch = 1; batch <= static_cast<std::int32_t>(plan.max_concurrency);
         ++batch) {
        WorkspaceLayoutBuilder ordinary;
        matrix(ordinary, DType::BF16, plan.geometry.residual, batch);
        target_body(ordinary, batch, batch, family::TextPhase::Verify, GdnWorkspacePath::Snapshot,
                    batch, 1, 1, text_envelope);
        scratch(ordinary,
                ops::sampling_workspace_capacity_bytes(plan.geometry.token_domain, batch, batch));
        out.ordinary_round = std::max(out.ordinary_round, finish(ordinary));
    }

    if (plan.features.mtp()) {
        WorkspaceLayoutBuilder mtp_prefill;
        text_common_root(mtp_prefill, chunk);
        target_body(mtp_prefill, 1, chunk, family::TextPhase::Prefill, GdnWorkspacePath::Prefill,
                    1, 1, chunk, text_envelope);
        matrix(mtp_prefill, DType::I32, 1, chunk);
        if (plan.features.vision) {
            matrix(mtp_prefill, DType::BF16, plan.geometry.hidden, chunk);
            (void)workspace_recipe::visual_scatter_indices(mtp_prefill, chunk);
        }
        mtp_prefill_chunk(mtp_prefill, 1, chunk, plan.features.vision);
        for (std::int32_t i = 1; i < drafts; ++i) {
            matrix(mtp_prefill, DType::BF16, plan.geometry.hidden, 1);
            mtp_full_call(mtp_prefill, 1, text_envelope, true);
        }
        out.mtp_prefill = finish(mtp_prefill);

        WorkspaceLayoutBuilder mtp_batch;
        mtp_full_call(mtp_batch, verify, text_envelope, false);
        WorkspaceLayoutBuilder mtp_ar;
        mtp_full_call(mtp_ar, 1, text_envelope, true);
        WorkspaceLayoutBuilder mtp_align;
        mtp_full_call(mtp_align, 1, text_envelope, false);
        WorkspaceLayoutBuilder mtp_proposal;
        proposal_scratch(mtp_proposal, 1);
        const std::size_t accept = ops::speculative_accept_greedy_drafts_workspace_capacity_bytes(
            plan.geometry.token_domain, drafts, drafts, 1, 1);
        out.mtp_round = std::max({accept, finish(mtp_batch), finish(mtp_ar), finish(mtp_proposal)});
        out.ordinary_round = std::max(out.ordinary_round, finish(mtp_align));

        for (std::int32_t batch = 1; batch <= static_cast<std::int32_t>(plan.max_concurrency);
             ++batch) {
            const std::int32_t aggregate = batch * verify;
            WorkspaceLayoutBuilder target;
            matrix(target, DType::BF16, plan.geometry.residual, aggregate);
            target_body(target, aggregate, aggregate, family::TextPhase::Verify,
                        GdnWorkspacePath::ReplayRecord, batch, verify, verify, text_envelope);

            const auto mtp_decode_core = [&](WorkspaceLayoutBuilder& layout, std::int32_t width) {
                const std::int32_t tokens = batch * width;
                auto core                 = layout.scope();
                mtp_stem(layout, tokens, false);
                (void)workspace_recipe::mtp_attention_projection(layout, plan.geometry, tokens);
                scratch(layout,
                        Variant::mtp_attention_projection_workspace_capacity_bytes(plan.geometry, tokens, tokens));
                (void)workspace_recipe::mtp_attention_results(layout, plan.geometry, tokens);
                scratch(layout, ops::gqa_attention_workspace_capacity_bytes(
                                    plan.geometry.head_dim, plan.geometry.query_heads,
                                    plan.geometry.kv_heads, plan.kv_dtype,
                                    text_envelope, batch, width, width));
                (void)workspace_recipe::mtp_post_attention(layout, plan.geometry, tokens);
                scratch(layout, Variant::mtp_post_mixer_workspace_capacity_bytes(plan.geometry, tokens, tokens));
            };

            WorkspaceLayoutBuilder alignment;
            mtp_decode_core(alignment, verify);
            WorkspaceLayoutBuilder ar;
            mtp_decode_core(ar, 1);
            WorkspaceLayoutBuilder proposal;
            proposal_scratch(proposal, batch);
            const std::size_t batch_accept =
                ops::speculative_accept_greedy_drafts_workspace_capacity_bytes(
                    plan.geometry.token_domain, drafts, drafts, batch, batch);
            out.mtp_round = std::max({out.mtp_round, finish(target), finish(alignment), finish(ar),
                                      finish(proposal), batch_accept});
        }
    }

    if (plan.features.dflash()) {
        if constexpr (!Variant::supports_dflash) {
            throw std::logic_error("unsupported target reached DFlash scratch planning");
        } else {
            const auto dflash_context_capacity = [&](std::int32_t tokens, bool compact_input) {
                WorkspaceLayoutBuilder layout;
                if (compact_input) {
                    matrix(layout, DType::BF16, DFlashConfig::feature_rows, tokens);
                }
                (void)workspace_recipe::dflash_context<DFlashConfig>(layout, tokens);
                {
                    auto layer = layout.scope();
                    (void)workspace_recipe::dflash_context_layer<DFlashConfig>(layout, tokens);
                }
                return finish(layout);
            };
            const auto dflash_proposal_capacity = [&](std::int32_t width, std::int32_t batch) {
                WorkspaceLayoutBuilder layout;
                const std::int32_t tokens = width * batch;
                matrix(layout, DType::BF16, DFlashConfig::hidden, tokens);
                {
                    auto attention = layout.scope();
                    (void)workspace_recipe::dflash_attention<DFlashConfig>(layout, tokens);
                    scratch(layout,
                            std::max(ops::swa_workspace_capacity_bytes({0, plan.capacity}, width,
                                                                       width, batch),
                                     ops::bidirectional_gqa_attention_workspace_capacity_bytes(
                                         {0, plan.capacity}, width, width, batch)));
                    scratch(layout, ops::linear_add_workspace_capacity_bytes(
                                        QType::W8G32_F16S, DFlashConfig::hidden,
                                        DFlashConfig::query_size, tokens, tokens));
                }
                {
                    auto mlp = layout.scope();
                    (void)workspace_recipe::dflash_mlp<DFlashConfig>(layout, tokens);
                    scratch(layout, ops::linear_swiglu_workspace_capacity_bytes(
                                        QType::W8G32_F16S, 2 * DFlashConfig::intermediate,
                                        DFlashConfig::hidden, tokens, tokens));
                    scratch(layout, ops::linear_add_workspace_capacity_bytes(
                                        QType::W8G32_F16S, DFlashConfig::hidden,
                                        DFlashConfig::intermediate, tokens, tokens));
                }
                matrix(layout, DType::BF16, DFlashConfig::hidden, drafts * batch);
                matrix(layout, DType::BF16, DFlashConfig::hidden, drafts * batch);
                if (plan.proposal_head == ProposalHead::Optimized) {
                    matrix(layout, DType::BF16, Variant::draft_head_rows, drafts * batch);
                } else {
                    matrix(layout, DType::BF16, plan.geometry.output_rows, drafts * batch);
                }
                return finish(layout);
            };

            out.dflash_context = dflash_context_capacity(chunk, false);
            for (std::int32_t batch = 1; batch <= static_cast<std::int32_t>(plan.max_concurrency);
                 ++batch) {
                const std::int32_t aggregate = verify * batch;
                WorkspaceLayoutBuilder target;
                matrix(target, DType::BF16, plan.geometry.residual, aggregate);
                target_body(target, aggregate, aggregate, family::TextPhase::Verify,
                            GdnWorkspacePath::ReplayRecord, batch, verify, verify, text_envelope);
                const std::size_t accept =
                    ops::speculative_accept_greedy_drafts_workspace_capacity_bytes(
                        plan.geometry.token_domain, drafts, drafts, batch, batch);
                const std::size_t proposal = dflash_proposal_capacity(verify, batch);
                out.dflash_round           = std::max({out.dflash_round, finish(target), accept,
                                                       dflash_context_capacity(aggregate, true), proposal});
            }
        }
    }

    if (plan.features.vision) {
        constexpr std::uint32_t kFrontendMergedLimit  = 32768;
        constexpr std::uint32_t kFrontendSegmentLimit = 768 / 2;
        const std::uint32_t merged = std::min(plan.capacity, kFrontendMergedLimit);
        // The planner has no artifact in hand, so it sizes the envelope for the tower this
        // target compiles. A checkpoint that declares a wider one is caught at admission,
        // where the item's own workspace is measured against this capacity.
        out.vision_encode          = schedule::VisionContext::workspace_capacity_bytes(
            schedule::compiled_vision_geometry(), merged,
            std::min(merged, kFrontendSegmentLimit));
    }

    out.capacity = std::max({out.text_prefill, out.ordinary_round, out.mtp_prefill, out.mtp_round,
                             out.dflash_context, out.dflash_round, out.vision_encode});
    return out;
}

void validate_target_options(DeviceContext& device, const EngineOptions& options) {
    if (options.max_context == 0 || options.max_context > Variant::maximum_context) {
        throw std::invalid_argument("max_context exceeds the variant native context capacity");
    }
    if (options.prefill_chunk == 0 || options.prefill_chunk % kPrefillChunkAlignment != 0) {
        throw std::invalid_argument("prefill_chunk must be a nonzero multiple of 128");
    }
    if (options.max_concurrency == 0 || options.max_concurrency > kMaximumConcurrency) {
        throw std::invalid_argument("max_concurrency must be in [1,8]");
    }
    const std::uint32_t logical_pages = page_count(options.max_context);
    const std::uint32_t minimum_pages = std::max(logical_pages, options.max_concurrency);
    const std::uint64_t maximum_pages64 =
        static_cast<std::uint64_t>(options.max_concurrency) * logical_pages;
    if (maximum_pages64 > std::numeric_limits<std::uint32_t>::max()) {
        throw std::overflow_error("maximum Main KV page count exceeds uint32");
    }
    switch (options.kv_capacity.mode) {
    case KvCapacityMode::Explicit: {
        if (options.kv_capacity.explicit_tokens < options.max_context) {
            throw std::invalid_argument("kv_capacity must be at least max_context");
        }
        const std::uint32_t requested_pages = page_count(options.kv_capacity.explicit_tokens);
        if (requested_pages < minimum_pages || requested_pages > maximum_pages64) {
            throw std::invalid_argument(
                "kv_capacity is outside the usable range for max_context and max_concurrency");
        }
        break;
    }
    case KvCapacityMode::Automatic:
        break;
    default:
        throw std::invalid_argument("unknown kv_capacity policy");
    }
    switch (options.speculative.backend) {
    case SpeculativeBackend::None:
        if (options.speculative.draft_tokens != 0 ||
            options.speculative.proposal_head != ProposalHead::Full) {
            throw std::invalid_argument(
                "disabled speculative decoding requires draft_tokens=0 and the full proposal head");
        }
        break;
    case SpeculativeBackend::Mtp:
        if (options.speculative.draft_tokens == 0 ||
            options.speculative.draft_tokens > kMaximumMtpDraftTokens) {
            throw std::invalid_argument("MTP draft window must be in [1,5]");
        }
        break;
    case SpeculativeBackend::DFlash:
        if (kMaximumDFlashDraftTokens == 0) {
            throw std::invalid_argument("DFlash is not supported by this target");
        }
        if (options.speculative.draft_tokens == 0 ||
            options.speculative.draft_tokens > kMaximumDFlashDraftTokens) {
            throw std::invalid_argument("DFlash draft window must be in [1,15]");
        }
        if (options.enable_vision) {
            throw std::invalid_argument("DFlash and Vision cannot be enabled together");
        }
        break;
    }
    if (device.sm() != 120) {
        throw std::invalid_argument("Qwen3.6 family runtime requires compute capability 12.0");
    }
}

std::unique_ptr<SequencePlanImpl> build_sequence_candidate(const SequencePlanningInputs& inputs,
                                                           std::uint32_t main_page_groups) {
    if (main_page_groups == 0) {
        throw std::invalid_argument("Main KV physical page count must be positive");
    }
    auto impl                 = std::make_unique<SequencePlanImpl>();
    impl->weights_profile     = inputs.weights_profile;
    impl->geometry            = inputs.geometry;
    impl->capacity            = inputs.capacity;
    impl->main_page_groups    = main_page_groups;
    // Elastic: lay out the virtual maximum (every lane at full context); the physical cap
    // stays main_page_groups, which is what the reservation and admission are sized from.
    impl->main_page_virtual   = main_page_groups;
    if (inputs.elastic_kv) {
        const std::uint64_t virtual_pages =
            static_cast<std::uint64_t>(inputs.max_concurrency) * page_count(inputs.capacity);
        impl->main_page_virtual = static_cast<std::uint32_t>(
            std::max<std::uint64_t>(main_page_groups,
                                    std::min<std::uint64_t>(virtual_pages,
                                                            std::numeric_limits<std::uint32_t>::max())));
    }
    impl->kv_capacity         = static_cast<std::uint32_t>(checked_i32(
        static_cast<std::uint64_t>(main_page_groups) * static_cast<std::uint32_t>(kPagedKVPageSize),
        "resolved Paged KV capacity exceeds int32"));
    impl->max_concurrency     = inputs.max_concurrency;
    impl->prefill_chunk       = inputs.prefill_chunk;
    impl->pipeline_stage_first = inputs.pipeline_stage_first;
    impl->pipeline_stage_last  = inputs.pipeline_stage_last;
    impl->pipeline_import_pinned = inputs.pipeline_import_pinned;
    impl->pipeline_boundary_columns = inputs.pipeline_boundary_columns;
    impl->draft_window        = inputs.draft_window;
    impl->speculative_backend = inputs.speculative_backend;
    impl->proposal_head       = inputs.proposal_head;
    impl->features            = inputs.features;
    impl->use_cuda_graph      = inputs.use_cuda_graph;
    impl->device              = inputs.device;
    impl->kv_dtype            = inputs.kv_dtype;
    impl->kv_quant_group      = inputs.kv_quant_group;
    impl->kv_skip_layers      = inputs.kv_skip_layers;
    impl->rewrite_checkpoints = inputs.rewrite_checkpoints;
    impl->elastic_kv          = inputs.elastic_kv;
    impl->elastic_kv_overcommit = inputs.elastic_kv_overcommit;
    impl->persistent          = persistent_layout(*impl);
    impl->workspace           = build_workspace_plan(*impl);
    if (impl->features.vision) {
        constexpr std::uint32_t kFrontendMergedLimit = 32768;
        const std::uint32_t merged = std::min(impl->capacity, kFrontendMergedLimit);
        // The planner has no artifact in hand, so it reserves for the tower this target
        // compiles -- but at the text width the checkpoint declared, since that is what the
        // merger writes and what every per-request transient will be measured against.
        const family::VisionGeometry vision = schedule::bound_vision_geometry(
            schedule::compiled_vision_geometry(), impl->geometry);
        impl->request_transient_capacity_bytes =
            schedule::VisionContext::output_transient_bytes(vision, merged);
    }
    if (impl->use_cuda_graph) {
        // Definitions remain per execution profile, but only one executable is instantiated for
        // each reachable node-topology class. These bounds cover the largest profile installed in
        // each class and the driver/module state materialized while qualifying all definitions.
        if (impl->speculative_backend == SpeculativeBackend::None) {
            impl->graph_allowance_bytes =
                checked_mul(ordinary_graph_allowance_per_lane_bytes<Variant>(),
                            impl->max_concurrency, "ordinary exact-b graph allowance");
        } else if (impl->speculative_backend == SpeculativeBackend::Mtp) {
            const auto profiles = mtp_graph_profiles(impl->capacity, impl->draft_window);
            const std::size_t per_batch_allowance = graph_topology_allowance(
                profiles,
                [&](GraphExecutionProfile profile) {
                    const std::uint64_t final_visible = std::min<std::uint64_t>(
                        impl->capacity,
                        static_cast<std::uint64_t>(profile.max) + 2ULL * impl->draft_window);
                    return (final_visible <= 4096 ? 12ULL : 82ULL) * kMiB;
                },
                "MTP graph allowance");
            impl->graph_allowance_bytes = checked_mul(per_batch_allowance, impl->max_concurrency,
                                                      "MTP exact-b graph allowance");
        } else {
            const auto class_allowance = [&](std::uint32_t batch_size) {
                const auto profiles =
                    dflash_graph_profiles(impl->capacity, impl->draft_window, batch_size);
                return graph_topology_allowance(
                    profiles,
                    [&](GraphExecutionProfile profile) {
                        const std::uint64_t final_visible = std::min<std::uint64_t>(
                            impl->capacity,
                            static_cast<std::uint64_t>(profile.max) + impl->draft_window + 1ULL);
                        return (final_visible <= 4096 ? 64ULL : 96ULL) * kMiB;
                    },
                    "DFlash graph allowance");
            };
            for (std::uint32_t batch_size = 1; batch_size <= impl->max_concurrency; ++batch_size) {
                impl->graph_allowance_bytes =
                    checked_add(impl->graph_allowance_bytes, class_allowance(batch_size),
                                "DFlash exact-b graph allowance");
            }
        }
    }

    impl->device_reservation_bytes = checked_add(
        checked_add(
            checked_add(checked_add(impl->persistent.bytes, impl->persistent.elastic_plane_bytes,
                                    "sequence memory plan"),
                        impl->workspace.capacity, "sequence memory plan"),
            impl->request_transient_capacity_bytes, "request transient reservation"),
        impl->graph_allowance_bytes, "sequence graph allowance");
    return impl;
}

} // namespace

std::unique_ptr<family::detail::SequencePlannerImpl<Variant>>
make_sequence_planner_impl(DeviceContext& device, const EngineOptions& options,
                           WeightsProfile weights_profile,
                           const family::TextGeometry& geometry) {
    validate_target_options(device, options);
    const KvCacheStorage kv_storage = resolve_kv_storage(options.kv_cache, geometry);
    if (kv_storage == KvCacheStorage::Fp8E4M3 &&
        options.speculative.backend == SpeculativeBackend::DFlash) {
        // DFlash commits its draft through kv_cache_append_prefix, which has no e4m3
        // path. Refuse the pair at startup: the alternative is an exception thrown
        // mid-round once a draft first lands.
        throw std::invalid_argument(
            "--spec dflash needs a bf16 KV cache (pass --kv-cache-dtype bf16); its draft commit "
            "has no fp8 path");
    }

    SequencePlanningInputs inputs{
        .weights_profile     = weights_profile,
        .geometry            = geometry,
        .capacity            = options.max_context,
        .max_concurrency     = options.max_concurrency,
        .prefill_chunk       = std::min(options.prefill_chunk, options.max_context),
        .draft_window        = options.speculative.draft_tokens,
        .speculative_backend = options.speculative.backend,
        .kv_dtype       = kv_storage_dtype(kv_storage),
        .kv_quant_group = kv_storage == KvCacheStorage::Int8Group64 ? family::kKvQuantGroup : 0,
        .kv_skip_layers = options.kv_cache_skip_layers,
        .rewrite_checkpoints = options.rewrite_checkpoints,
        .elastic_kv     = options.elastic_kv,
        .elastic_kv_overcommit = options.elastic_kv_overcommit,
        .proposal_head  = options.speculative.proposal_head,
        .features       = family::startup_features(options),
        .use_cuda_graph = options.use_cuda_graph,
        .device         = options.device,
        .pipeline_stage_first = options.pipeline_stage_first,
        .pipeline_stage_last = options.pipeline_stage_last,
        .pipeline_import_pinned = options.pipeline_import_pinned,
        .pipeline_boundary_columns = options.pipeline_boundary_columns,
    };
    const std::uint32_t logical_pages = page_count(inputs.capacity);
    const std::uint32_t minimum_pages = std::max(logical_pages, inputs.max_concurrency);
    const std::uint64_t maximum_pages64 =
        static_cast<std::uint64_t>(inputs.max_concurrency) * logical_pages;
    if (maximum_pages64 > std::numeric_limits<std::uint32_t>::max()) {
        throw std::overflow_error("maximum Main KV page count exceeds uint32");
    }
    const auto maximum_pages = static_cast<std::uint32_t>(maximum_pages64);

    auto planner     = std::make_unique<family::detail::SequencePlannerImpl<Variant>>();
    planner->inputs  = inputs;
    planner->minimum = build_sequence_candidate(inputs, minimum_pages);
    planner->curve   = runtime::SequenceCapacityCurve{
          .main_page_tokens                     = static_cast<std::uint32_t>(kPagedKVPageSize),
          .minimum_main_page_groups             = minimum_pages,
          .maximum_main_page_groups             = maximum_pages,
          .minimum_device_reservation_bytes     = planner->minimum->device_reservation_bytes,
          .bytes_per_additional_main_page_group = 0,
    };
    if (minimum_pages < maximum_pages) {
        auto adjacent = build_sequence_candidate(inputs, minimum_pages + 1U);
        if (adjacent->device_reservation_bytes <= planner->minimum->device_reservation_bytes) {
            throw std::logic_error("Qwen3.6 sequence layout has a nonpositive KV capacity stride");
        }
        planner->curve.bytes_per_additional_main_page_group =
            adjacent->device_reservation_bytes - planner->minimum->device_reservation_bytes;
    }
    return planner;
}

std::unique_ptr<SequencePlanImpl>
finalize_sequence_plan_impl(std::unique_ptr<family::detail::SequencePlannerImpl<Variant>> planner,
                            std::uint32_t main_page_groups) {
    if (planner == nullptr || planner->minimum == nullptr) {
        throw std::invalid_argument("Qwen3.6 sequence planner is empty");
    }
    const std::size_t expected = planner->curve.reservation_bytes(main_page_groups);
    std::unique_ptr<SequencePlanImpl> plan;
    if (main_page_groups == planner->curve.minimum_main_page_groups) {
        plan = std::move(planner->minimum);
    } else {
        plan = build_sequence_candidate(planner->inputs, main_page_groups);
    }
    if (plan->device_reservation_bytes != expected) {
        throw std::logic_error(
            "Qwen3.6 physical sequence layout is not affine in Main KV page capacity");
    }
    return plan;
}

} // namespace sinfer::family::detail::SINFER_FAMILY_RUNTIME_NS
