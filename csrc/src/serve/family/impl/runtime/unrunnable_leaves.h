#pragma once

// The leaves a text-only, unspeculated target cannot run.
//
// The family runtime is a template over the whole Variant interface, so a target
// with no linear-attention mixer and no draft head still has to define the GDN
// and MTP leaves. Each one throws rather than returning quietly: a silent no-op
// would be a layer that contributed nothing to the residual, which reads as a
// model that merely answers badly.
//
// The definitions were identical in every such target -- 110 lines apiece, three
// copies, differing in nothing -- so adding a leaf to the interface meant the
// same edit in each, with the compiler pointing at three separate files. This
// macro is that block once. The two arguments are the target's own refusal
// helpers, because what is worth saying differs per target: a Llama says every
// one of its 22 layers is full attention, which a Gemma 3 cannot say.
//
// Expand it inside the target's own namespace, after `Variant` is complete.

#include "family/impl/runtime/target_support.h"

// clang-format off
#define SINFER_FAMILY_UNRUNNABLE_LEAVES(NO_LINEAR, NO_SPEC) \
    void Variant::gdn_input_projection(const Tensor&, const GdnProjectionWeights&, Tensor&, Tensor&, \
                                       family::TextPhase, WorkspaceArena&, cudaStream_t) { \
        NO_LINEAR("gdn_input_projection"); \
    } \
 \
    void Variant::gdn_input_projection_snapshot(const Tensor&, const GdnProjectionWeights&, \
                                                const Tensor&, Tensor&, const Tensor&, const Tensor&, \
                                                const Tensor&, Tensor&, Tensor&, Tensor&, Tensor&, \
                                                family::TextPhase, WorkspaceArena&, cudaStream_t) { \
        NO_LINEAR("gdn_input_projection_snapshot"); \
    } \
 \
    void Variant::gdn_input_projection_record(const Tensor&, const GdnProjectionWeights&, const Tensor&, \
                                              const Tensor&, const Tensor&, const Tensor&, Tensor&, \
                                              Tensor&, Tensor&, Tensor&, Tensor&, family::TextPhase, \
                                              WorkspaceArena&, cudaStream_t) { \
        NO_LINEAR("gdn_input_projection_record"); \
    } \
 \
    void Variant::gdn_output_projection(const Tensor&, const Weight&, Tensor&, family::TextPhase, \
                                        WorkspaceArena&, cudaStream_t) { \
        NO_LINEAR("gdn_output_projection"); \
    } \
 \
    void Variant::gdn_norm_control_projection(const Tensor&, const Tensor&, float, \
                                              const GdnProjectionWeights&, Tensor&, Tensor&, Tensor&, \
                                              WorkspaceArena&, cudaStream_t) { \
        NO_LINEAR("gdn_norm_control_projection"); \
    } \
 \
    std::size_t Variant::gdn_input_projection_workspace_capacity_bytes(WeightsProfile, \
                                                                       family::TextPhase, \
                                                                       std::int32_t, std::int32_t) { \
        return 0; \
    } \
 \
    std::size_t Variant::gdn_input_projection_snapshot_workspace_capacity_bytes(WeightsProfile, \
                                                                                family::TextPhase, \
                                                                                std::int32_t, \
                                                                                std::int32_t, \
                                                                                std::int32_t) { \
        return 0; \
    } \
 \
    std::size_t Variant::gdn_input_projection_record_workspace_capacity_bytes(WeightsProfile, \
                                                                              family::TextPhase, \
                                                                              std::int32_t, \
                                                                              std::int32_t, \
                                                                              std::int32_t) { \
        return 0; \
    } \
 \
    std::size_t Variant::gdn_output_projection_workspace_capacity_bytes(WeightsProfile, \
                                                                        family::TextPhase, \
                                                                        std::int32_t, std::int32_t) { \
        return 0; \
    } \
 \
    std::size_t Variant::gdn_norm_control_projection_workspace_capacity_bytes(std::int32_t, \
                                                                              std::int32_t) { \
        return 0; \
    } \
 \
    void Variant::mtp_attention_projection(const Tensor&, const MtpAttentionProjectionWeights&, Tensor&, \
                                           Tensor&, Tensor&, Tensor&, WorkspaceArena&, cudaStream_t) { \
        NO_SPEC("mtp_attention_projection"); \
    } \
 \
    void Variant::mtp_kv_projection(const Tensor&, const MtpAttentionProjectionWeights&, Tensor&, \
                                    Tensor&, WorkspaceArena&, cudaStream_t) { \
        NO_SPEC("mtp_kv_projection"); \
    } \
 \
    void Variant::mtp_q_gate_projection(const Tensor&, const MtpAttentionProjectionWeights&, Tensor&, \
                                        Tensor&, WorkspaceArena&, cudaStream_t) { \
        NO_SPEC("mtp_q_gate_projection"); \
    } \
 \
    void Variant::mtp_post_mixer(const Tensor&, const MtpPostMixerWeights&, Tensor&, WorkspaceArena&, \
                                 cudaStream_t) { \
        NO_SPEC("mtp_post_mixer"); \
    } \
 \
    std::size_t Variant::mtp_attention_projection_workspace_capacity_bytes(std::int32_t, std::int32_t) { \
        return 0; \
    } \
 \
    std::size_t Variant::mtp_kv_projection_workspace_capacity_bytes(std::int32_t, std::int32_t) { \
        return 0; \
    } \
 \
    std::size_t Variant::mtp_q_gate_projection_workspace_capacity_bytes(std::int32_t, std::int32_t) { \
        return 0; \
    } \
 \
    std::size_t Variant::mtp_post_mixer_workspace_capacity_bytes(const family::TextGeometry&, \
                                                                 std::int32_t, std::int32_t) { \
        return 0; \
    } \
 \
 \
    // Parity probe. SUROGATE_SERVE_DUMP_RESIDUAL=<dir> writes each tagged attention \
    // intermediate of the first forward as raw BF16 behind a 16-byte header \
    // {magic, rows, columns, occurrence}. Layers run in order, so the occurrence \
    // count of a tag is its layer index -- which keeps the family's probe signature \
    // (tag, tensor, stream) unchanged. Synchronises the stream, so it is only ever \
    // on for parity work. \
    namespace { \
 \
 \
    } // namespace
// clang-format on
