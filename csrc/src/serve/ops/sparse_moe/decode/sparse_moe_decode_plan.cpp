#include "ops/linear/ggml/ggml_dispatch.h"
#include "ops/sparse_moe/decode/sparse_moe_decode.h"

#include "core/layout.h"

#include <stdexcept>

namespace sinfer::ops::detail {

std::size_t sparse_moe_decode_workspace_bytes(const SparseMoeGeometry& geometry, std::int32_t tokens) {
    WorkspaceLayoutBuilder layout;
    (void)allocate_sparse_moe_decode_workspace(layout, geometry, tokens);
    return layout.peak_bytes(1);
}

SparseMoeDecodePlan resolve_sparse_moe_decode_plan(const SparseMoeGeometry& geometry,
                                                   QType routed_gate_up, QType routed_down) {
    const bool main_profile =
        routed_gate_up == QType::Q4G64_F16S &&
        (routed_down == QType::Q5G64_F16S || routed_down == QType::Q6G64_F16S);
    const bool w8_profile =
        routed_gate_up == QType::W8G32_F16S && routed_down == QType::W8G32_F16S;
    const bool nvfp4_profile = routed_gate_up == QType::NVFP4 && routed_down == QType::NVFP4;
    // A GGUF's routed experts stay in their own superblocks; gate/up and down carry the K-quant
    // the file chose for each, which is often not the same one.
    // Every GGML block format the codec decodes, which is the whole set the artifact can
    // store: the seam addresses a block and a lane within it, so a 32-value block is four
    // lanes where a superblock is thirty-two.
    const auto is_ggml_k = [](QType qtype) { return detail::ggml::is_ggml_qtype(qtype); };
    const bool ggml_k_profile = is_ggml_k(routed_gate_up) && is_ggml_k(routed_down);
    if (!main_profile && !w8_profile && !nvfp4_profile && !ggml_k_profile) {
        throw std::invalid_argument("sparse_moe: unsupported routed codec profile");
    }
    SparseMoeDecodePlan plan;
    plan.workspace_bytes = sparse_moe_decode_workspace_bytes(geometry);
    return plan;
}

} // namespace sinfer::ops::detail
