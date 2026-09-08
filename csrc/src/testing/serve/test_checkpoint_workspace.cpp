#include "targets/qwen3_5/impl/variant.h"
#include "targets/qwen3_5_moe/impl/variant.h"
#include "api/ops/attn_input_proj.h"
#include "api/ops/gdn_input_proj.h"
#include "api/ops/linear.h"
#include "api/ops/sparse_moe.h"

#include <cassert>

using namespace sinfer;
using Dense = targets::qwen3_5::detail::Variant;
using Moe = targets::qwen3_5_moe::detail::Variant;

int main() {
    family::TextGeometry g;
    g.hidden = 512; g.layers = 4; g.intermediate = 512;
    g.declare_attention_layer(0); g.declare_attention_layer(2);
    const auto put = [&](int layer, const char* role, int n, int k, QType type) {
        g.linear_storage["text/layers/" + std::to_string(layer) + "/" + role] = {n, k, {type}};
    };
    for (int layer : {0, 2}) {
        put(layer, "attention/query_key_gate_value", 768, 512, QType::BF16_CTRL);
    }
    constexpr auto phase = family::TextPhase::Prefill;
    assert(Dense::attention_projection_workspace_capacity_bytes(g, Dense::WeightsProfile::Fp8Block, phase, 1, 64) == 0);
    put(2, "attention/query_key_gate_value", 768, 512, QType::FP8_E4M3FN_BLK128_F32S);
    const auto expected = ops::attn_input_proj_workspace_capacity_bytes(QType::FP8_E4M3FN_BLK128_F32S,
        768, 512, ops::LinearPolicy::AllowA8, 1, 64);
    assert(expected > 0);
    for (auto profile : {Dense::WeightsProfile::GroupwiseInt, Dense::WeightsProfile::Nvfp4All,
                          Dense::WeightsProfile::Nvfp4MlpOnly, Dense::WeightsProfile::Fp8Block}) {
        assert(Dense::attention_projection_workspace_capacity_bytes(g, profile, phase, 1, 64) == expected);
    }
    for (int layer : {1, 3}) {
        put(layer, "gdn/query_key_value", 384, 512, QType::BF16_CTRL);
        put(layer, "gdn/z", 128, 512, QType::BF16_CTRL);
    }
    put(3, "gdn/z", 128, 512, QType::NVFP4);
    const auto split = ops::gdn_input_proj_split_workspace_capacity_bytes(QType::BF16_CTRL, QType::NVFP4,
        384, 128, 512, ops::LinearPolicy::A16Only, ops::LinearPolicy::AllowA4, 1, 64);
    assert(split > 0);
    assert(Dense::gdn_input_projection_workspace_capacity_bytes(g, Dense::WeightsProfile::Fp8Block, phase, 1, 64) == split);
    assert(Moe::gdn_input_projection_workspace_capacity_bytes(g, Moe::WeightsProfile::GroupwiseInt, phase, 1, 64) == split);
    const auto snapshot = ops::gdn_input_proj_conv_snapshot_split_workspace_capacity_bytes(QType::BF16_CTRL, QType::NVFP4,
        384, 128, 512, ops::LinearPolicy::A16Only, ops::LinearPolicy::AllowA4, 2, 1, 4);
    assert(Dense::gdn_input_projection_snapshot_workspace_capacity_bytes(g, Dense::WeightsProfile::GroupwiseInt, phase, 2, 1, 4) == snapshot);
    assert(Moe::gdn_input_projection_snapshot_workspace_capacity_bytes(g, Moe::WeightsProfile::GroupwiseInt, phase, 2, 1, 4) == snapshot);
    // A missing descriptor must not silently fall back to the profile's preset.
    g.linear_storage.erase("text/layers/2/attention/query_key_gate_value");
    bool refused = false;
    try { (void)Dense::attention_projection_workspace_capacity_bytes(g, Dense::WeightsProfile::Fp8Block, phase, 1, 64); }
    catch (const std::invalid_argument&) { refused = true; }
    assert(refused);
}
