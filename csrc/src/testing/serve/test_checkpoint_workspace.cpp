#include "targets/qwen3_5/impl/variant.h"
#include "targets/qwen3_5_moe/impl/variant.h"
#include "targets/llama/impl/variant.h"
#include "targets/qwen3/impl/variant.h"
#include "api/targets/llama/package.h"
#include "api/targets/qwen3/package.h"
#include "artifact/reader.h"
#include "api/ops/attn_input_proj.h"
#include "api/ops/gdn_input_proj.h"
#include "api/ops/linear.h"
#include "api/ops/linear_swiglu.h"
#include "api/ops/sparse_moe.h"

#include <cassert>
#include <cstdlib>

using namespace sinfer;
using Dense = targets::qwen3_5::detail::Variant;
using Moe = targets::qwen3_5_moe::detail::Variant;

int main() {
    family::TextGeometry g;
    g.hidden = 512; g.layers = 4; g.intermediate = 512;
    g.gdn_key_heads = g.gdn_value_heads = 2;
    g.gdn_key_head_dim = g.gdn_value_head_dim = 64;
    g.gdn_conv_kernel = 4;
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
    // The family may also reserve the materialized adapter route beside this leaf.
    assert(Dense::gdn_input_projection_snapshot_workspace_capacity_bytes(g, Dense::WeightsProfile::GroupwiseInt, phase, 2, 1, 4) >= snapshot);
    assert(Moe::gdn_input_projection_snapshot_workspace_capacity_bytes(g, Moe::WeightsProfile::GroupwiseInt, phase, 2, 1, 4) >= snapshot);
    // A missing descriptor must not silently fall back to the profile's preset.
    g.linear_storage.erase("text/layers/2/attention/query_key_gate_value");
    bool refused = false;
    try { (void)Dense::attention_projection_workspace_capacity_bytes(g, Dense::WeightsProfile::Fp8Block, phase, 1, 64); }
    catch (const std::invalid_argument&) { refused = true; }
    assert(refused);

    // The artifact's gate/up formats determine the base path, while a server
    // admitting adapters must retain the original adapter-safe reservation.
    for (int layer = 0; layer < g.layers; ++layer) {
        put(layer, "mlp/gate_up", 2 * g.intermediate, g.hidden, QType::Q5_K);
        put(layer, "mlp/down", g.hidden, g.intermediate, QType::Q6_K);
    }
    const auto mlp = [&](bool lora) {
        return Dense::post_mixer_workspace_capacity_bytes(
            g, Dense::WeightsProfile::GroupwiseInt, phase, 1, 2048, lora);
    };
    const auto base = mlp(false), adapted = mlp(true);
    assert(adapted == Dense::post_mixer_workspace_capacity_bytes(
        g, Dense::WeightsProfile::GroupwiseInt, phase, 1, 2048));
    assert(base + 4ULL * g.intermediate * 2048 <= adapted);
    const auto fused = ops::linear_swiglu_workspace_capacity_bytes(
        QType::Q5_K, QType::Q5_K, 2 * g.intermediate, g.hidden,
        ops::LinearPolicy::A16Only, 1, 2048);
    const auto conservative = ops::linear_swiglu_workspace_capacity_bytes(
        QType::Q5_K, 2 * g.intermediate, g.hidden, ops::LinearPolicy::A16Only, 1, 2048);
    assert(fused + 4ULL * g.intermediate * 2048 <= conservative);
    // A segment requiring the general path must keep its intermediate planes.
    g.linear_storage["text/layers/2/mlp/gate_up"].formats.push_back(QType::F16);
    const auto mixed = mlp(false);
    assert(mixed >= base + 4ULL * g.intermediate * 2048);
    assert(mlp(true) >= mixed);

    const auto check_dense = [&]<class Variant>() {
        const auto profile = Variant::WeightsProfile::GroupwiseInt;
        const auto capacity = [&](bool lora) {
            return Variant::post_mixer_workspace_capacity_bytes(g, profile, phase, 1, 2048, lora);
        };
        for (auto type : {QType::Q8_0, QType::IQ4_NL, QType::Q5_K}) {
            for (int layer = 0; layer < g.layers; ++layer) {
                put(layer, "mlp/gate_up", 2 * g.intermediate, g.hidden, type);
            }
            const auto fused_base = capacity(false), adapters = capacity(true);
            assert(fused_base + 4ULL * g.intermediate * 2048 <= adapters);
            assert(adapters <= Variant::post_mixer_workspace_capacity_bytes(g, profile, phase, 1, 2048));
            auto& formats = g.linear_storage["text/layers/2/mlp/gate_up"].formats;
            formats.push_back(QType::F16);
            assert(capacity(false) >= fused_base + 4ULL * g.intermediate * 2048);
            assert(capacity(true) >= capacity(false));
        }
        // Qwen3 may store separate matrices; their temporary outputs remain necessary.
        const auto fused_base = [&] {
            for (int layer = 0; layer < g.layers; ++layer) {
                put(layer, "mlp/gate_up", 2 * g.intermediate, g.hidden, QType::Q8_0);
            }
            return capacity(false);
        }();
        const auto fused_adapters = capacity(true);
        put(2, "mlp/gate", g.intermediate, g.hidden, QType::Q8_0);
        put(2, "mlp/up", g.intermediate, g.hidden, QType::IQ4_NL);
        assert(capacity(false) >= fused_base + 4ULL * g.intermediate * 2048);
        assert(capacity(true) >= std::max(capacity(false), fused_adapters));
        g.linear_storage.erase("text/layers/2/mlp/gate");
        g.linear_storage.erase("text/layers/2/mlp/up");
        // Partial metadata must not silently replace a missing layer with a preset.
        g.linear_storage.erase("text/layers/2/mlp/gate_up");
        bool refused = false;
        try { (void)capacity(false); } catch (const std::invalid_argument&) { refused = true; }
        assert(refused);
    };
    check_dense.template operator()<targets::llama::detail::Variant>();
    check_dense.template operator()<targets::qwen3::detail::Variant>();

    // Startup asks the package for geometry before materializing its weights.
    // The package must expose the same storage facts as the later binding plan.
    if (const auto* path = std::getenv("SUROGATE_CHECKPOINT_WORKSPACE_ARTIFACT")) {
        const artifact::Reader reader(path);
        const auto declared = reader.identity().architecture == "llama"
            ? targets::llama::Package::declared_geometry(reader)
            : targets::qwen3::Package::declared_geometry(reader);
        assert(!declared.linear_storage.empty());
        assert(declared.linear_storage.contains("text/layers/0/mlp/gate_up") ||
               declared.linear_storage.contains("text/layers/0/mlp/gate"));
    }
}
