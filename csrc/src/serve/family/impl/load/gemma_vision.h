#pragma once
#include "api/family/vision.h"
#include "api/family/text_geometry.h"
#include "artifact/typed_binding.h"

namespace sinfer::family {
struct GemmaVisionPlan {
    struct TensorPlan {
        artifact::ObjectHandle object;
        artifact::NumericFormat format;
        std::vector<std::uint64_t> shape;
    };

    VisionGeometry vision_geometry;

    struct LinearPlan {
        artifact::LinearBinding binding;
        std::int32_t rows, columns;
    };

    std::map<std::string, LinearPlan> vision_linears;
    std::map<std::string, TensorPlan> vision_tensors;
};

inline void bind_gemma_vision(artifact::Binder& binder, GemmaVisionPlan& out,
                              const TextGeometry& text, bool enabled) {
    if (!binder.has("vision/patch_embedding")) {
        if (enabled) {
            throw std::invalid_argument("this Gemma checkpoint contains no vision weights");
        }
        return;
    }
    out.vision_geometry = VisionGeometry::resolved(binder.reader().vision_geometry());
    const auto& g       = out.vision_geometry;
    if (!g.gemma_version || g.output_hidden != text.hidden || g.gemma_pad_token < 0 ||
        (text.output_rows > 0 && g.gemma_pad_token >= text.output_rows)) {
        throw std::invalid_argument("Gemma vision geometry disagrees with the decoder");
    }
    using artifact::NumericFormat;
    const auto place =
        enabled ? artifact::TensorPlacement::Device : artifact::TensorPlacement::ValidateOnly;
    const auto linear = [&](const std::string& name, int rows, int columns) {
        out.vision_linears[name] = {
            artifact::bind_linear(binder, "vision/" + name, rows, columns, place), rows, columns};
    };
    const auto tensor = [&](const std::string& name, std::initializer_list<std::uint64_t> shape,
                            NumericFormat format = NumericFormat::BF16) {
        out.vision_tensors[name] = {
            artifact::bind_tensor(binder, "vision/" + name, format, shape, place), format, shape};
    };
    const auto h = static_cast<std::uint64_t>(g.hidden);
    const auto m = static_cast<std::uint64_t>(g.intermediate);
    linear("patch_embedding", g.hidden, g.patch_dim);
    tensor("position_embedding",
           {static_cast<std::uint64_t>(g.position_embeddings) * (g.gemma_version == 4 ? 2 : 1), h});
    linear("projection", text.hidden, g.hidden);
    if (g.encoder_free) {
        tensor("patch_embedding_bias", {h});
        for (const auto& [name, width] : std::vector<std::pair<std::string, std::uint64_t>>{
                 {"patch_norm1", static_cast<std::uint64_t>(g.patch_dim)},
                 {"patch_norm2", h},
                 {"position_norm", h}}) {
            tensor(name + "/weight", {width});
            tensor(name + "/bias", {width});
        }
        return;
    }
    if (g.gemma_version == 3) {
        tensor("patch_embedding_bias", {h});
        tensor("post_norm/weight", {h});
        tensor("post_norm/bias", {h});
        tensor("projection_norm", {h});
    } else if (g.standardize) {
        tensor("std_bias", {h}, NumericFormat::FP32);
        tensor("std_scale", {h}, NumericFormat::FP32);
    }
    for (int layer = 0; layer < g.layers; ++layer) {
        const auto p = "layers/" + std::to_string(layer) + "/";
        if (g.gemma_version == 3) {
            linear(p + "attention/qkv", 3 * g.hidden, g.hidden);
            tensor(p + "attention/qkv_bias", {3 * h});
            linear(p + "attention/output", g.hidden, g.hidden);
            tensor(p + "attention/output_bias", {h});
            linear(p + "mlp/fc1", g.intermediate, g.hidden);
            tensor(p + "mlp/fc1_bias", {m});
            linear(p + "mlp/fc2", g.hidden, g.intermediate);
            tensor(p + "mlp/fc2_bias", {h});
            for (const auto* norm : {"norm1", "norm2"}) {
                tensor(p + norm + "/weight", {h});
                tensor(p + norm + "/bias", {h});
            }
        } else {
            for (const auto* name : {"attention/query", "attention/key", "attention/value",
                                     "attention/output", "mlp/gate", "mlp/up", "mlp/down"}) {
                const std::string role(name);
                linear(p + role, role == "mlp/gate" || role == "mlp/up" ? g.intermediate : g.hidden,
                       role == "mlp/down" ? g.intermediate : g.hidden);
                if (g.clipped_linears) { tensor(p + role + "/clip", {4}, NumericFormat::FP32); }
            }
            for (const auto* name : {"input_norm", "post_attention_norm", "pre_feedforward_norm",
                                     "post_feedforward_norm"}) {
                tensor(p + name, {h});
            }
            tensor(p + "attention/query_norm", {h / g.heads});
            tensor(p + "attention/key_norm", {h / g.heads});
        }
    }
}

inline void bind_muse_vision(artifact::Binder& binder, GemmaVisionPlan& out,
                             const TextGeometry& text, bool enabled) {
    if (!binder.has("vision/patch_embedding")) {
        if (enabled) { throw std::invalid_argument("Muse-Glimmer image input requires its vision projector (--mmproj)"); }
        return;
    }
    out.vision_geometry = VisionGeometry::resolved(binder.reader().vision_geometry());
    const auto& g = out.vision_geometry;
    if (!g.muse_glimmer || g.output_hidden != text.hidden) {
        throw std::invalid_argument("Muse-Glimmer vision geometry disagrees with the decoder");
    }
    const auto place = enabled ? artifact::TensorPlacement::Device : artifact::TensorPlacement::ValidateOnly;
    const auto linear = [&](const std::string& name, int rows, int cols) {
        out.vision_linears[name] = {artifact::bind_linear(binder, "vision/" + name, rows, cols, place), rows, cols};
    };
    const auto tensor = [&](const std::string& name, std::initializer_list<std::uint64_t> shape) {
        out.vision_tensors[name] = {artifact::bind_tensor(binder, "vision/" + name,
            artifact::NumericFormat::BF16, shape, place), artifact::NumericFormat::BF16, shape};
    };
    const auto h = static_cast<std::uint64_t>(g.hidden);
    linear("patch_embedding", g.hidden, g.patch_dim);
    tensor("position_embedding", {static_cast<std::uint64_t>(g.position_embeddings), h});
    for (const auto* name : {"pre_norm", "post_norm"}) {
        tensor(std::string(name) + "/weight", {h});
        tensor(std::string(name) + "/bias", {h});
    }
    linear("projector/0", g.projector_hidden, g.merger_hidden());
    linear("projector/1", g.projector_hidden, g.projector_hidden);
    linear("projector/2", g.output_hidden, g.projector_hidden);
    for (int i = 0; i < g.layers; ++i) {
        const auto p = "layers/" + std::to_string(i) + "/";
        for (const auto* name : {"attention/query", "attention/key", "attention/value",
                                 "attention/output", "mlp/fc1", "mlp/fc2"}) {
            const std::string role(name);
            const int rows = role == "mlp/fc1" ? g.intermediate : g.hidden;
            linear(p + role, rows, role == "mlp/fc2" ? g.intermediate : g.hidden);
            tensor(p + role + "_bias", {static_cast<std::uint64_t>(rows)});
        }
        for (const auto* name : {"norm1", "norm2"}) {
            tensor(p + name + "/weight", {h});
            tensor(p + name + "/bias", {h});
        }
    }
}

inline VisionWeights materialize_gemma_vision(const artifact::MaterializedArtifact& backing,
                                              const GemmaVisionPlan& plan) {
    VisionWeights result;
    for (const auto& [name, binding] : plan.vision_linears) {
        result.extra_linears[name] =
            artifact::materialized_linear(backing, binding.binding, binding.rows, binding.columns);
    }
    for (const auto& [name, tensor] : plan.vision_tensors) {
        if (tensor.shape.size() == 1) {
            result.extra_tensors[name] =
                artifact::materialized_tensor(backing, tensor.object, tensor.format,
                                              {static_cast<std::int32_t>(tensor.shape[0])});
        } else {
            result.extra_tensors[name] =
                artifact::materialized_tensor(backing, tensor.object, tensor.format,
                                              {static_cast<std::int32_t>(tensor.shape[1]),
                                               static_cast<std::int32_t>(tensor.shape[0])});
        }
    }
    return result;
}
} // namespace sinfer::family
