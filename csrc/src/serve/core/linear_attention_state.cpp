#include "core/linear_attention_state.h"

#include "core/device.h"

#include <limits>
#include <stdexcept>
#include <string>

namespace sinfer {
namespace {

constexpr std::size_t kArenaAlign = 256;

void validate_positive(std::int32_t value, const char* message) {
    if (value <= 0) { throw std::invalid_argument(message); }
}

void validate_layer_slot(const LinearAttentionStatePool& pool, std::uint32_t layer,
                         std::int32_t slot, const char* label) {
    if (layer >= pool.layer_count()) {
        throw std::out_of_range(std::string(label) + " layer out of range");
    }
    if (slot < 0 || slot >= pool.slot_count()) {
        throw std::out_of_range(std::string(label) + " slot out of range");
    }
}

void validate_state_tensor(const Tensor& tensor, DType dtype,
                           std::initializer_list<std::int32_t> shape, const char* label) {
    const Tensor expected(nullptr, dtype, shape);
    if (tensor.data == nullptr || tensor.dtype != dtype || !tensor.is_contiguous() ||
        tensor.bytes() != expected.bytes()) {
        throw std::logic_error(std::string("LinearAttentionStatePool ") + label +
                               " tensor is inconsistent");
    }
    for (int dim = 0; dim < 4; ++dim) {
        if (tensor.ne[dim] != expected.ne[dim]) {
            throw std::logic_error(std::string("LinearAttentionStatePool ") + label +
                                   " shape is inconsistent");
        }
    }
}

std::int64_t layer_stride_bytes(const std::vector<Tensor>& tensors, const char* label) {
    if (tensors.empty()) {
        throw std::logic_error(std::string("LinearAttentionStatePool has no ") + label + " layers");
    }
    if (tensors.size() == 1) { return 0; }

    const auto first  = reinterpret_cast<std::uintptr_t>(tensors[0].data);
    const auto second = reinterpret_cast<std::uintptr_t>(tensors[1].data);
    if (second <= first ||
        second - first > static_cast<std::uintptr_t>(std::numeric_limits<std::int64_t>::max())) {
        throw std::logic_error(std::string("LinearAttentionStatePool ") + label +
                               " layer stride is invalid");
    }
    const auto stride = static_cast<std::int64_t>(second - first);
    if (static_cast<std::uint64_t>(stride) < tensors.front().bytes()) {
        throw std::logic_error(std::string("LinearAttentionStatePool ") + label +
                               " layers overlap");
    }
    for (std::size_t layer = 2; layer < tensors.size(); ++layer) {
        const auto previous = reinterpret_cast<std::uintptr_t>(tensors[layer - 1].data);
        const auto current  = reinterpret_cast<std::uintptr_t>(tensors[layer].data);
        if (current <= previous || current - previous != static_cast<std::uintptr_t>(stride)) {
            throw std::logic_error(std::string("LinearAttentionStatePool ") + label +
                                   " layer stride is not constant");
        }
    }
    return stride;
}

} // namespace

LinearAttentionStatePoolLayout
plan_linear_attention_state_pool(LayoutBuilder& builder, const LinearAttentionStatePoolSpec& spec) {
    if (spec.layers == 0) {
        // A pure-attention target has no linear mixer. That is a real model
        // shape, not a misconfiguration: the pool is empty, occupies nothing,
        // and every accessor below refuses because nothing may ask it for a
        // layer that does not exist.
        LinearAttentionStatePoolLayout empty;
        empty.spec = spec;
        return empty;
    }
    if (spec.layers > static_cast<std::uint32_t>(std::numeric_limits<std::int32_t>::max())) {
        throw std::overflow_error("LinearAttentionStatePool layer count exceeds int32");
    }
    validate_positive(spec.conv_channels,
                      "LinearAttentionStatePool conv_channels must be positive");
    validate_positive(spec.conv_width, "LinearAttentionStatePool conv_width must be positive");
    // All three recurrent dimensions or none: a mixer either carries a recurrent state or it
    // does not, and a spec that names two of the three has lost one of them somewhere rather
    // than describing a model.
    if (spec.has_recurrent()) {
        validate_positive(spec.value_head_dim,
                          "LinearAttentionStatePool value_head_dim must be positive");
        validate_positive(spec.key_head_dim,
                          "LinearAttentionStatePool key_head_dim must be positive");
    } else if (spec.value_head_dim != 0 || spec.key_head_dim != 0) {
        throw std::invalid_argument("LinearAttentionStatePool declares recurrent head dimensions "
                                    "without any heads to carry them");
    }
    validate_positive(spec.slot_count, "LinearAttentionStatePool slot_count must be positive");
    if (spec.conv_dtype != DType::BF16 && spec.conv_dtype != DType::FP32) {
        throw std::invalid_argument("LinearAttentionStatePool conv_dtype must be BF16 or FP32");
    }

    const Tensor conv_shape(nullptr, spec.conv_dtype,
                            {spec.conv_channels, spec.conv_width, spec.slot_count});
    // Built only where there is one to build: a Tensor of zero extents is not a description of
    // an empty region, it is a shape that does not exist, and constructing one throws.
    const Tensor recurrent_shape =
        spec.has_recurrent() ? Tensor(nullptr, DType::BF16,
                                      {spec.key_head_dim, spec.value_head_dim, spec.value_heads,
                                       spec.slot_count})
                             : Tensor{};

    LinearAttentionStatePoolLayout layout;
    layout.spec = spec;
    layout.conv.reserve(spec.layers);
    if (spec.has_recurrent()) { layout.recurrent.reserve(spec.layers); }
    for (std::uint32_t layer = 0; layer < spec.layers; ++layer) {
        const std::string prefix = "Linear Attention layer " + std::to_string(layer);
        layout.conv.push_back(builder.add(conv_shape.bytes(), kArenaAlign, prefix + " conv"));
        if (spec.has_recurrent()) {
            layout.recurrent.push_back(
                builder.add(recurrent_shape.bytes(), kArenaAlign, prefix + " recurrent"));
        }
    }
    return layout;
}

LinearAttentionStatePool::LinearAttentionStatePool(DeviceSpan backing,
                                                   const LinearAttentionStatePoolLayout& layout)
    : spec(layout.spec) {
    if (spec.layers == 0) {
        if (!layout.conv.empty() || !layout.recurrent.empty()) {
            throw std::invalid_argument(
                "LinearAttentionStatePool declares no layers but carries state");
        }
        return; // pure-attention target: nothing to map
    }
    const std::size_t expected_recurrent = spec.has_recurrent() ? layout.conv.size() : 0U;
    if (layout.conv.empty() || layout.recurrent.size() != expected_recurrent ||
        layout.conv.size() != spec.layers) {
        throw std::invalid_argument(
            "LinearAttentionStatePool layout layer counts are inconsistent");
    }

    const Tensor conv_shape(nullptr, spec.conv_dtype,
                            {spec.conv_channels, spec.conv_width, spec.slot_count});
    const Tensor recurrent_shape =
        spec.has_recurrent() ? Tensor(nullptr, DType::BF16,
                                      {spec.key_head_dim, spec.value_head_dim, spec.value_heads,
                                       spec.slot_count})
                             : Tensor{};
    conv.reserve(layout.conv.size());
    recurrent.reserve(layout.recurrent.size());
    for (std::size_t layer = 0; layer < layout.conv.size(); ++layer) {
        if (layout.conv[layer].bytes != conv_shape.bytes()) {
            throw std::logic_error(
                "LinearAttentionStatePool layout tensor byte size is inconsistent");
        }
        conv.emplace_back(layout.conv[layer].bind(backing).data, spec.conv_dtype,
                          std::initializer_list<std::int32_t>{spec.conv_channels, spec.conv_width,
                                                              spec.slot_count});
        if (!spec.has_recurrent()) { continue; }
        if (layout.recurrent[layer].bytes != recurrent_shape.bytes()) {
            throw std::logic_error(
                "LinearAttentionStatePool layout tensor byte size is inconsistent");
        }
        recurrent.emplace_back(
            layout.recurrent[layer].bind(backing).data, DType::BF16,
            std::initializer_list<std::int32_t>{spec.key_head_dim, spec.value_head_dim,
                                                spec.value_heads, spec.slot_count});
    }
}

std::uint32_t LinearAttentionStatePool::layer_count() const noexcept {
    return static_cast<std::uint32_t>(conv.size());
}

std::int32_t LinearAttentionStatePool::slot_count() const noexcept { return spec.slot_count; }

std::int64_t LinearAttentionStatePool::conv_slot_stride_elements() const noexcept {
    return static_cast<std::int64_t>(spec.conv_channels) *
           static_cast<std::int64_t>(spec.conv_width);
}

std::int64_t LinearAttentionStatePool::recurrent_slot_stride_elements() const noexcept {
    if (!spec.has_recurrent()) { return 0; }
    return static_cast<std::int64_t>(spec.key_head_dim) *
           static_cast<std::int64_t>(spec.value_head_dim) *
           static_cast<std::int64_t>(spec.value_heads);
}

LinearAttentionStateAllLayersView LinearAttentionStatePool::all_layers_view() const {
    const std::size_t expected_recurrent = spec.has_recurrent() ? spec.layers : 0U;
    if (conv.size() != spec.layers || recurrent.size() != expected_recurrent || conv.empty()) {
        throw std::logic_error("LinearAttentionStatePool layer inventory is inconsistent");
    }
    for (std::size_t layer = 0; layer < conv.size(); ++layer) {
        validate_state_tensor(conv[layer], spec.conv_dtype,
                              {spec.conv_channels, spec.conv_width, spec.slot_count}, "conv");
        if (!spec.has_recurrent()) { continue; }
        validate_state_tensor(
            recurrent[layer], DType::BF16,
            {spec.key_head_dim, spec.value_head_dim, spec.value_heads, spec.slot_count},
            "recurrent");
    }
    return LinearAttentionStateAllLayersView{
        .conv_layer0                  = conv.front(),
        // A conv-only mixer has no recurrent image, and a null tensor is what says so. Every
        // reader of this view is a gated-delta kernel, which a short-conv target never reaches.
        .recurrent_layer0             = spec.has_recurrent() ? recurrent.front() : Tensor{},
        .conv_layer_stride_bytes      = layer_stride_bytes(conv, "conv"),
        .recurrent_layer_stride_bytes =
            spec.has_recurrent() ? layer_stride_bytes(recurrent, "recurrent") : 0,
        .spec                         = spec,
    };
}

Tensor LinearAttentionStatePool::conv_slot(std::uint32_t layer, std::int32_t slot) const {
    validate_layer_slot(*this, layer, slot, "LinearAttentionStatePool conv_slot");
    return conv.at(layer).slice(2, slot, 1).view({spec.conv_channels, spec.conv_width});
}

Tensor LinearAttentionStatePool::recurrent_slot(std::uint32_t layer, std::int32_t slot) const {
    if (!spec.has_recurrent()) {
        throw std::logic_error("LinearAttentionStatePool has no recurrent state to slot into");
    }
    validate_layer_slot(*this, layer, slot, "LinearAttentionStatePool recurrent_slot");
    return recurrent.at(layer)
        .slice(3, slot, 1)
        .view({spec.key_head_dim, spec.value_head_dim, spec.value_heads});
}

void LinearAttentionStatePool::copy_slot(std::int32_t src, std::int32_t dst, cudaStream_t stream) {
    // A pure-attention target holds an empty pool: there is no per-layer state
    // to move, and validating "layer 0" would refuse a layer that correctly
    // does not exist.
    if (layer_count() == 0) { return; }
    validate_layer_slot(*this, 0, src, "LinearAttentionStatePool copy_slot source");
    validate_layer_slot(*this, 0, dst, "LinearAttentionStatePool copy_slot destination");
    if (src == dst) { return; }
    for (std::uint32_t layer = 0; layer < layer_count(); ++layer) {
        const Tensor source      = conv_slot(layer, src);
        const Tensor destination = conv_slot(layer, dst);
        CUDA_CHECK(cudaMemcpyAsync(destination.data, source.data, source.bytes(),
                                   cudaMemcpyDeviceToDevice, stream));
    }
    if (!spec.has_recurrent()) { return; }
    for (std::uint32_t layer = 0; layer < layer_count(); ++layer) {
        const Tensor source      = recurrent_slot(layer, src);
        const Tensor destination = recurrent_slot(layer, dst);
        CUDA_CHECK(cudaMemcpyAsync(destination.data, source.data, source.bytes(),
                                   cudaMemcpyDeviceToDevice, stream));
    }
}

void LinearAttentionStatePool::zero_slot(std::int32_t slot, cudaStream_t stream) {
    if (layer_count() == 0) { return; } // empty pool: nothing to clear
    validate_layer_slot(*this, 0, slot, "LinearAttentionStatePool zero_slot");
    for (std::uint32_t layer = 0; layer < layer_count(); ++layer) {
        const Tensor state = conv_slot(layer, slot);
        CUDA_CHECK(cudaMemsetAsync(state.data, 0, state.bytes(), stream));
    }
    if (!spec.has_recurrent()) { return; }
    for (std::uint32_t layer = 0; layer < layer_count(); ++layer) {
        const Tensor state = recurrent_slot(layer, slot);
        CUDA_CHECK(cudaMemsetAsync(state.data, 0, state.bytes(), stream));
    }
}

} // namespace sinfer
