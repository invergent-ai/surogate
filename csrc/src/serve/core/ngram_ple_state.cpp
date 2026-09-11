#include "core/ngram_ple_state.h"

#include "core/device.h"

#include <cuda.h>

#include <stdexcept>
#include <string>

namespace sinfer {
namespace {

constexpr std::size_t kArenaAlign = 256;

void validate_slot(const NgramPleStatePool& pool, std::int32_t slot, const char* label) {
    if (pool.empty()) { throw std::logic_error(std::string(label) + " on an empty PLE pool"); }
    if (slot < 0 || slot >= pool.slot_count()) {
        throw std::out_of_range(std::string(label) + " slot out of range");
    }
}

} // namespace

NgramPleStatePoolLayout plan_ngram_ple_state_pool(LayoutBuilder& builder,
                                                  const NgramPleStatePoolSpec& spec) {
    if (spec.history_tokens <= 0 || spec.conv_history <= 0 || spec.channels <= 0 ||
        spec.slot_count <= 0) {
        throw std::invalid_argument("NgramPleStatePool spec must be positive");
    }
    const Tensor history_shape(nullptr, DType::I32, {spec.history_tokens, spec.slot_count});
    const Tensor conv_shape(nullptr, DType::BF16,
                            {spec.conv_history, spec.channels, spec.slot_count});
    NgramPleStatePoolLayout layout;
    layout.spec    = spec;
    layout.history = builder.add(history_shape.bytes(), kArenaAlign, "PLE token history");
    layout.conv    = builder.add(conv_shape.bytes(), kArenaAlign, "PLE conv history");
    return layout;
}

NgramPleStatePool::NgramPleStatePool(DeviceSpan backing, const NgramPleStatePoolLayout& layout)
    : history(layout.history.bind(backing).data, DType::I32,
              std::initializer_list<std::int32_t>{layout.spec.history_tokens,
                                                  layout.spec.slot_count}),
      conv_state(layout.conv.bind(backing).data, DType::BF16,
                 std::initializer_list<std::int32_t>{layout.spec.conv_history,
                                                     layout.spec.channels, layout.spec.slot_count}),
      spec(layout.spec) {}

Tensor NgramPleStatePool::history_slot(std::int32_t slot) const {
    validate_slot(*this, slot, "NgramPleStatePool history_slot");
    if (slot >= spec.slot_count) {
        const auto* checkpoint = checkpoint_slots.at(slot - spec.slot_count);
        if (!checkpoint) { throw std::logic_error("PLE checkpoint is not allocated"); }
        return checkpoint->history_slot(0);
    }
    return history.slice(1, slot, 1);
}

Tensor NgramPleStatePool::conv_slot(std::int32_t slot) const {
    validate_slot(*this, slot, "NgramPleStatePool conv_slot");
    if (slot >= spec.slot_count) {
        const auto* checkpoint = checkpoint_slots.at(slot - spec.slot_count);
        if (!checkpoint) { throw std::logic_error("PLE checkpoint is not allocated"); }
        return checkpoint->conv_slot(0);
    }
    return conv_state.slice(2, slot, 1);
}

void NgramPleStatePool::copy_slot(std::int32_t src, std::int32_t dst, cudaStream_t stream) {
    validate_slot(*this, src, "NgramPleStatePool copy_slot source");
    validate_slot(*this, dst, "NgramPleStatePool copy_slot destination");
    if (src == dst) { return; }
    const Tensor history_src = history_slot(src);
    const Tensor history_dst = history_slot(dst);
    CUDA_CHECK(cudaMemcpyAsync(history_dst.data, history_src.data, history_src.bytes(),
                               cudaMemcpyDeviceToDevice, stream));
    const Tensor conv_src = conv_slot(src);
    const Tensor conv_dst = conv_slot(dst);
    CUDA_CHECK(cudaMemcpyAsync(conv_dst.data, conv_src.data, conv_src.bytes(),
                               cudaMemcpyDeviceToDevice, stream));
}

void NgramPleStatePool::reset_slot(std::int32_t slot, cudaStream_t stream) {
    validate_slot(*this, slot, "NgramPleStatePool reset_slot");
    // A 32-bit fill (the driver API has one) keeps the reset capturable: no host memory is
    // read at execution time.
    const Tensor history_slot = this->history_slot(slot);
    const CUresult status =
        cuMemsetD32Async(reinterpret_cast<CUdeviceptr>(history_slot.data),
                         static_cast<unsigned int>(spec.eos_token),
                         static_cast<std::size_t>(spec.history_tokens), stream);
    if (status != CUDA_SUCCESS) {
        throw std::runtime_error("NgramPleStatePool reset_slot: cuMemsetD32Async failed");
    }
    const Tensor conv_slot = this->conv_slot(slot);
    CUDA_CHECK(cudaMemsetAsync(conv_slot.data, 0, conv_slot.bytes(), stream));
}

} // namespace sinfer
