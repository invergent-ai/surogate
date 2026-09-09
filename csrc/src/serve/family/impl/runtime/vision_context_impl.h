#include "family/impl/runtime/instance.h"
#include "family/impl/runtime/vision_context.h"

#include "core/device.h"
#include "core/layout.h"
#include <api/family/vision_control.h>
#include "api/ops/add_bias.h"
#include "api/ops/gelu.h"
#include "api/ops/layer_norm.h"
#include "api/ops/linear.h"
#include "api/ops/residual_add.h"
#include "api/ops/rope.h"
#include "api/ops/vision_attention.h"
#include "api/ops/vision_pos_embed.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>

namespace sinfer::family::detail::SINFER_FAMILY_RUNTIME_NS::schedule {

family::VisionContext vision_context_for(DeviceContext& device, const LoadedModelData& model) {
    if (!model.vision) {
        throw std::invalid_argument("Vision execution was requested without materialized weights");
    }
    return family::VisionContext(device, *model.vision, model.vision_geometry, model.geometry,
                                 [](const char* tag, const Tensor& tensor, std::int32_t layers,
                                    cudaStream_t stream) {
                                     debug_probe<Variant>(tag, tensor, layers, stream);
                                 });
}


VisionPrefillSession::VisionPrefillSession(DeviceContext& device, const LoadedModelData& model,
                                           WorkspaceArena& workspace,
                                           family::PreparedPromptData& prompt,
                                           const VisionPrefillPlan& plan,
                                           runtime::TransientRegion transient)
    : device_(device), workspace_(workspace), prompt_(prompt), plan_(plan), transient_(transient),
      context_(vision_context_for(device, model)) {
    if (plan_.control == nullptr || plan_.control->items.empty() || plan_.uses.empty()) {
        throw std::invalid_argument("Vision prefill plan has no suffix item spans");
    }
    if (transient_.data == nullptr || transient_.alignment < kWorkspaceAlignment) {
        throw std::invalid_argument("Vision item output transient is missing or misaligned");
    }
    encoded_payloads_pending_release_.reserve(plan_.uses.size());
    timers_.reserve(plan_.uses.size());
}

VisionChunk VisionPrefillSession::prepare_chunk(std::uint32_t begin, std::uint32_t nominal_length) {
    if (nominal_length == 0 || begin >= prompt_.token_ids.size()) {
        throw std::invalid_argument("Vision chunk range is empty or outside the prompt");
    }
    const std::uint64_t nominal_end64 =
        static_cast<std::uint64_t>(begin) + static_cast<std::uint64_t>(nominal_length);
    std::uint32_t end = static_cast<std::uint32_t>(
        std::min<std::uint64_t>(nominal_end64, prompt_.token_ids.size()));

    const VisionUseSpan* active = nullptr;
    for (const VisionUseSpan& use : plan_.uses) {
        if (use.end <= begin) { continue; }
        if (use.begin >= end) { break; }
        if (active == nullptr) {
            active = &use;
        } else {
            end = std::min(end, use.begin);
            break;
        }
    }
    if (end <= begin) { throw std::logic_error("Vision chunk cap made no forward progress"); }
    if (active == nullptr) {
        return VisionChunk{static_cast<std::int32_t>(end - begin), nullptr, {}};
    }
    if (active->item_index >= plan_.control->items.size() ||
        active->item_index >= prompt_.vision_items.size() ||
        active->item_index >= prompt_.media_payloads.size()) {
        throw std::logic_error("Vision prefill item index is out of range");
    }
    const family::VisionItemControl& control = plan_.control->items[active->item_index];
    const family::VisionItem& source         = prompt_.vision_items[active->item_index];
    if (source.modality != control.modality || source.grid.temporal != control.grid.temporal ||
        source.grid.height != control.grid.height || source.grid.width != control.grid.width ||
        source.patch_begin != control.patch_begin || source.patch_count != control.patch_count) {
        throw std::invalid_argument("Vision prefill plan does not describe the prepared item");
    }
    if (control.merged_count > static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max())) {
        throw std::overflow_error("Vision item output columns exceed int32");
    }
    const family::VisionGeometry& g = context_.geometry();
    const std::size_t output_bytes = VisionContext::output_transient_bytes(g, control.merged_count);
    if (output_bytes > transient_.size) {
        throw std::invalid_argument("Vision item output transient is too small");
    }
    Tensor output(transient_.data, DType::BF16,
                  {g.output_hidden, static_cast<std::int32_t>(control.merged_count), 1 + g.deepstack_layers});

    if (!active_item_ || *active_item_ != active->item_index) {
        if (active_item_ && active->item_index <= *active_item_) {
            throw std::logic_error("Vision items are not consumed in strictly increasing order");
        }
        const std::size_t patch_elements =
            checked_mul(control.patch_count, static_cast<std::size_t>(g.patch_dim),
                        "item patch elements");
        const auto& payload = prompt_.media_payloads[active->item_index];
        if (!payload || payload->patch_elements != patch_elements) {
            throw std::invalid_argument("Vision item patch payload has an invalid shape");
        }
        timers_.emplace_back(device_);
        timers_.back().start();
        context_.encode(VisionItemView{payload->span(), &control}, output, workspace_);
        timers_.back().record_stop();
        workspace_.reset();
        active_item_ = active->item_index;
        encoded_payloads_pending_release_.push_back(active->item_index);
    }
    return VisionChunk{static_cast<std::int32_t>(end - begin), &control, output.slice(2, 0, 1),
                       g.deepstack_layers ? output.slice(2, 1, g.deepstack_layers) : Tensor{}};
}

void VisionPrefillSession::release_encoded_media_payloads() noexcept {
    for (const std::uint32_t item_index : encoded_payloads_pending_release_) {
        if (item_index >= prompt_.media_payloads.size()) { std::terminate(); }
        prompt_.media_payloads[item_index].reset();
    }
    encoded_payloads_pending_release_.clear();
}

double VisionPrefillSession::elapsed_seconds() const {
    double milliseconds = 0.0;
    for (const CudaEventTimer& timer : timers_) { milliseconds += timer.elapsed_ms(); }
    return milliseconds / 1000.0;
}


} // namespace sinfer::family::detail::SINFER_FAMILY_RUNTIME_NS::schedule
