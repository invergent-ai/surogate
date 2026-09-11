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

std::uint32_t VisionPrefillSession::chunk_length(std::uint32_t begin, std::uint32_t nominal_length) const {
    if (nominal_length == 0 || begin >= prompt_.token_ids.size()) { return 0; }
    auto end = static_cast<std::uint32_t>(std::min<std::uint64_t>(
        std::uint64_t(begin) + nominal_length, prompt_.token_ids.size()));
    bool active = false;
    for (const auto& use : plan_.uses) {
        if (use.end <= begin) { continue; }
        if (use.begin >= end) { break; }
        if (!active) {
            if (context_.geometry().attention_mode && begin < use.begin) { end = use.begin; break; }
            if (context_.geometry().attention_mode && end < use.end) { return 0; }
            active = true;
        } else { end = std::min(end, use.begin); break; }
    }
    return end - begin;
}

const VisionUseSpan* VisionPrefillSession::use_for_chunk(std::uint32_t begin,
                                                        std::uint32_t nominal_length) const {
    const auto length = chunk_length(begin, nominal_length);
    if (length == 0) { throw std::invalid_argument("Vision chunk cannot make progress"); }
    for (const auto& use : plan_.uses) {
        if (use.end <= begin) { continue; }
        if (use.begin >= begin + length) { break; }
        return &use;
    }
    return nullptr;
}

bool VisionPrefillSession::chunk_ready(std::uint32_t begin, std::uint32_t nominal_length) const {
    const auto* use = use_for_chunk(begin, nominal_length);
    return use == nullptr || (active_item_ && *active_item_ == use->item_index);
}

bool VisionPrefillSession::needs_text_slicing(std::uint32_t begin, std::uint32_t count) const {
    return context_.geometry().attention_mode && use_for_chunk(begin, count) != nullptr;
}

ImageTextPrefillState& VisionPrefillSession::text_state(std::uint32_t begin, const Tensor& residual) {
    if (!active_item_ || encoding_item_) { throw std::logic_error("image text block precedes its encoder"); }
    if (!text_ || text_->begin != begin) {
        const auto offset = VisionContext::output_transient_bytes(
            context_.geometry(), plan_.control->items[*active_item_].merged_count);
        if (offset > transient_.size || residual.bytes() > transient_.size - offset) {
            throw std::logic_error("image text residual exceeds its transient allocation");
        }
        text_ = ImageTextPrefillState{.begin = begin,
            .residual = Tensor(static_cast<std::byte*>(transient_.data) + offset, residual.dtype,
                               {residual.ne[0], residual.ne[1]})};
    }
    if (text_->residual.bytes() != residual.bytes() || text_->residual.ne[1] != residual.ne[1]) {
        throw std::logic_error("image text block changed shape while suspended");
    }
    return *text_;
}

VisionChunk VisionPrefillSession::prepare_chunk(std::uint32_t begin, std::uint32_t nominal_length) {
    // The serving scheduler prepares the tower incrementally before entering
    // this text chunk. Keep the synchronous form for standalone/bridge callers.
    while (!chunk_ready(begin, nominal_length)) { (void)advance_encoding(begin, nominal_length); }
    const auto* active = use_for_chunk(begin, nominal_length);
    const auto length = static_cast<std::int32_t>(chunk_length(begin, nominal_length));
    if (!active) { return VisionChunk{length, nullptr, {}}; }
    const auto& control = plan_.control->items[active->item_index];
    const auto& g = context_.geometry();
    Tensor output(transient_.data, DType::BF16,
                  {g.output_hidden, static_cast<std::int32_t>(control.merged_count), 1 + g.deepstack_layers});
    return VisionChunk{length, &control, output.slice(2, 0, 1),
                       g.deepstack_layers ? output.slice(2, 1, g.deepstack_layers) : Tensor{}};
}

bool VisionPrefillSession::advance_encoding(std::uint32_t begin, std::uint32_t nominal_length) {
    const auto* active = use_for_chunk(begin, nominal_length);
    if (!active || (active_item_ && *active_item_ == active->item_index)) { return true; }
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
    if (VisionContext::encoding_transient_bytes(g, control.merged_count) > transient_.size) {
        throw std::invalid_argument("Vision item output transient is too small");
    }
    Tensor output(transient_.data, DType::BF16,
                  {g.output_hidden, static_cast<std::int32_t>(control.merged_count), 1 + g.deepstack_layers});

    if (!encoding_item_) {
        if (active_item_ && active->item_index <= *active_item_) {
            throw std::logic_error("Vision items are not consumed in strictly increasing order");
        }
        encoding_item_ = active->item_index;
        encoding_ = family::VisionEncodeState{};
        encoding_.residual = Tensor(static_cast<std::byte*>(transient_.data) + output_bytes,
                                    DType::BF16, {g.hidden, static_cast<std::int32_t>(control.patch_count)});
    } else if (*encoding_item_ != active->item_index) {
        throw std::logic_error("Vision encoding changed items before completion");
    }
    const auto patch_elements = checked_mul(control.patch_count, static_cast<std::size_t>(g.patch_dim),
                                             "item patch elements");
    const auto& payload = prompt_.media_payloads[active->item_index];
    if (!payload || payload->patch_elements != patch_elements) {
        throw std::invalid_argument("Vision item patch payload has an invalid shape");
    }
    timers_.emplace_back(device_);
    timers_.back().start();
    bool complete = false;
    for (std::size_t step = 0; step < family::kVisionEncodeStepsPerSlice && !complete; ++step) {
        complete = context_.encode_step(VisionItemView{payload->span(), &control}, output,
                                        workspace_, encoding_);
    }
    timers_.back().record_stop();
    workspace_.reset();
    if (complete) {
        active_item_ = active->item_index;
        encoding_item_.reset();
        encoded_payloads_pending_release_.push_back(active->item_index);
    }
    return complete;
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
