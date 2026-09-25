// Completed-prefix storage for the model-owned radix index. GPU execution slots remain
// fixed; snapshots use bounded host memory and restore into the admitted slot's pages.
#include "family/impl/runtime/target_support.h"
#include "family/impl/runtime/program.h"

namespace sinfer::family::detail::SINFER_FAMILY_RUNTIME_NS {

std::vector<Tensor> ProgramImplCore::prefix_state_tensors(const SequenceState& sequence,
                                                         bool checkpoint, bool draft) const {
    const auto slot = checkpoint
        ? LinearStateSlots::rewrite_checkpoint_state_slot(sequence.lane, max_concurrency)
        : LinearStateSlots::current_state_slot(sequence.lane, max_concurrency);
    std::vector<Tensor> tensors;
    const int first = pipeline_stage() ? stage.first : 0;
    const int last = pipeline_stage() ? stage.last : cfg.layers;
    std::uint32_t linear = 0;
    for (int layer = 0; layer < cfg.layers; ++layer) {
        if (cfg.layer_attends(layer)) { continue; }
        if (layer >= first && layer < last) {
            tensors.push_back(decoder->linear_attention.conv_slot(linear, slot));
            if (decoder->linear_attention.spec.has_recurrent()) {
                tensors.push_back(decoder->linear_attention.recurrent_slot(linear, slot));
            }
        }
        ++linear;
    }
    if (!decoder->ple.empty() && model.geometry.ple_layer >= first && model.geometry.ple_layer < last) {
        tensors.push_back(decoder->ple.history_slot(slot));
        tensors.push_back(decoder->ple.conv_slot(slot));
    }
    const Tensor hidden = checkpoint ? sequence.rewrite_checkpoint_hidden : sequence.tail_hidden;
    if (hidden.data) { tensors.push_back(hidden); }
    if (dflash && draft) {
        auto& local = checkpoint ? decoder->checkpoint_dflash(sequence.lane) : dflash->local;
        for (std::uint32_t layer = 0; layer < local.layer_count(); ++layer) {
            const auto view = local.layer_view(layer);
            const auto lane = checkpoint ? 0 : static_cast<std::int32_t>(sequence.lane);
            tensors.push_back(view.k.slice(3, lane, 1));
            tensors.push_back(view.v.slice(3, lane, 1));
        }
        if (!checkpoint) {
            tensors.push_back(dflash->pending_features.slice(2, sequence.lane, 1));
        }
    }
    return tensors;
}

namespace {
std::size_t tensor_image_bytes(std::span<const Tensor> tensors) {
    std::size_t bytes = 0;
    for (const auto& t : tensors) { bytes += t.bytes(); }
    return bytes;
}

std::vector<std::vector<std::byte>> download_prefix_state(std::span<const Tensor> tensors,
                                                         cudaStream_t stream) {
    std::vector<std::vector<std::byte>> image;
    image.reserve(tensors.size());
    for (const auto& tensor : tensors) {
        if (!tensor.is_contiguous()) { throw std::logic_error("noncontiguous prefix state"); }
        image.emplace_back(tensor.bytes());
        CUDA_CHECK(cudaMemcpyAsync(image.back().data(), tensor.data, tensor.bytes(),
                                   cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }
    return image;
}

void upload_prefix_state(const std::vector<std::vector<std::byte>>& image,
                         std::span<const Tensor> tensors, cudaStream_t stream) {
    if (image.size() != tensors.size()) { throw std::logic_error("prefix state layout changed"); }
    for (std::size_t i = 0; i < tensors.size(); ++i) {
        if (!tensors[i].is_contiguous() || image[i].size() != tensors[i].bytes()) {
            throw std::logic_error("prefix state geometry changed");
        }
    }
    for (std::size_t i = 0; i < tensors.size(); ++i) {
        CUDA_CHECK(cudaMemcpyAsync(tensors[i].data, image[i].data(), image[i].size(),
                                   cudaMemcpyHostToDevice, stream));
    }
}
} // namespace

void ProgramImplCore::archive_sequence(const SequenceState& sequence) {
    if (sequence.target_only || !sequence.retained || !sequence.cacheable || !sequence.kv || sequence.ledger.empty()) { return; }
    try {
        std::vector<std::uint32_t> boundaries;
        const auto append = reusable_append_frontier(sequence);
        if (append) { boundaries.push_back(append); }
        const bool checkpoint = sequence.rewrite_checkpoint.valid && decoder->has_checkpoint(sequence.lane);
        if (checkpoint && sequence.rewrite_checkpoint.frontier) {
            boundaries.push_back(sequence.rewrite_checkpoint.frontier);
        }
        if (boundaries.empty()) { return; }
        const auto current = prefix_state_tensors(sequence, false);
        const auto saved = checkpoint ? prefix_state_tensors(sequence, true) : std::vector<Tensor>{};
        std::size_t bytes = tensor_image_bytes(current) + tensor_image_bytes(saved) + sizeof(ArchivedSequence);
        bytes += sequence.kv->text.mapped_page_count() * decoder->text_kv.pool().occupancy().page_bytes;
        if (sequence.kv->backend) {
            bytes += sequence.kv->backend->mapped_page_count() * backend_kv_cache()->pool().occupancy().page_bytes;
        }
        // Ledger, three position axes, token types, radix keys at both boundaries, and scores.
        bytes += sequence.ledger.size() * (9 * sizeof(TokenId) + 1);
        bytes += sequence.cached_scores.size() * sizeof(TokenScore);
        for (const auto& score : sequence.cached_scores) { bytes += score.top.size() * sizeof(TokenLogprob); }
        if (!archived_prefixes.reserve(bytes)) { return; }
        auto image = std::make_shared<ArchivedSequence>();
        image->state.copy_metadata(sequence);
        if (!checkpoint) { image->state.rewrite_checkpoint = {}; }
        image->current = download_prefix_state(current, device.stream);
        if (checkpoint) { image->checkpoint = download_prefix_state(saved, device.stream); }
        image->text = decoder->text_kv.pool().download_pages(sequence.kv->text.page_ids(), device.stream);
        if (sequence.kv->backend) {
            image->backend = backend_kv_cache()->pool().download_pages(sequence.kv->backend->page_ids(), device.stream);
        }
        (void)archived_prefixes.insert(image->state.ledger, boundaries, image, bytes);
    } catch (const std::bad_alloc&) {
        // Retention is opportunistic. An unavailable host allocation must not
        // take down a healthy engine; the next request can prefill normally.
    }
}

void ProgramImplCore::restore_archived_sequence(SequenceState& sequence, const RequestPlanImpl& plan) {
    const auto& image = *plan.archived;
    clear_lane(sequence, requests[sequence.lane]);
    sequence.copy_metadata(image.state);
    reserve_sequence_kv(sequence, plan.text_kv_page_entitlement, plan.backend_kv_page_entitlement);
    const auto backend_tokens = speculative_backend == SpeculativeBackend::Mtp
        ? plan.reuse_base - 1 : speculative_backend == SpeculativeBackend::DFlash ? plan.reuse_base : 0;
    materialize_sequence_kv(sequence, plan.reuse_base, backend_tokens);
    decoder->text_kv.pool().upload_pages(image.text, sequence.kv->text.page_ids(), device.stream);
    if (sequence.kv->backend) {
        backend_kv_cache()->pool().upload_pages(image.backend, sequence.kv->backend->page_ids(), device.stream);
    }
    upload_prefix_state(image.current, prefix_state_tensors(sequence, false), device.stream);
    if (sequence.rewrite_checkpoint.valid) {
        if (decoder->try_acquire_checkpoint(sequence.lane)) {
            upload_prefix_state(image.checkpoint, prefix_state_tensors(sequence, true), device.stream);
        } else {
            sequence.rewrite_checkpoint = {};
            if (is_rewrite_checkpoint_restore(plan.reuse)) {
                throw std::logic_error("admitted prefix checkpoint lost its state slot");
            }
        }
    }
    device.synchronize();
    ++checkpoint_revisions[sequence.lane];
}


void ProgramImplCore::prune_gpu_prefixes() {
    // All mutation and destruction happens on the engine worker, between rounds.
    for (auto it = gpu_prefixes.begin(); it != gpu_prefixes.end();) {
        if (it->second->key.expired()) it = gpu_prefixes.erase(it);
        else ++it;
    }
}

void ProgramImplCore::capture_gpu_prefix(SequenceState& sequence,
                                        const std::shared_ptr<const GpuPrefixKey>& key,
                                        const std::shared_ptr<GpuPrefixStorage>& storage) {
    if (!key || !sequence.kv || !sequence.tail_hidden_valid) {
        throw std::logic_error("GPU prefix capture has no completed frontier");
    }
    if (gpu_prefixes.contains(key.get())) { throw std::invalid_argument("GPU prefix key was already captured"); }
    auto image = std::make_shared<GpuPrefix>();
    image->key = key;
    image->state.copy_metadata(sequence);
    image->state.rewrite_checkpoint = {};
    const auto tensors = prefix_state_tensors(sequence, false, false);
    if (!storage || storage->free.empty()) throw std::logic_error("unreserved GPU prefix state");
    image->storage = storage;
    image->slot = storage->free.back();
    storage->free.pop_back();
    DeviceArena frame(DeviceSpan{static_cast<std::byte*>(storage->memory.base()) + image->slot * storage->stride,
                                 storage->stride});
    for (const auto& tensor : tensors) {
        if (!tensor.is_contiguous()) throw std::logic_error("noncontiguous GPU prefix state");
        auto memory = frame.alloc_bytes(tensor.bytes());
        auto copy = tensor;
        copy.data = memory.data;
        image->current.push_back(copy);
        CUDA_CHECK(cudaMemcpyAsync(copy.data, tensor.data, tensor.bytes(), cudaMemcpyDeviceToDevice, device.stream));
    }
    device.synchronize();
    image->pages = std::make_shared<SequenceKVBundle>(std::move(*sequence.kv));
    if (take_injected_fault(InjectedFault::PoisonGpuPrefix)) {
        // Test fault: bf16 0xFFFF is NaN, so every question read on this prefix is non-finite.
        // Only the pages this prefix owns: pages it borrowed from a parent prefix are read by
        // that parent's other forks too.
        const auto& text = image->pages->text;
        const auto pages = text.page_ids().subspan(text.borrowed_pages());
        if (!pages.empty()) { decoder->text_kv.pool().zero_pages(pages, device.stream, 0xFF); }
    }
    sequence.kv.reset();
    sequence.retained = false;
    gpu_prefixes.emplace(key.get(), std::move(image));
}

void ProgramImplCore::restore_gpu_prefix(SequenceState& sequence, const RequestPlanImpl& plan) {
    const auto image = plan.device_prefix;
    clear_lane(sequence, requests[sequence.lane]);
    sequence.copy_metadata(image->state);
    SequenceKVBundle bundle;
    auto owner = std::shared_ptr<const PagedKVAllocation>(image->pages, &image->pages->text);
    bundle.text = decoder->text_kv.pool().fork_prefix(std::move(owner), plan.reuse_base,
                                                     plan.text_kv_page_entitlement, device.stream);
    sequence.kv.emplace(std::move(bundle));
    const auto tensors = prefix_state_tensors(sequence, false, false);
    if (tensors.size() != image->current.size()) throw std::logic_error("GPU prefix state layout changed");
    for (std::size_t i = 0; i < tensors.size(); ++i) {
        if (!tensors[i].is_contiguous() || tensors[i].bytes() != image->current[i].bytes())
            throw std::logic_error("GPU prefix state geometry changed");
        CUDA_CHECK(cudaMemcpyAsync(tensors[i].data, image->current[i].data, tensors[i].bytes(),
                                   cudaMemcpyDeviceToDevice, device.stream));
    }
    ++checkpoint_revisions[sequence.lane];
}

} // namespace sinfer::family::detail::SINFER_FAMILY_RUNTIME_NS
