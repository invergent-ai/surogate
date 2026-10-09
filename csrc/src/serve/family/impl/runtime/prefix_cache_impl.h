// Completed-prefix storage for the model-owned radix index. GPU execution slots remain
// fixed; snapshots use bounded host memory and restore into the admitted slot's pages.
#include "family/impl/runtime/target_support.h"
#include "family/impl/runtime/program.h"

namespace sinfer::family::detail::SINFER_FAMILY_RUNTIME_NS {

std::vector<Tensor> ProgramImplCore::prefix_state_tensors(const SequenceState& sequence,
                                                         bool checkpoint, bool draft,
                                                         bool with_hidden) const {
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
    if (hidden.data && with_hidden) { tensors.push_back(hidden); }
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

void upload_prefix_state(std::span<const std::span<std::byte>> image,
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

        // One block holds the whole image: every state tensor and KV plane at an aligned offset.
        const auto& text_pool = decoder->text_kv.pool();
        const auto text_ids = sequence.kv->text.page_ids();
        const PagedKVPool* backend_pool = sequence.kv->backend ? &backend_kv_cache()->pool() : nullptr;
        const auto backend_ids = sequence.kv->backend ? sequence.kv->backend->page_ids()
                                                      : std::span<const std::int32_t>{};
        std::size_t payload = 0;
        for (const auto& tensor : current) { payload += family::detail::archive_aligned(tensor.bytes()); }
        for (const auto& tensor : saved) { payload += family::detail::archive_aligned(tensor.bytes()); }
        for (std::size_t plane = 0; plane < text_pool.plane_count(); ++plane) {
            payload += family::detail::archive_aligned(text_pool.image_plane_bytes(plane, text_ids.size()));
        }
        for (std::size_t plane = 0; backend_pool && plane < backend_pool->plane_count(); ++plane) {
            payload += family::detail::archive_aligned(backend_pool->image_plane_bytes(plane, backend_ids.size()));
        }
        auto step_started = std::chrono::steady_clock::now();
        if (!archived_prefixes.reserve(bytes)) { return; }
        admit_trace_step("archive: make room", sequence.lane, step_started);

        // Evicting above may have returned the range this image fits in.
        step_started = std::chrono::steady_clock::now();
        if (!archive_arena && !archive_arena_refused) {
            archive_arena = family::detail::PinnedArchiveArena::create(kArchivedPrefixBytes + (16ULL << 20));
            archive_arena_refused = !archive_arena;
        }
        admit_trace_step("archive: create page-locked arena", sequence.lane, step_started);
        step_started = std::chrono::steady_clock::now();
        std::optional<family::detail::ArchiveBlock> pinned;
        if (archive_arena) { pinned = archive_arena->allocate(payload); }
        auto image = std::make_shared<ArchivedSequence>();
        image->storage = pinned ? std::move(*pinned) : family::detail::heap_archive_block(payload);
        if (admit_trace_enabled() && !image->storage.pinned) {
            std::fprintf(stderr, "admit-trace: lane %u archive of %zu bytes goes to the heap\n",
                         sequence.lane, payload);
        }
        admit_trace_step("archive: allocate image", sequence.lane, step_started);
        step_started = std::chrono::steady_clock::now();
        image->state.copy_metadata(sequence);
        if (!checkpoint) { image->state.rewrite_checkpoint = {}; }
        std::byte* cursor = image->storage.data;
        const auto take = [&](std::size_t size) {
            std::byte* at = cursor;
            cursor += family::detail::archive_aligned(size);
            return std::span<std::byte>(at, size);
        };
        const auto download = [&](std::span<const Tensor> tensors, std::vector<std::span<std::byte>>& out) {
            out.reserve(tensors.size());
            for (const auto& tensor : tensors) {
                if (!tensor.is_contiguous()) { throw std::logic_error("noncontiguous prefix state"); }
                out.push_back(take(tensor.bytes()));
                CUDA_CHECK(cudaMemcpyAsync(out.back().data(), tensor.data, tensor.bytes(),
                                           cudaMemcpyDeviceToHost, device.stream));
            }
        };
        download(current, image->current);
        if (checkpoint) { download(saved, image->checkpoint); }
        image->text_pages = static_cast<std::uint32_t>(text_ids.size());
        for (std::size_t plane = 0; plane < text_pool.plane_count(); ++plane) {
            image->text.push_back(take(text_pool.image_plane_bytes(plane, text_ids.size())).data());
        }
        text_pool.download_pages_to(text_ids, image->text, device.stream);
        if (backend_pool) {
            image->backend_pages = static_cast<std::uint32_t>(backend_ids.size());
            for (std::size_t plane = 0; plane < backend_pool->plane_count(); ++plane) {
                image->backend.push_back(take(backend_pool->image_plane_bytes(plane, backend_ids.size())).data());
            }
            backend_pool->download_pages_to(backend_ids, image->backend, device.stream);
        }
        // A page-locked image needs no wait: the stream orders its copies before the lane's next
        // round overwrites the state, and before any restore reads them. A heap image has been
        // copied by now; the fence makes sure of it before the block can be freed.
        if (!image->storage.pinned) { CUDA_CHECK(cudaStreamSynchronize(device.stream)); }
        admit_trace_step("archive: copy image", sequence.lane, step_started);
        step_started = std::chrono::steady_clock::now();
        (void)archived_prefixes.insert(image->state.ledger, boundaries, image, bytes);
        admit_trace_step("archive: index image", sequence.lane, step_started);
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
    const auto planes = [](const std::vector<std::byte*>& views) {
        return std::vector<const std::byte*>(views.begin(), views.end());
    };
    decoder->text_kv.pool().upload_pages_from(planes(image.text), image.text_pages,
                                             sequence.kv->text.page_ids(), device.stream);
    if (sequence.kv->backend) {
        backend_kv_cache()->pool().upload_pages_from(planes(image.backend), image.backend_pages,
                                                    sequence.kv->backend->page_ids(), device.stream);
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

std::shared_ptr<const std::vector<std::uint64_t>>
ProgramImplCore::shared_prefix_hashes(const PreparedPromptData& prompt, std::int32_t lora_slot) const {
    constexpr auto page = static_cast<std::size_t>(kPagedKVPageSize);
    // Whole pages before the prompt's last token: a fork must leave at least one token to
    // prefill, whose hidden the request samples from.
    const std::size_t pages = (prompt.token_ids.size() - 1) / page;
    if (pages == 0) { return nullptr; }
    auto hashes = std::make_shared<std::vector<std::uint64_t>>();
    hashes->reserve(pages);
    // Chained, so a page's hash names the whole prefix through it; seeded by the adapter,
    // whose KV is its own. A collision costs at most a capture: lookups compare the tokens.
    std::uint64_t h = 0xcbf29ce484222325ULL ^ (static_cast<std::uint64_t>(lora_slot + 2) << 32);
    for (std::size_t p = 0; p < pages; ++p) {
        for (std::size_t i = 0; i < page; ++i) {
            h ^= static_cast<std::uint32_t>(prompt.token_ids[p * page + i]);
            h *= 0x100000001b3ULL;
        }
        h ^= h >> 31;
        h *= 0x9e3779b97f4a7c15ULL;
        hashes->push_back(h);
    }
    return hashes;
}

std::shared_ptr<SharedPrefix> ProgramImplCore::find_shared_prefix(const PreparedPromptData& prompt,
    const std::vector<std::uint64_t>& hashes, std::int32_t lora_slot) const {
    std::shared_ptr<SharedPrefix> best;
    for (const auto& [hash, entry] : shared_prefixes) {
        const std::size_t pages = entry->tokens / kPagedKVPageSize;
        if (pages == 0 || pages > hashes.size() || hashes[pages - 1] != hash ||
            (best && best->tokens >= entry->tokens) ||
            !family::detail::prefix_matches(prompt, entry->state.ledger, entry->state.prefix_identity,
                                            entry->tokens, lora_slot)) {
            continue;
        }
        best = entry;
    }
    return best;
}

void ProgramImplCore::plan_shared_prefix_capture(RequestControl::Prefill& staged,
                                                 const std::vector<std::uint64_t>& hashes) {
    static const std::uint32_t min_tokens = [] {
        const char* raw = std::getenv("SUROGATE_SERVE_SHARED_PREFIX_MIN_TOKENS");
        return raw ? static_cast<std::uint32_t>(std::strtoul(raw, nullptr, 10)) : 1024U;
    }();
    // The deepest page an earlier request carried too: the prefix they have in common.
    std::size_t seen = 0;
    for (std::size_t k = hashes.size(); k > 0 && seen == 0; --k) {
        if (shared_prefix_seen[0].contains(hashes[k - 1]) || shared_prefix_seen[1].contains(hashes[k - 1])) {
            seen = k;
        }
    }
    constexpr std::size_t kSeenGeneration = 1U << 16;
    if (shared_prefix_seen[0].size() + hashes.size() > kSeenGeneration) {
        shared_prefix_seen[1] = std::move(shared_prefix_seen[0]);
        shared_prefix_seen[0].clear();
    }
    shared_prefix_seen[0].insert(hashes.begin(), hashes.end());
    const auto tokens = static_cast<std::uint32_t>(seen * kPagedKVPageSize);
    if (seen == 0 || tokens < std::max<std::uint32_t>(min_tokens, kPagedKVPageSize) ||
        tokens <= staged.cursor || tokens >= staged.prompt_tokens ||
        shared_prefixes.contains(hashes[seen - 1])) {
        return;
    }
    staged.shared_capture = tokens;
    staged.shared_capture_hash = hashes[seen - 1];
}

std::uint32_t ProgramImplCore::shared_prefix_pages() const noexcept {
    std::uint32_t pages = 0;
    for (const auto& item : shared_prefixes) { pages += item.second->pages->owned_entitlement(); }
    return pages;
}

void ProgramImplCore::drop_shared_prefixes() noexcept {
    if (shared_prefixes.empty()) { shared_prefix_storage.reset(); return; }
    // Pages a running or retained fork still borrows stay with it until it lets them go.
    shared_prefixes.clear();
    shared_prefix_storage.reset(); // its memory goes back once no plan holds an entry
    ++shared_prefix_revision;
}

void ProgramImplCore::capture_shared_prefix(SequenceState& sequence, RequestControl::Prefill& staged) {
    const std::uint32_t tokens = *staged.shared_capture;
    const std::uint64_t hash = staged.shared_capture_hash;
    staged.shared_capture.reset();
    if (shared_prefix_slots == 0) { return; }
    const auto pages = tokens / static_cast<std::uint32_t>(kPagedKVPageSize);
    if (!sequence.kv || sequence.kv->backend || sequence.text_kv_valid != tokens ||
        tokens % kPagedKVPageSize != 0 || pages <= sequence.kv->text.borrowed_pages() ||
        pages > sequence.kv->text.mapped_page_count() || shared_prefixes.contains(hash)) {
        return;
    }
    const std::uint32_t owned = pages - sequence.kv->text.borrowed_pages();
    if (owned > shared_prefix_page_budget) { return; }
    // Make room: the least useful entry goes first (never hit, then least recently used).
    while (!shared_prefixes.empty() && (shared_prefixes.size() >= shared_prefix_slots ||
                                        shared_prefix_pages() + owned > shared_prefix_page_budget)) {
        auto victim = shared_prefixes.begin();
        for (auto it = shared_prefixes.begin(); it != shared_prefixes.end(); ++it) {
            const bool cold = it->second->hits == 0, victim_cold = victim->second->hits == 0;
            if (cold != victim_cold ? cold : it->second->last_use < victim->second->last_use) { victim = it; }
        }
        shared_prefixes.erase(victim);
        ++shared_prefix_revision;
    }
    const auto tensors = prefix_state_tensors(sequence, false, false, false);
    if (!shared_prefix_storage) {
        std::size_t bytes = 0;
        for (const auto& t : tensors) { bytes = (bytes + 255) / 256 * 256 + t.bytes(); }
        bytes = (bytes + 255) / 256 * 256;
        try {
            shared_prefix_storage = std::make_shared<GpuPrefixStorage>(bytes, shared_prefix_slots);
        } catch (const std::exception&) {
            // Opportunistic like the host archive: no room for the states means no cache.
            (void)cudaGetLastError(); // the failed cudaMalloc must not fail a later launch check
            shared_prefix_slots = 0;
            return;
        }
    }
    // A plan that still holds an evicted entry keeps its slot until the plan is dropped.
    if (shared_prefix_storage->free.empty()) { return; }
    auto entry = std::make_shared<SharedPrefix>();
    entry->hash = hash;
    entry->tokens = tokens;
    entry->state.copy_metadata(sequence);
    entry->state.ledger.resize(tokens);
    entry->state.prefix_identity.truncate(tokens);
    entry->state.cached_scores.clear();
    entry->state.execution_frontier = tokens;
    entry->state.ledger_frontier = tokens;
    entry->state.text_kv_valid = tokens;
    entry->state.mtp_kv_valid = 0;
    entry->state.dflash_context_frontier = 0;
    entry->state.mtp_draft_count = 0;
    // No hidden is kept: a fork always has prompt left to prefill past the boundary, so the
    // append path never samples from it, but it needs the frontier marked complete.
    entry->state.tail_hidden_valid = true;
    entry->state.retained = true;
    entry->state.cacheable = true;
    entry->state.target_only = false;
    entry->state.rewrite_checkpoint = {};
    entry->storage = shared_prefix_storage;
    entry->slot = shared_prefix_storage->free.back();
    shared_prefix_storage->free.pop_back();
    DeviceArena frame(DeviceSpan{static_cast<std::byte*>(shared_prefix_storage->memory.base()) +
                                     entry->slot * shared_prefix_storage->stride,
                                 shared_prefix_storage->stride});
    for (const auto& tensor : tensors) {
        if (!tensor.is_contiguous()) { throw std::logic_error("noncontiguous shared prefix state"); }
        auto memory = frame.alloc_bytes(tensor.bytes());
        auto copy = tensor;
        copy.data = memory.data;
        entry->current.push_back(copy);
        CUDA_CHECK(cudaMemcpyAsync(copy.data, tensor.data, tensor.bytes(), cudaMemcpyDeviceToDevice,
                                   device.stream));
    }
    entry->pages = decoder->text_kv.pool().share_prefix(sequence.kv->text, pages);
    entry->last_use = ++shared_prefix_clock;
    static const bool trace = std::getenv("SUROGATE_SERVE_SHARED_PREFIX_TRACE") != nullptr;
    if (trace) {
        std::fprintf(stderr, "shared-prefix: captured %u tokens on lane %u (%zu entries, %u pages)\n",
                     tokens, sequence.lane, shared_prefixes.size() + 1, shared_prefix_pages() + owned);
    }
    shared_prefixes.emplace(hash, std::move(entry));
    ++shared_prefix_revision;
}

void ProgramImplCore::restore_shared_prefix(SequenceState& sequence, const RequestPlanImpl& plan) {
    SharedPrefix& entry = *plan.shared_prefix;
    clear_lane(sequence, requests[sequence.lane]);
    sequence.copy_metadata(entry.state);
    SequenceKVBundle bundle;
    bundle.text = decoder->text_kv.pool().fork_prefix(entry.pages, entry.tokens,
                                                     plan.text_kv_page_entitlement, device.stream);
    sequence.kv.emplace(std::move(bundle));
    const auto tensors = prefix_state_tensors(sequence, false, false, false);
    if (tensors.size() != entry.current.size()) { throw std::logic_error("shared prefix state layout changed"); }
    for (std::size_t i = 0; i < tensors.size(); ++i) {
        if (!tensors[i].is_contiguous() || tensors[i].bytes() != entry.current[i].bytes()) {
            throw std::logic_error("shared prefix state geometry changed");
        }
        CUDA_CHECK(cudaMemcpyAsync(tensors[i].data, entry.current[i].data, tensors[i].bytes(),
                                   cudaMemcpyDeviceToDevice, device.stream));
    }
    entry.last_use = ++shared_prefix_clock;
    ++entry.hits;
    ++checkpoint_revisions[sequence.lane];
}

} // namespace sinfer::family::detail::SINFER_FAMILY_RUNTIME_NS
