// The host prefix tier's program side (--prefix-cache-host-gb; objects and keys in
// api/kv_tier.h, memory in family/impl/kv_tier/host_pool.h).
//
// A finished conversation stays retained on its lane as before. At a round boundary with the
// copy stream free, the oldest retained lane without a current snapshot is copied to the host in
// the background: the 1,024-token chunks the pool does not already hold (a longer turn of the
// same conversation, or another conversation with the same opening, stored the rest), the pages
// past the last chunk, and the recurrent state at the conversation's frontiers. When the KV pool
// then needs the lane, eviction costs nothing; a lane without a snapshot is copied at that moment
// instead of dropped. A later turn that finds the snapshot is restored with host-to-device copies.
//
// Every copy runs on the tier's copy stream, after an event on the compute stream that orders it
// behind the work that wrote (or last read) those pages. Background copies hold up nothing: the
// compute stream waits for one only when the lane it reads is touched again first (a resumed or
// evicted lane, fence_lane_copies). A restore's copies are waited for before the lane's next GPU
// work, without a host synchronisation.
#include "family/impl/runtime/target_support.h"
#include "family/impl/runtime/program.h"

#include "api/ops/lora_store.h"

#include <algorithm>
#include <cstring>
#include <functional>

namespace sinfer::family::detail::SINFER_FAMILY_RUNTIME_NS {

namespace host_tier_detail {

enum Reason : int { kBackground = 0, kSpill = 1, kTakeover = 2, kRestore = 3 };

inline constexpr std::uint32_t kChunkPages = kvtier::kChunkTokens / static_cast<std::uint32_t>(kPagedKVPageSize);

[[nodiscard]] inline std::uint32_t pages_for(std::uint32_t tokens) noexcept {
    return (tokens + static_cast<std::uint32_t>(kPagedKVPageSize) - 1) / static_cast<std::uint32_t>(kPagedKVPageSize);
}

/// Copies logical pages [first, last) of `pool` (device ids `ids[first..last)`) to or from a host
/// image that starts at logical page `image_first` and holds `image_pages` pages per plane, at
/// `host(plane)` -- PagedKVPool::download_pages_to's layout, for a slice of it.
inline void copy_pages(const PagedKVPool& pool, std::span<const std::int32_t> ids, std::uint32_t first,
                       std::uint32_t last, std::uint32_t image_first, std::uint32_t image_pages,
                       const std::function<std::byte*(std::size_t plane)>& host, bool to_host,
                       cudaStream_t stream) {
    if (last <= first) { return; }
    if (last > ids.size() || first < image_first || last > image_first + image_pages) {
        throw std::logic_error("host snapshot page range outside its image");
    }
    // Only a page-major pool maps in granules (PagedKVOccupancy::granule_pages), so this names
    // the layout without widening the pool's interface.
    const bool page_major = pool.occupancy().granule_pages != 0;
    for (std::size_t i = 0; i < pool.plane_count(); ++i) {
        const Tensor& plane     = pool.plane(i);
        const int page_axis     = page_major ? 3 : 2;
        const auto stride       = static_cast<std::size_t>(plane.nb[page_axis]);
        const std::size_t heads = page_major ? 1 : static_cast<std::size_t>(plane.ne[3]);
        std::byte* image        = host(i);
        auto* device_base       = static_cast<std::byte*>(plane.data);
        for (std::uint32_t run = first; run < last;) {
            std::uint32_t end = run + 1;
            while (end < last && ids[end] == ids[end - 1] + 1) { ++end; }
            std::byte* h = image + static_cast<std::size_t>(run - image_first) * stride;
            std::byte* d = device_base + static_cast<std::size_t>(ids[run]) * stride;
            const std::size_t width       = static_cast<std::size_t>(end - run) * stride;
            const std::size_t host_pitch  = static_cast<std::size_t>(image_pages) * stride;
            const std::size_t device_pitch = page_major ? width : static_cast<std::size_t>(plane.nb[3]);
            if (to_host) {
                CUDA_CHECK(cudaMemcpy2DAsync(h, host_pitch, d, device_pitch, width, heads,
                                             cudaMemcpyDeviceToHost, stream));
            } else {
                CUDA_CHECK(cudaMemcpy2DAsync(d, device_pitch, h, host_pitch, width, heads,
                                             cudaMemcpyHostToDevice, stream));
            }
            run = end;
        }
    }
}

/// Lays pieces one after another in blocks of `block_bytes`, a piece never straddling two.
struct Packer {
    std::size_t block_bytes;
    std::uint32_t block = 0;
    std::size_t offset  = 0;
    bool fits           = true;
    HostSnapshot::Piece place(std::size_t bytes) {
        if (bytes > block_bytes) { fits = false; return {}; }
        if (offset + bytes > block_bytes) {
            ++block;
            offset = 0;
        }
        HostSnapshot::Piece piece{block, offset, bytes};
        offset += archive_aligned(bytes);
        return piece;
    }
    [[nodiscard]] std::size_t total() const noexcept { return static_cast<std::size_t>(block) * block_bytes + std::max<std::size_t>(offset, 1); }
};

inline void append_u64(std::vector<std::uint8_t>& out, std::uint64_t value) {
    for (int i = 0; i < 8; ++i) { out.push_back(static_cast<std::uint8_t>(value >> (8 * i))); }
}

} // namespace host_tier_detail

cudaEvent_t ProgramImplCore::HostTier::event() {
    if (!spare_events.empty()) {
        cudaEvent_t event = spare_events.back();
        spare_events.pop_back();
        return event;
    }
    cudaEvent_t event = nullptr;
    CUDA_CHECK(cudaEventCreate(&event));
    return event;
}

void ProgramImplCore::HostTier::recycle(cudaEvent_t event) noexcept {
    if (event != nullptr) {
        try { spare_events.push_back(event); } catch (...) { (void)cudaEventDestroy(event); }
    }
}

ProgramImplCore::HostTier::~HostTier() {
    if (copy_stream != nullptr) { (void)cudaStreamSynchronize(copy_stream); }
    for (auto& copy : copies) {
        if (copy.start) { (void)cudaEventDestroy(copy.start); }
        if (copy.done) { (void)cudaEventDestroy(copy.done); }
    }
    copies.clear();
    for (cudaEvent_t event : spare_events) { (void)cudaEventDestroy(event); }
    if (copy_stream != nullptr) { (void)cudaStreamDestroy(copy_stream); }
    if (pool) { pool->drop_owner(owner); }
}

void ProgramImplCore::configure_host_tier(const std::optional<std::size_t>& budget, std::uint64_t weights_identity) {
    using namespace host_tier_detail;
    if (!budget) { return; }
    host_tier  = true;
    auto tier  = std::make_unique<HostTier>();
    tier->budget = *budget;
    // The pool decides what stays; the index only finds it.
    archived_prefixes.configure(std::numeric_limits<std::size_t>::max() / 4, std::size_t{1} << 20);
    const auto& text_pool           = decoder->text_kv.pool();
    const PagedKVPool* backend_pool = backend_kv_cache() ? &backend_kv_cache()->pool() : nullptr;
    auto& model = tier->model;
    for (const std::uint64_t value :
         {weights_identity, static_cast<std::uint64_t>(cfg.layers), static_cast<std::uint64_t>(cfg.hidden),
          static_cast<std::uint64_t>(cfg.token_domain), static_cast<std::uint64_t>(kv_dtype),
          static_cast<std::uint64_t>(kv_quant_group), static_cast<std::uint64_t>(speculative_backend),
          static_cast<std::uint64_t>(draft_window), static_cast<std::uint64_t>(static_cast<std::uint32_t>(stage.first)),
          static_cast<std::uint64_t>(static_cast<std::uint32_t>(stage.last)),
          static_cast<std::uint64_t>(kPagedKVPageSize), static_cast<std::uint64_t>(text_pool.plane_count()),
          static_cast<std::uint64_t>(backend_pool ? backend_pool->plane_count() : 0)}) {
        append_u64(model, value);
    }
    std::size_t offset = 0;
    for (std::size_t plane = 0; plane < text_pool.plane_count(); ++plane) {
        append_u64(model, text_pool.image_plane_bytes(plane, 1));
        tier->text_offset.push_back(offset);
        offset += archive_aligned(text_pool.image_plane_bytes(plane, kChunkPages));
    }
    for (std::size_t plane = 0; backend_pool && plane < backend_pool->plane_count(); ++plane) {
        append_u64(model, backend_pool->image_plane_bytes(plane, 1));
        tier->backend_offset.push_back(offset);
        offset += archive_aligned(backend_pool->image_plane_bytes(plane, kChunkPages));
    }
    // A checkpoint's state has the current state's geometry (its slot is mapped only while a lane
    // holds one, so it cannot be asked here).
    std::size_t largest = 0;
    for (const auto& tensor : prefix_state_tensors(sequences.front(), false)) {
        largest = std::max(largest, tensor.bytes());
    }
    tier->block_bytes = (std::max(offset, largest) + 4095) / 4096 * 4096;
    tier->lane_copy.assign(max_concurrency, nullptr);
    tier->lane_snapshot.resize(max_concurrency);
    tier->lane_parent.resize(max_concurrency);
    tier->lane_started.assign(max_concurrency, 0);
    if (*budget != 0) {
        const auto started = std::chrono::steady_clock::now();
        std::string error;
        tier->pool = family::detail::HostKvPool::attach(tier->block_bytes, *budget, &error);
        const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
        if (!tier->pool) {
            std::fprintf(stderr,
                         "host prefix cache: %.1f GiB for GPU %d not available (%s); conversations the KV "
                         "cache gives back are dropped\n",
                         static_cast<double>(*budget) / (1ULL << 30), device.device, error.c_str());
        } else {
            tier->owner = tier->pool->add_owner();
            CUDA_CHECK(cudaStreamCreateWithFlags(&tier->copy_stream, cudaStreamNonBlocking));
            std::fprintf(stderr,
                         "host prefix cache: %.1f GiB page-locked for GPU %d in %.1f s (%.1f MiB blocks, "
                         "process pool %.1f GiB)\n",
                         static_cast<double>(*budget) / (1ULL << 30), device.device, seconds,
                         static_cast<double>(tier->block_bytes) / (1 << 20),
                         static_cast<double>(tier->pool->capacity_bytes()) / (1ULL << 30));
        }
    }
    host_tier_state = std::move(tier);
}

bool ProgramImplCore::lane_has_current_snapshot(std::uint32_t lane) const noexcept {
    if (!host_tier_state || lane >= host_tier_state->lane_snapshot.size()) { return false; }
    const auto image = host_tier_state->lane_snapshot[lane].lock();
    if (!image || !image->tier || !host_tier_state->pool) { return false; }
    const SequenceState& sequence = sequences[lane];
    try {
        return sequence.retained && image->state.ledger.size() == sequence.ledger.size() &&
               image->state.execution_frontier == sequence.execution_frontier &&
               host_tier_state->pool->alive(image->tier->entry);
    } catch (...) {
        return false;
    }
}

void ProgramImplCore::fence_lane_copies(std::uint32_t lane) noexcept {
    if (!host_tier_state || lane >= host_tier_state->lane_copy.size()) { return; }
    cudaEvent_t& event = host_tier_state->lane_copy[lane];
    if (event == nullptr) { return; }
    // A copy still reads (or writes) this lane's pages and state: everything the compute stream
    // does from here on comes after it. The host does not wait.
    if (cudaStreamWaitEvent(device.stream, event, 0) != cudaSuccess) { (void)cudaGetLastError(); }
    event = nullptr;
}

void ProgramImplCore::poll_host_copies() noexcept {
    if (!host_tier_state) { return; }
    auto& tier = *host_tier_state;
    while (!tier.copies.empty()) {
        auto& copy = tier.copies.front();
        const cudaError_t state = cudaEventQuery(copy.done);
        if (state == cudaErrorNotReady) { break; }
        if (state != cudaSuccess) { (void)cudaGetLastError(); }
        float ms = 0.0F;
        if (cudaEventElapsedTime(&ms, copy.start, copy.done) == cudaSuccess) {
            (copy.reason == host_tier_detail::kRestore ? tier.restore_ms : tier.offload_ms).add(ms);
        } else {
            (void)cudaGetLastError();
        }
        try {
            if (!copy.created.empty() && tier.pool) { tier.pool->mark_ready(copy.created); }
        } catch (...) {}
        if (copy.lane < tier.lane_copy.size() && tier.lane_copy[copy.lane] == copy.done) {
            tier.lane_copy[copy.lane] = nullptr;
        }
        tier.recycle(copy.start);
        tier.recycle(copy.done);
        tier.copies.pop_front();
    }
}

void ProgramImplCore::drain_host_losses() {
    if (!host_tier_state || !host_tier_state->pool) { return; }
    auto dropped = host_tier_state->pool->take_dropped(host_tier_state->owner);
    if (dropped.empty()) { return; }
    std::sort(dropped.begin(), dropped.end());
    (void)archived_prefixes.erase_if([&](const ArchivedSequence& value) {
        return value.tier && std::binary_search(dropped.begin(), dropped.end(), value.tier->entry);
    });
}

std::shared_ptr<family::detail::HostKvPool::Lease> ProgramImplCore::lease_host_snapshot(const ArchivedSequence& image) {
    if (!image.tier || !host_tier_state || !host_tier_state->pool) { return nullptr; }
    auto lease = host_tier_state->pool->lease(image.tier->entry, host_tier_state->owner);
    if (!lease) { return nullptr; }
    return std::make_shared<family::detail::HostKvPool::Lease>(std::move(lease));
}

bool ProgramImplCore::snapshot_lane(std::uint32_t lane, int reason) {
    using namespace host_tier_detail;
    if (!host_tier_state || !host_tier_state->pool || lane >= max_concurrency) { return false; }
    auto& tier = *host_tier_state;
    const SequenceState& sequence = sequences[lane];
    if (sequence.target_only || !sequence.retained || !sequence.cacheable || !sequence.kv || sequence.ledger.empty() ||
        !sequence.prefix_identity.text_only()) {
        return false;
    }
    const std::uint32_t frontier = reusable_append_frontier(sequence);
    if (frontier == 0 || frontier > sequence.ledger.size()) { return false; }
    drain_host_losses();
    const bool checkpoint = sequence.rewrite_checkpoint.valid && decoder->has_checkpoint(lane) &&
                            sequence.rewrite_checkpoint.frontier != 0 &&
                            sequence.rewrite_checkpoint.frontier <= frontier;
    const std::uint32_t checkpoint_frontier = checkpoint ? sequence.rewrite_checkpoint.frontier : 0;
    const auto& text_pool           = decoder->text_kv.pool();
    const bool draft_pages          = sequence.kv->backend.has_value();
    const PagedKVPool* backend_pool = draft_pages ? &backend_kv_cache()->pool() : nullptr;
    const auto text_ids             = sequence.kv->text.page_ids();
    const auto backend_ids          = draft_pages ? sequence.kv->backend->page_ids() : std::span<const std::int32_t>{};
    // KV through the frontier; the draft head's through the token before it (its last column
    // pairs with the next token, which the next turn's bridge writes).
    const std::uint32_t text_pages    = std::min<std::uint32_t>(static_cast<std::uint32_t>(text_ids.size()), pages_for(frontier));
    const std::uint32_t draft_tokens  = draft_pages ? frontier - 1 : 0;
    const std::uint32_t backend_pages = std::min<std::uint32_t>(static_cast<std::uint32_t>(backend_ids.size()), pages_for(draft_tokens));
    if (text_pages < pages_for(frontier) || (draft_pages && backend_pages < pages_for(draft_tokens))) { return false; }
    // Whole chunks only where every plane of them is complete.
    std::uint32_t chunks = (draft_pages ? draft_tokens : frontier) / kvtier::kChunkTokens;
    chunks = std::min({chunks, text_pages / kChunkPages, draft_pages ? backend_pages / kChunkPages : chunks});

    const std::int32_t slot = sequence.prefix_identity.lora_slot();
    kvkeys::AdapterDigest adapter{};
    if (slot >= 0) {
        adapter = ops::lora_store_for_current_device().slot_identity(slot);
        if (adapter == kvkeys::AdapterDigest{}) {
            // An adapter of unknown content (one the trainer shares in place): this program's own
            // name for the slot until the next invalidation.
            std::vector<std::uint8_t> bytes;
            append_u64(bytes, static_cast<std::uint64_t>(slot));
            append_u64(bytes, tier.epoch);
            append_u64(bytes, tier.owner);
            adapter = kvkeys::adapter_digest(bytes);
        }
    }
    const auto root  = kvkeys::namespace_root(tier.model, adapter);
    const auto ledger = std::span<const std::int32_t>(sequence.ledger);
    const auto chain = kvkeys::chunk_chain(root, ledger, frontier / kvtier::kChunkTokens);

    auto snapshot = std::make_unique<HostSnapshot>();
    snapshot->chunks        = chunks;
    snapshot->text_pages    = text_pages;
    snapshot->backend_pages = backend_pages;
    std::vector<family::detail::HostKvPool::Placement> placements;
    const auto give_back = [&] {
        tier.pool->release_objects(placements);
        ++host_refused;
        return false;
    };
    for (std::uint32_t c = 0; c < chunks; ++c) {
        kvtier::ObjectRef ref;
        ref.key = kvkeys::chunk_key(chain[c], draft_pages ? std::optional<std::int32_t>(ledger[(c + 1) * kvtier::kChunkTokens])
                                                          : std::nullopt);
        ref.kind        = kvtier::Kind::Chunk;
        ref.first_token = c * kvtier::kChunkTokens;
        ref.tokens      = kvtier::kChunkTokens;
        ref.bytes       = tier.block_bytes;
        auto placement  = tier.pool->acquire(ref, tier.owner);
        if (!placement) { return give_back(); }
        placements.push_back(*placement);
        snapshot->chunk_objects.push_back(placement->object);
    }
    const auto state_now   = prefix_state_tensors(sequence, false);
    const auto state_saved = checkpoint ? prefix_state_tensors(sequence, true) : std::vector<Tensor>{};
    const std::uint32_t tail_first = chunks * kChunkPages;
    Packer boundary_pack{tier.block_bytes};
    for (std::size_t plane = 0; plane < text_pool.plane_count(); ++plane) {
        snapshot->tail_text.push_back(boundary_pack.place(text_pool.image_plane_bytes(plane, text_pages - tail_first)));
    }
    for (std::size_t plane = 0; backend_pool && plane < backend_pool->plane_count(); ++plane) {
        snapshot->tail_backend.push_back(
            boundary_pack.place(backend_pool->image_plane_bytes(plane, backend_pages - std::min(backend_pages, tail_first))));
    }
    for (const auto& tensor : state_now) {
        if (!tensor.is_contiguous()) { return give_back(); }
        snapshot->state.push_back(boundary_pack.place(tensor.bytes()));
    }
    Packer checkpoint_pack{tier.block_bytes};
    for (const auto& tensor : state_saved) {
        if (!tensor.is_contiguous()) { return give_back(); }
        snapshot->checkpoint_state.push_back(checkpoint_pack.place(tensor.bytes()));
    }
    if (!boundary_pack.fits || !checkpoint_pack.fits) { return give_back(); }
    const std::uint32_t tail_tokens_first = chunks * kvtier::kChunkTokens;
    {
        kvtier::ObjectRef ref;
        ref.key = kvkeys::boundary_key(chunks ? chain[chunks - 1] : root,
                                       ledger.subspan(tail_tokens_first, frontier - tail_tokens_first),
                                       kvkeys::BoundaryKind::AppendFrontier, frontier);
        ref.kind        = kvtier::Kind::Boundary;
        ref.first_token = tail_tokens_first;
        ref.tokens      = frontier - tail_tokens_first;
        ref.bytes       = boundary_pack.total();
        auto placement  = tier.pool->acquire(ref, tier.owner);
        if (!placement) { return give_back(); }
        placements.push_back(*placement);
        snapshot->boundary = placement->object;
    }
    if (checkpoint) {
        const std::uint32_t whole = checkpoint_frontier / kvtier::kChunkTokens;
        kvtier::ObjectRef ref;
        ref.key = kvkeys::boundary_key(whole ? chain[whole - 1] : root,
                                       ledger.subspan(whole * kvtier::kChunkTokens,
                                                      checkpoint_frontier - whole * kvtier::kChunkTokens),
                                       kvkeys::BoundaryKind::RewriteCheckpoint, checkpoint_frontier);
        ref.kind        = kvtier::Kind::Boundary;
        ref.first_token = whole * kvtier::kChunkTokens;
        ref.tokens      = checkpoint_frontier - ref.first_token;
        ref.bytes       = checkpoint_pack.total();
        auto placement  = tier.pool->acquire(ref, tier.owner);
        if (!placement) { return give_back(); }
        placements.push_back(*placement);
        snapshot->checkpoint     = placement->object;
        snapshot->has_checkpoint = true;
    }

    // The copies: behind everything the compute stream has done to these pages and this state.
    cudaEvent_t ordered = tier.event();
    cudaEvent_t start   = nullptr;
    cudaEvent_t done    = nullptr;
    std::vector<family::detail::HostKvPool::ObjectId> created;
    try {
        CUDA_CHECK(cudaEventRecord(ordered, device.stream));
        CUDA_CHECK(cudaStreamWaitEvent(tier.copy_stream, ordered, 0));
        start = tier.event();
        done  = tier.event();
        CUDA_CHECK(cudaEventRecord(start, tier.copy_stream));
        std::uint64_t bytes = 0;
        for (std::uint32_t c = 0; c < chunks; ++c) {
            if (!placements[c].created) { continue; }
            std::byte* block = tier.pool->block(placements[c].object, 0).data();
            copy_pages(text_pool, text_ids, c * kChunkPages, (c + 1) * kChunkPages, c * kChunkPages, kChunkPages,
                       [&](std::size_t plane) { return block + tier.text_offset[plane]; }, true, tier.copy_stream);
            if (backend_pool) {
                copy_pages(*backend_pool, backend_ids, c * kChunkPages, (c + 1) * kChunkPages, c * kChunkPages,
                           kChunkPages, [&](std::size_t plane) { return block + tier.backend_offset[plane]; }, true,
                           tier.copy_stream);
            }
            created.push_back(placements[c].object);
            bytes += tier.block_bytes;
        }
        const auto piece_at = [&](family::detail::HostKvPool::ObjectId object, const HostSnapshot::Piece& piece) {
            return tier.pool->block(object, piece.block).data() + piece.offset;
        };
        const auto& boundary_placement = placements[chunks];
        if (boundary_placement.created) {
            copy_pages(text_pool, text_ids, tail_first, text_pages, tail_first, text_pages - tail_first,
                       [&](std::size_t plane) { return piece_at(snapshot->boundary, snapshot->tail_text[plane]); }, true,
                       tier.copy_stream);
            if (backend_pool && backend_pages > tail_first) {
                copy_pages(*backend_pool, backend_ids, tail_first, backend_pages, tail_first, backend_pages - tail_first,
                           [&](std::size_t plane) { return piece_at(snapshot->boundary, snapshot->tail_backend[plane]); },
                           true, tier.copy_stream);
            }
            for (std::size_t i = 0; i < state_now.size(); ++i) {
                CUDA_CHECK(cudaMemcpyAsync(piece_at(snapshot->boundary, snapshot->state[i]), state_now[i].data,
                                           state_now[i].bytes(), cudaMemcpyDeviceToHost, tier.copy_stream));
            }
            created.push_back(snapshot->boundary);
            bytes += boundary_pack.total();
        }
        if (checkpoint && placements.back().created) {
            for (std::size_t i = 0; i < state_saved.size(); ++i) {
                CUDA_CHECK(cudaMemcpyAsync(piece_at(snapshot->checkpoint, snapshot->checkpoint_state[i]),
                                           state_saved[i].data, state_saved[i].bytes(), cudaMemcpyDeviceToHost,
                                           tier.copy_stream));
            }
            created.push_back(snapshot->checkpoint);
            bytes += checkpoint_pack.total();
        }
        CUDA_CHECK(cudaEventRecord(done, tier.copy_stream));
        tier.recycle(ordered);
        ordered = nullptr;
        snapshot->bytes = bytes;
    } catch (...) {
        // Copies already enqueued finish into blocks we still hold; wait for them, then give the
        // objects back unfinished.
        (void)cudaStreamSynchronize(tier.copy_stream);
        (void)cudaGetLastError();
        tier.recycle(ordered);
        tier.recycle(start);
        tier.recycle(done);
        (void)give_back();
        throw;
    }

    std::vector<family::detail::HostKvPool::ObjectId> objects;
    objects.reserve(placements.size());
    for (const auto& placement : placements) { objects.push_back(placement.object); }
    snapshot->pool  = tier.pool;
    snapshot->entry = tier.pool->add_entry(tier.owner, std::move(objects));
    auto writer = std::make_shared<family::detail::HostKvPool::Lease>(tier.pool->hold(snapshot->entry));

    auto image = std::make_shared<ArchivedSequence>();
    image->state.copy_metadata(sequence);
    if (!checkpoint) { image->state.rewrite_checkpoint = {}; }
    image->text_pages    = text_pages;
    image->backend_pages = backend_pages;
    const std::uint64_t bytes = snapshot->bytes;
    image->tier = std::move(snapshot);
    // An older snapshot of the same conversation is dead weight once this one has moved past it.
    // Agents resend the conversation and append: after the answer as generated (this ledger
    // extends the old one), or after their own rendering of it, from the old one's rewrite
    // checkpoint -- then the old frontier is never asked for again, but only the snapshot this
    // lane descends from can be told apart from a parallel attempt that shares the prompt.
    const auto parent = tier.lane_parent[lane].lock();
    host_superseded += archived_prefixes.erase_if([&](const ArchivedSequence& value) {
        const auto& old = value.state.ledger;
        if (old.size() <= sequence.ledger.size() && std::equal(old.begin(), old.end(), sequence.ledger.begin()) &&
            value.state.prefix_identity.is_prefix_of(sequence.prefix_identity)) {
            return true;
        }
        if (&value != parent.get() || !value.state.rewrite_checkpoint.valid) { return false; }
        const std::size_t through = std::min<std::size_t>(value.state.rewrite_checkpoint.frontier, old.size());
        return through != 0 && through < sequence.ledger.size() &&
               std::equal(old.begin(), old.begin() + static_cast<std::ptrdiff_t>(through), sequence.ledger.begin());
    });
    tier.lane_parent[lane].reset();
    std::vector<std::uint32_t> boundaries{frontier};
    if (checkpoint && checkpoint_frontier != frontier) { boundaries.push_back(checkpoint_frontier); }
    // The index is only an index here: its byte count is the metadata the image keeps on the heap.
    const std::size_t index_bytes = sizeof(ArchivedSequence) + sequence.ledger.size() * (9 * sizeof(TokenId) + 1);
    if (!archived_prefixes.insert(image->state.ledger, boundaries, image, index_bytes)) {
        ++host_refused; // the copies still run; the image is unreachable and goes with its writer lease
    } else {
        ++host_archived;
    }
    tier.copies.push_back(HostTier::Copy{start, done, reason, lane, std::move(created), std::move(writer), bytes});
    tier.lane_copy[lane]     = done;
    tier.lane_snapshot[lane] = image;
    tier.offload_bytes += bytes;
    if (reason == kBackground) { ++tier.offloads; }
    if (reason == kSpill) { ++tier.spills; }
    return true;
}

void ProgramImplCore::restore_host_snapshot(SequenceState& sequence, const RequestPlanImpl& plan) {
    using namespace host_tier_detail;
    auto& tier                 = *host_tier_state;
    const ArchivedSequence& image = *plan.archived;
    const HostSnapshot& snapshot  = *image.tier;
    clear_lane(sequence, requests[sequence.lane]);
    sequence.copy_metadata(image.state);
    reserve_sequence_kv(sequence, plan.text_kv_page_entitlement, plan.backend_kv_page_entitlement);
    const auto backend_tokens = speculative_backend == SpeculativeBackend::Mtp
        ? plan.reuse_base - 1 : speculative_backend == SpeculativeBackend::DFlash ? plan.reuse_base : 0;
    materialize_sequence_kv(sequence, plan.reuse_base, backend_tokens);
    const auto text_ids    = sequence.kv->text.page_ids();
    const auto backend_ids = sequence.kv->backend ? sequence.kv->backend->page_ids() : std::span<const std::int32_t>{};
    const auto text_need    = static_cast<std::uint32_t>(text_ids.size());
    const auto backend_need = static_cast<std::uint32_t>(backend_ids.size());
    if (text_need > snapshot.text_pages || backend_need > snapshot.backend_pages) {
        throw std::logic_error("host prefix snapshot holds fewer pages than its restore needs");
    }
    const auto& text_pool           = decoder->text_kv.pool();
    const PagedKVPool* backend_pool = sequence.kv->backend ? &backend_kv_cache()->pool() : nullptr;
    const auto piece_at = [&](family::detail::HostKvPool::ObjectId object, const HostSnapshot::Piece& piece) {
        return tier.pool->block(object, piece.block).data() + piece.offset;
    };
    const std::uint32_t tail_first = snapshot.chunks * kChunkPages;
    const auto restore_pages = [&](const PagedKVPool& pool, std::span<const std::int32_t> ids, std::uint32_t need,
                                   std::uint32_t held, const std::vector<std::size_t>& offsets,
                                   const std::vector<HostSnapshot::Piece>& tail) {
        for (std::uint32_t c = 0; c < snapshot.chunks && c * kChunkPages < need; ++c) {
            std::byte* block = tier.pool->block(snapshot.chunk_objects[c], 0).data();
            copy_pages(pool, ids, c * kChunkPages, std::min(need, (c + 1) * kChunkPages), c * kChunkPages, kChunkPages,
                       [&](std::size_t plane) { return block + offsets[plane]; }, false, tier.copy_stream);
        }
        if (need > tail_first) {
            copy_pages(pool, ids, tail_first, need, tail_first, held - tail_first,
                       [&](std::size_t plane) { return piece_at(snapshot.boundary, tail[plane]); }, false,
                       tier.copy_stream);
        }
    };
    cudaEvent_t ordered = tier.event();
    cudaEvent_t start   = tier.event();
    cudaEvent_t done    = tier.event();
    CUDA_CHECK(cudaEventRecord(ordered, device.stream));
    CUDA_CHECK(cudaStreamWaitEvent(tier.copy_stream, ordered, 0));
    CUDA_CHECK(cudaEventRecord(start, tier.copy_stream));
    restore_pages(text_pool, text_ids, text_need, snapshot.text_pages, tier.text_offset, snapshot.tail_text);
    if (backend_pool) {
        restore_pages(*backend_pool, backend_ids, backend_need, snapshot.backend_pages, tier.backend_offset,
                      snapshot.tail_backend);
    }
    const auto upload = [&](family::detail::HostKvPool::ObjectId object, const std::vector<HostSnapshot::Piece>& pieces,
                            const std::vector<Tensor>& tensors) {
        if (pieces.size() != tensors.size()) { throw std::logic_error("prefix state layout changed"); }
        for (std::size_t i = 0; i < tensors.size(); ++i) {
            if (!tensors[i].is_contiguous() || tensors[i].bytes() != pieces[i].bytes) {
                throw std::logic_error("prefix state geometry changed");
            }
            CUDA_CHECK(cudaMemcpyAsync(tensors[i].data, piece_at(object, pieces[i]), pieces[i].bytes,
                                       cudaMemcpyHostToDevice, tier.copy_stream));
        }
    };
    upload(snapshot.boundary, snapshot.state, prefix_state_tensors(sequence, false));
    if (sequence.rewrite_checkpoint.valid) {
        if (snapshot.has_checkpoint && decoder->try_acquire_checkpoint(sequence.lane)) {
            upload(snapshot.checkpoint, snapshot.checkpoint_state, prefix_state_tensors(sequence, true));
        } else {
            sequence.rewrite_checkpoint = {};
            if (is_rewrite_checkpoint_restore(plan.reuse)) {
                throw std::logic_error("admitted prefix checkpoint lost its state slot");
            }
        }
    }
    CUDA_CHECK(cudaEventRecord(done, tier.copy_stream));
    // The lane's next GPU work reads these bytes; the host goes on staging.
    CUDA_CHECK(cudaStreamWaitEvent(device.stream, done, 0));
    tier.recycle(ordered);
    std::uint64_t bytes = static_cast<std::uint64_t>(text_need) * text_pool.occupancy().page_bytes;
    if (backend_pool) { bytes += static_cast<std::uint64_t>(backend_need) * backend_pool->occupancy().page_bytes; }
    for (const auto& piece : snapshot.state) { bytes += piece.bytes; }
    tier.copies.push_back(HostTier::Copy{start, done, kRestore, sequence.lane, {}, plan.host_lease, bytes});
    tier.restore_bytes += bytes;
    ++tier.restores;
    tier.restored_tokens += plan.reuse_base;
    ++checkpoint_revisions[sequence.lane];
}

void ProgramImplCore::background_offload() noexcept {
    if (!host_tier_state || !host_tier_state->pool) { return; }
    auto& tier = *host_tier_state;
    try {
        poll_host_copies();
        drain_host_losses();
        // One background copy in flight: the stream stays free for restores and spills.
        for (const auto& copy : tier.copies) {
            if (copy.reason == host_tier_detail::kBackground) { return; }
        }
        static const std::uint32_t min_tokens = [] {
            const char* raw = std::getenv("SUROGATE_SERVE_HOST_TIER_MIN_TOKENS");
            return raw ? static_cast<std::uint32_t>(std::strtoul(raw, nullptr, 10)) : kvtier::kChunkTokens;
        }();
        std::optional<std::uint32_t> pick;
        for (std::uint32_t lane = 0; lane < max_concurrency; ++lane) {
            const SequenceState& sequence = sequences[lane];
            if (!sequence.retained || !sequence.cacheable || sequence.target_only || !sequence.kv ||
                sequence.ledger.size() < min_tokens || requests[lane].lifecycle == Lifecycle::Prefilling ||
                requests[lane].lifecycle == Lifecycle::Active || requests[lane].lifecycle == Lifecycle::Pending ||
                lane_has_current_snapshot(lane)) {
                continue;
            }
            if (!pick || tier.lane_started[lane] < tier.lane_started[*pick]) { pick = lane; }
        }
        if (pick) { (void)snapshot_lane(*pick, host_tier_detail::kBackground); }
    } catch (...) {
        // Opportunistic: the lane keeps its GPU copy, and a spill can still take one.
        (void)cudaGetLastError();
    }
}

void ProgramImplCore::spill_retained_lane(std::uint32_t lane) noexcept {
    if (!has_retained_lane(lane)) {
        evict_retained_lane(lane);
        return;
    }
    const SequenceState& sequence = sequences[lane];
    const bool keepable = sequence.cacheable && !sequence.target_only && sequence.kv && !sequence.ledger.empty();
    if (!keepable) {
        evict_retained_lane(lane);
        return;
    }
    if (!host_tier_state || !host_tier_state->pool) {
        ++host_dropped_lanes;
        evict_retained_lane(lane);
        return;
    }
    const auto started = std::chrono::steady_clock::now();
    try {
        poll_host_copies();
        drain_host_losses();
        if (lane_has_current_snapshot(lane)) {
            ++host_tier_state->spills_free;
        } else if (!snapshot_lane(lane, host_tier_detail::kSpill)) {
            ++host_dropped_lanes;
        }
    } catch (...) {
        (void)cudaGetLastError();
        ++host_dropped_lanes;
    }
    admit_trace_step("evict: spill retained lane to host", lane, started);
    // clear_lane fences the lane's copy, so the pages go back only behind it.
    evict_retained_lane(lane);
}

HostKvStats ProgramImplCore::host_kv_stats() const noexcept {
    HostKvStats out;
    out.entries      = static_cast<std::uint32_t>(archived_prefixes.size());
    out.dropped_lanes = host_dropped_lanes;
    out.archived     = host_archived;
    out.refused      = host_refused;
    out.superseded   = host_superseded;
    out.gpu_hit_requests    = reuse_requests_gpu;
    out.shared_hit_requests = reuse_requests_shared;
    out.host_hit_requests   = reuse_requests_host;
    out.miss_requests       = reuse_requests_none;
    out.gpu_hit_tokens      = reuse_tokens_gpu;
    out.shared_hit_tokens   = reuse_tokens_shared;
    out.host_hit_tokens     = reuse_tokens_host;
    if (!host_tier) {
        out.capacity_bytes = kArchivedPrefixBytes;
        out.used_bytes     = archived_prefixes.bytes();
        out.budget_bytes   = kArchivedPrefixBytes;
        return out;
    }
    if (!host_tier_state) { return out; }
    const auto& tier = *host_tier_state;
    out.budget_bytes    = tier.budget;
    out.spills          = tier.spills;
    out.spills_free     = tier.spills_free;
    out.offloads        = tier.offloads;
    out.restores        = tier.restores;
    out.restored_tokens = tier.restored_tokens;
    out.offload_bytes   = tier.offload_bytes;
    out.restore_bytes   = tier.restore_bytes;
    out.offload_ms      = tier.offload_ms;
    out.restore_ms      = tier.restore_ms;
    if (tier.pool) {
        try {
            out.capacity_bytes = tier.pool->capacity_bytes();
            out.used_bytes     = tier.pool->used_bytes();
            out.evictions      = tier.pool->evictions();
            out.shared_objects = tier.pool->shared_objects();
        } catch (...) {}
    }
    return out;
}

} // namespace sinfer::family::detail::SINFER_FAMILY_RUNTIME_NS
