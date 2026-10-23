#include "family/impl/kv_tier/host_pool.h"

#include <cuda_runtime_api.h>
#include <sys/mman.h>

#include <algorithm>
#include <cstring>
#include <limits>
#include <map>
#include <stdexcept>
#include <thread>

namespace sinfer::family::detail {
namespace {

/// Page-locks `bytes` of fresh anonymous memory: mmap, first touch from many threads, then
/// cudaHostRegister. cudaHostAlloc pins at ~1.8 GB/s and serialises (load/host_bank.cpp says so
/// for the expert bank); this runs at memory bandwidth. Null when a step refuses.
std::shared_ptr<std::byte> register_host_range(std::size_t bytes) noexcept {
    void* mem = ::mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (mem == MAP_FAILED) { return nullptr; }
    try {
        const std::size_t workers = std::clamp<std::size_t>(std::thread::hardware_concurrency() / 2, 1, 16);
        const std::size_t chunk   = (bytes + workers - 1) / workers;
        std::vector<std::jthread> touchers;
        for (std::size_t w = 0; w < workers && w * chunk < bytes; ++w) {
            touchers.emplace_back([mem, w, chunk, bytes] {
                const std::size_t begin = w * chunk;
                std::memset(static_cast<std::byte*>(mem) + begin, 0, std::min(chunk, bytes - begin));
            });
        }
    } catch (...) {
        (void)::munmap(mem, bytes);
        return nullptr;
    }
    if (cudaHostRegister(mem, bytes, cudaHostRegisterPortable) != cudaSuccess) {
        (void)cudaGetLastError();
        (void)::munmap(mem, bytes);
        return nullptr;
    }
    try {
        return std::shared_ptr<std::byte>(static_cast<std::byte*>(mem), [bytes](std::byte* p) {
            (void)cudaHostUnregister(p);
            (void)::munmap(p, bytes);
        });
    } catch (...) {
        return nullptr; // the deleter has run
    }
}

std::mutex& registry_mutex() {
    static std::mutex mutex;
    return mutex;
}

std::map<std::size_t, std::weak_ptr<HostKvPool>>& registry() {
    static std::map<std::size_t, std::weak_ptr<HostKvPool>> pools;
    return pools;
}

} // namespace

std::shared_ptr<HostKvPool> HostKvPool::attach(std::size_t block_bytes, std::size_t add_bytes, std::string* error) {
    if (block_bytes == 0 || add_bytes < block_bytes) {
        if (error) { *error = "the budget holds less than one block"; }
        return nullptr;
    }
    const std::size_t blocks = add_bytes / block_bytes;
    auto memory = register_host_range(blocks * block_bytes);
    if (!memory) {
        if (error) { *error = "the host would not page-lock it"; }
        return nullptr;
    }
    std::lock_guard lock(registry_mutex());
    auto& slot = registry()[block_bytes];
    std::shared_ptr<HostKvPool> pool = slot.lock();
    if (!pool) {
        pool = std::shared_ptr<HostKvPool>(new HostKvPool(block_bytes));
        slot = pool;
    }
    pool->add_arena(Arena{std::move(memory), blocks});
    return pool;
}

std::shared_ptr<HostKvPool> HostKvPool::create_unpinned(std::size_t block_bytes, std::size_t blocks) {
    auto pool = std::shared_ptr<HostKvPool>(new HostKvPool(block_bytes));
    std::shared_ptr<std::byte> memory(new std::byte[block_bytes * blocks], std::default_delete<std::byte[]>());
    pool->add_arena(Arena{std::move(memory), blocks});
    return pool;
}

HostKvPool::~HostKvPool() = default;

void HostKvPool::add_arena(Arena arena) {
    std::lock_guard lock(mutex_);
    arena_first_.push_back(total_blocks_);
    for (std::size_t i = 0; i < arena.blocks; ++i) {
        free_blocks_.push_back(static_cast<std::uint32_t>(total_blocks_ + arena.blocks - 1 - i));
    }
    total_blocks_ += arena.blocks;
    arenas_.push_back(std::move(arena));
}

std::byte* HostKvPool::block_address(std::uint32_t block) const {
    const auto it = std::upper_bound(arena_first_.begin(), arena_first_.end(), static_cast<std::size_t>(block));
    const std::size_t arena = static_cast<std::size_t>(it - arena_first_.begin()) - 1;
    return arenas_[arena].memory.get() + (block - arena_first_[arena]) * block_bytes_;
}

std::size_t HostKvPool::capacity_bytes() const {
    std::lock_guard lock(mutex_);
    return total_blocks_ * block_bytes_;
}

std::size_t HostKvPool::used_bytes() const {
    std::lock_guard lock(mutex_);
    return (total_blocks_ - free_blocks_.size()) * block_bytes_;
}

std::size_t HostKvPool::entries() const {
    std::lock_guard lock(mutex_);
    std::size_t alive = 0;
    for (const auto& [id, entry] : entries_) { alive += entry.alive ? 1 : 0; }
    return alive;
}

std::size_t HostKvPool::objects() const {
    std::lock_guard lock(mutex_);
    return objects_.size() - free_objects_.size();
}

std::uint64_t HostKvPool::evictions() const {
    std::lock_guard lock(mutex_);
    return evictions_;
}

std::uint64_t HostKvPool::shared_objects() const {
    std::lock_guard lock(mutex_);
    return shared_objects_;
}

HostKvPool::OwnerId HostKvPool::add_owner() {
    std::lock_guard lock(mutex_);
    return next_owner_++;
}

std::vector<HostKvPool::EntryId> HostKvPool::take_dropped(OwnerId owner) {
    std::lock_guard lock(mutex_);
    auto it = dropped_.find(owner);
    if (it == dropped_.end()) { return {}; }
    std::vector<EntryId> out = std::move(it->second);
    dropped_.erase(it);
    return out;
}

void HostKvPool::unhold(ObjectId id, std::vector<kvtier::ObjectRef>& removed) {
    Object& object = objects_[id];
    if (object.holds == 0) { throw std::logic_error("host KV object released more often than held"); }
    if (--object.holds != 0) { return; }
    for (const std::uint32_t block : object.blocks) { free_blocks_.push_back(block); }
    object.blocks.clear();
    if (auto it = by_key_.find(object.ref.key); it != by_key_.end() && it->second == id) { by_key_.erase(it); }
    if (object.ready && !object.poisoned) { removed.push_back(object.ref); }
    object = Object{};
    free_objects_.push_back(id);
}

void HostKvPool::retire_entry(EntryId id, std::vector<kvtier::ObjectRef>& removed) {
    auto it = entries_.find(id);
    if (it == entries_.end()) { return; }
    const std::vector<ObjectId> objects = std::move(it->second.objects);
    entries_.erase(it);
    for (const ObjectId object : objects) { unhold(object, removed); }
}

bool HostKvPool::make_room(std::size_t blocks, std::vector<kvtier::ObjectRef>& removed) {
    while (free_blocks_.size() < blocks) {
        auto victim = entries_.end();
        for (auto it = entries_.begin(); it != entries_.end(); ++it) {
            const Entry& entry = it->second;
            if (!entry.alive || entry.leases != 0) { continue; }
            if (victim == entries_.end()) { victim = it; continue; }
            const Entry& best = victim->second;
            // Unpinned before pinned (captured shared prefixes), then least recently used.
            if (entry.pinned != best.pinned ? !entry.pinned : entry.last_use < best.last_use) { victim = it; }
        }
        if (victim == entries_.end()) { return false; }
        const EntryId id = victim->first;
        if (victim->second.owner != kNoOwner) { dropped_[victim->second.owner].push_back(id); }
        for (auto ext = external_.begin(); ext != external_.end(); ++ext) {
            if (ext->second == id) { external_.erase(ext); break; }
        }
        ++evictions_;
        retire_entry(id, removed);
    }
    return true;
}

std::optional<HostKvPool::Placement> HostKvPool::acquire(const kvtier::ObjectRef& ref, OwnerId writer) {
    std::vector<kvtier::ObjectRef> removed;
    std::optional<Placement> out;
    {
        std::lock_guard lock(mutex_);
        if (auto it = by_key_.find(ref.key); it != by_key_.end()) {
            Object& object = objects_[it->second];
            if (!object.poisoned && object.ref.bytes == ref.bytes && object.ref.kind == ref.kind) {
                ++object.holds;
                ++shared_objects_;
                out = Placement{it->second, false};
            }
        }
        if (!out) {
            const std::size_t need = std::max<std::size_t>(1, (ref.bytes + block_bytes_ - 1) / block_bytes_);
            if (need <= total_blocks_ && make_room(need, removed)) {
                ObjectId id = 0;
                if (!free_objects_.empty()) {
                    id = free_objects_.back();
                    free_objects_.pop_back();
                } else {
                    id = static_cast<ObjectId>(objects_.size());
                    objects_.emplace_back();
                }
                Object& object = objects_[id];
                object.ref     = ref;
                object.holds   = 1;
                object.writer  = writer;
                object.ready   = false;
                object.live    = true;
                object.blocks.assign(free_blocks_.end() - static_cast<std::ptrdiff_t>(need), free_blocks_.end());
                free_blocks_.resize(free_blocks_.size() - need);
                by_key_[ref.key] = id;
                out = Placement{id, true};
            }
        }
    }
    notify(removed, false);
    return out;
}

std::span<std::byte> HostKvPool::block(ObjectId object, std::size_t index) {
    std::lock_guard lock(mutex_);
    const Object& o = objects_.at(object);
    return {block_address(o.blocks.at(index)), block_bytes_};
}

std::size_t HostKvPool::block_count(ObjectId object) const {
    std::lock_guard lock(mutex_);
    return objects_.at(object).blocks.size();
}

void HostKvPool::release_objects(std::span<const Placement> placements) {
    std::vector<kvtier::ObjectRef> removed;
    {
        std::lock_guard lock(mutex_);
        for (const Placement& placement : placements) {
            Object& object = objects_.at(placement.object);
            if (placement.created && !object.ready) {
                // Nobody will complete these bytes: entries that found the object in the
                // meantime must not lease it, and the next writer starts afresh.
                object.poisoned = true;
                if (auto it = by_key_.find(object.ref.key); it != by_key_.end() && it->second == placement.object) {
                    by_key_.erase(it);
                }
            }
            unhold(placement.object, removed);
        }
    }
    notify(removed, false);
}

HostKvPool::EntryId HostKvPool::add_entry(OwnerId owner, std::vector<ObjectId> objects, bool pinned) {
    std::lock_guard lock(mutex_);
    const EntryId id = next_entry_++;
    Entry entry;
    entry.owner    = owner;
    entry.objects  = std::move(objects);
    entry.last_use = ++clock_;
    entry.pinned   = pinned;
    entries_.emplace(id, std::move(entry));
    return id;
}

void HostKvPool::mark_ready(std::span<const ObjectId> objects) {
    std::vector<kvtier::ObjectRef> stored;
    {
        std::lock_guard lock(mutex_);
        for (const ObjectId id : objects) {
            Object& object = objects_.at(id);
            if (!object.live || object.poisoned || object.ready) { continue; }
            object.ready = true;
            stored.push_back(object.ref);
        }
    }
    notify(stored, true);
}

void HostKvPool::drop_entry(EntryId id) {
    std::vector<kvtier::ObjectRef> removed;
    {
        std::lock_guard lock(mutex_);
        auto it = entries_.find(id);
        if (it == entries_.end() || !it->second.alive) { return; }
        it->second.alive = false;
        if (it->second.leases == 0) { retire_entry(id, removed); }
    }
    notify(removed, false);
}

void HostKvPool::touch(EntryId id) {
    std::lock_guard lock(mutex_);
    if (auto it = entries_.find(id); it != entries_.end()) { it->second.last_use = ++clock_; }
}

HostKvPool::Lease HostKvPool::lease(EntryId id, OwnerId reader) {
    std::lock_guard lock(mutex_);
    auto it = entries_.find(id);
    if (it == entries_.end() || !it->second.alive) { return {}; }
    for (const ObjectId object : it->second.objects) {
        const Object& o = objects_[object];
        if (o.poisoned || (!o.ready && o.writer != reader)) { return {}; }
    }
    ++it->second.leases;
    it->second.last_use = ++clock_;
    return Lease(shared_from_this(), id);
}

HostKvPool::Lease HostKvPool::hold(EntryId id) {
    std::lock_guard lock(mutex_);
    auto it = entries_.find(id);
    if (it == entries_.end()) { return {}; }
    ++it->second.leases;
    return Lease(shared_from_this(), id);
}

bool HostKvPool::alive(EntryId id) const {
    std::lock_guard lock(mutex_);
    const auto it = entries_.find(id);
    return it != entries_.end() && it->second.alive;
}

void HostKvPool::drop_owner(OwnerId owner) {
    std::vector<EntryId> ids;
    {
        std::lock_guard lock(mutex_);
        for (const auto& [id, entry] : entries_) {
            if (entry.owner == owner && entry.alive) { ids.push_back(id); }
        }
        dropped_.erase(owner);
    }
    for (const EntryId id : ids) { drop_entry(id); }
}

void HostKvPool::Lease::reset() noexcept {
    if (entry_ != 0 && pool_) { pool_->release_lease(entry_); }
    entry_ = 0;
    pool_.reset();
}

void HostKvPool::release_lease(EntryId id) noexcept {
    std::vector<kvtier::ObjectRef> removed;
    try {
        std::lock_guard lock(mutex_);
        auto it = entries_.find(id);
        if (it == entries_.end() || it->second.leases == 0) { return; }
        if (--it->second.leases == 0 && !it->second.alive) { retire_entry(id, removed); }
    } catch (...) {
        return;
    }
    try { notify(removed, false); } catch (...) {}
}

void HostKvPool::notify(std::vector<kvtier::ObjectRef>& refs, bool stored) {
    if (refs.empty()) { return; }
    kvtier::Listener listener;
    {
        std::lock_guard lock(mutex_);
        listener = listener_;
    }
    if (listener) { listener(refs, stored); }
    refs.clear();
}

// ---- kvtier::Backend ---------------------------------------------------------------------------

kvtier::Caps HostKvPool::caps() const {
    return kvtier::Caps{.persistent = 0, .cross_process = 0, .zero_copy_registered = 1, .async = 0,
                        .chunk_tokens = kvtier::kChunkTokens};
}

bool HostKvPool::register_host(void*, std::size_t) { return true; }

void HostKvPool::contains(std::span<const kvtier::ObjectRef> objects, std::span<std::uint8_t> present) {
    std::lock_guard lock(mutex_);
    for (std::size_t i = 0; i < objects.size() && i < present.size(); ++i) {
        const auto it = by_key_.find(objects[i].key);
        present[i] = it != by_key_.end() && objects_[it->second].ready && !objects_[it->second].poisoned ? 1 : 0;
    }
}

kvtier::Ticket HostKvPool::put(std::span<const kvtier::ObjectRef> objects, std::span<const kvtier::Span> spans) {
    if (objects.size() != spans.size()) { throw std::invalid_argument("kvtier put needs one span per object"); }
    bool ok = true;
    for (std::size_t i = 0; i < objects.size(); ++i) {
        {
            std::lock_guard lock(mutex_);
            if (external_.contains(objects[i].key)) { continue; }
        }
        auto placement = acquire(objects[i], kNoOwner);
        if (!placement) { ok = false; continue; }
        if (placement->created) {
            const auto* source = static_cast<const std::byte*>(spans[i].data);
            std::size_t left   = std::min<std::size_t>(spans[i].bytes, objects[i].bytes);
            for (std::size_t b = 0; left != 0; ++b) {
                const std::size_t n = std::min(left, block_bytes_);
                std::memcpy(block(placement->object, b).data(), source, n);
                source += n;
                left -= n;
            }
            mark_ready(std::span<const ObjectId>(&placement->object, 1));
        }
        const EntryId entry = add_entry(kNoOwner, {placement->object});
        std::lock_guard lock(mutex_);
        external_[objects[i].key] = entry;
    }
    return ok ? next_ticket_++ : 0;
}

kvtier::Ticket HostKvPool::get(std::span<const kvtier::ObjectRef> objects, std::span<const kvtier::Span> spans) {
    if (objects.size() != spans.size()) { throw std::invalid_argument("kvtier get needs one span per object"); }
    std::lock_guard lock(mutex_);
    for (std::size_t i = 0; i < objects.size(); ++i) {
        const auto it = by_key_.find(objects[i].key);
        if (it == by_key_.end() || !objects_[it->second].ready || objects_[it->second].poisoned) { return 0; }
        const Object& object = objects_[it->second];
        auto* target         = static_cast<std::byte*>(spans[i].data);
        std::size_t left     = std::min<std::size_t>(spans[i].bytes, object.ref.bytes);
        for (std::size_t b = 0; left != 0; ++b) {
            const std::size_t n = std::min(left, block_bytes_);
            std::memcpy(target, block_address(object.blocks.at(b)), n);
            target += n;
            left -= n;
        }
    }
    return next_ticket_++;
}

int HostKvPool::poll(kvtier::Ticket ticket) { return ticket == 0 ? SINFER_KVTIER_FAILED : SINFER_KVTIER_DONE; }

void HostKvPool::erase(std::span<const kvtier::Key> keys) {
    std::vector<EntryId> drop;
    {
        std::lock_guard lock(mutex_);
        for (const kvtier::Key& key : keys) {
            if (auto it = external_.find(key); it != external_.end()) {
                drop.push_back(it->second);
                external_.erase(it);
            }
        }
    }
    for (const EntryId id : drop) { drop_entry(id); }
}

void HostKvPool::set_listener(kvtier::Listener listener) {
    std::lock_guard lock(mutex_);
    listener_ = std::move(listener);
}

// ---- C table over the pool, and the C++ face of a C table --------------------------------------

namespace {
HostKvPool* as_pool(void* ctx) { return static_cast<HostKvPool*>(ctx); }
std::vector<kvtier::ObjectRef> refs_of(const sinfer_kvtier_object* objects, std::size_t count) {
    std::vector<kvtier::ObjectRef> out;
    out.reserve(count);
    for (std::size_t i = 0; i < count; ++i) { out.push_back(kvtier::from_c(objects[i])); }
    return out;
}
} // namespace

sinfer_kvtier_v1 host_pool_c_table(HostKvPool* pool) {
    sinfer_kvtier_v1 table{};
    table.abi_version = SINFER_KVTIER_ABI_VERSION;
    table.ctx         = pool;
    table.caps        = [](void* ctx) { return as_pool(ctx)->caps(); };
    table.register_host = [](void* ctx, void* base, std::size_t bytes) {
        return as_pool(ctx)->register_host(base, bytes) ? 0 : -1;
    };
    table.contains = [](void* ctx, const sinfer_kvtier_object* objects, std::size_t count, std::uint8_t* present) {
        const auto refs = refs_of(objects, count);
        as_pool(ctx)->contains(refs, std::span<std::uint8_t>(present, count));
    };
    table.put = [](void* ctx, const sinfer_kvtier_object* objects, const sinfer_kvtier_span* spans, std::size_t count) {
        const auto refs = refs_of(objects, count);
        return as_pool(ctx)->put(refs, std::span<const kvtier::Span>(spans, count));
    };
    table.get = [](void* ctx, const sinfer_kvtier_object* objects, const sinfer_kvtier_span* spans, std::size_t count) {
        const auto refs = refs_of(objects, count);
        return as_pool(ctx)->get(refs, std::span<const kvtier::Span>(spans, count));
    };
    table.poll  = [](void* ctx, std::uint64_t ticket) { return as_pool(ctx)->poll(ticket); };
    table.erase = [](void* ctx, const sinfer_kvtier_key* keys, std::size_t count) {
        std::vector<kvtier::Key> out(count);
        for (std::size_t i = 0; i < count; ++i) { std::memcpy(out[i].bytes.data(), keys[i].bytes, 16); }
        as_pool(ctx)->erase(out);
    };
    table.set_listener = [](void* ctx,
                            void (*callback)(void*, const sinfer_kvtier_object*, std::size_t, int),
                            void* listener_ctx) {
        if (callback == nullptr) {
            as_pool(ctx)->set_listener({});
            return;
        }
        as_pool(ctx)->set_listener([callback, listener_ctx](std::span<const kvtier::ObjectRef> refs, bool stored) {
            std::vector<sinfer_kvtier_object> objects;
            objects.reserve(refs.size());
            for (const auto& ref : refs) { objects.push_back(kvtier::to_c(ref)); }
            callback(listener_ctx, objects.data(), objects.size(), stored ? 1 : 0);
        });
    };
    table.close = [](void*) {};
    return table;
}

CBackend::CBackend(const sinfer_kvtier_v1* table) : table_(table) {
    if (table_ == nullptr || table_->abi_version != SINFER_KVTIER_ABI_VERSION) {
        throw std::invalid_argument("KV tier back end speaks another ABI version");
    }
}

CBackend::~CBackend() {
    if (table_->set_listener) { table_->set_listener(table_->ctx, nullptr, nullptr); }
    if (table_->close) { table_->close(table_->ctx); }
}

kvtier::Caps CBackend::caps() const { return table_->caps(table_->ctx); }

bool CBackend::register_host(void* base, std::size_t bytes) {
    return table_->register_host(table_->ctx, base, bytes) == 0;
}

void CBackend::contains(std::span<const kvtier::ObjectRef> objects, std::span<std::uint8_t> present) {
    std::vector<sinfer_kvtier_object> c;
    c.reserve(objects.size());
    for (const auto& ref : objects) { c.push_back(kvtier::to_c(ref)); }
    table_->contains(table_->ctx, c.data(), c.size(), present.data());
}

kvtier::Ticket CBackend::put(std::span<const kvtier::ObjectRef> objects, std::span<const kvtier::Span> spans) {
    std::vector<sinfer_kvtier_object> c;
    c.reserve(objects.size());
    for (const auto& ref : objects) { c.push_back(kvtier::to_c(ref)); }
    return table_->put(table_->ctx, c.data(), spans.data(), c.size());
}

kvtier::Ticket CBackend::get(std::span<const kvtier::ObjectRef> objects, std::span<const kvtier::Span> spans) {
    std::vector<sinfer_kvtier_object> c;
    c.reserve(objects.size());
    for (const auto& ref : objects) { c.push_back(kvtier::to_c(ref)); }
    return table_->get(table_->ctx, c.data(), spans.data(), c.size());
}

int CBackend::poll(kvtier::Ticket ticket) { return table_->poll(table_->ctx, ticket); }

void CBackend::erase(std::span<const kvtier::Key> keys) {
    std::vector<sinfer_kvtier_key> c(keys.size());
    for (std::size_t i = 0; i < keys.size(); ++i) { std::memcpy(c[i].bytes, keys[i].bytes.data(), 16); }
    table_->erase(table_->ctx, c.data(), c.size());
}

void CBackend::forward(void* context, const sinfer_kvtier_object* objects, std::size_t count, int stored) {
    auto* listener = static_cast<kvtier::Listener*>(context);
    if (listener && *listener) { (*listener)(refs_of(objects, count), stored != 0); }
}

void CBackend::set_listener(kvtier::Listener listener) {
    listener_ = std::make_unique<kvtier::Listener>(std::move(listener));
    table_->set_listener(table_->ctx, &CBackend::forward, listener_.get());
}

} // namespace sinfer::family::detail
