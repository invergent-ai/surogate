// The engine's host RAM tier (--prefix-cache-host-gb): page-locked memory in fixed-size blocks,
// shared by every program of the process whose objects have that block size -- the data-parallel
// replicas of one model, the stages of a pipeline.
//
// What lives here is described in api/kv_tier.h: chunk and boundary objects, keyed by a chained
// SHA-256, so programs that saw the same prefix share its chunks. A program's snapshot of one
// conversation is an *entry*: the objects it needs to resume at its frontiers. Entries are the
// unit of eviction (least recently used first, across programs); an object goes when no entry
// holds it. A program learns that one of its entries was dropped to make room for another program
// from `take_dropped`.
//
// Writers fill blocks with their own copy streams and mark the objects ready once the copies are
// done. A reader on another program may lease an entry only when every object is ready; the
// writer itself may sooner (its copy stream orders a restore after its own writes).
#pragma once

#include "api/kv_tier.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <string>
#include <unordered_map>
#include <vector>

namespace sinfer::family::detail {

class HostKvPool final : public kvtier::Backend, public std::enable_shared_from_this<HostKvPool> {
public:
    using ObjectId = std::uint32_t;
    using EntryId  = std::uint64_t;
    using OwnerId  = std::uint32_t;
    static constexpr OwnerId kNoOwner = 0;

    /// The process's pool for `block_bytes` blocks, grown by `add_bytes` of page-locked memory (one
    /// program's budget). Null with `error` set when the memory cannot be page-locked.
    [[nodiscard]] static std::shared_ptr<HostKvPool> attach(std::size_t block_bytes, std::size_t add_bytes,
                                                            std::string* error);
    /// A pool over ordinary heap memory, for tests that have no GPU.
    [[nodiscard]] static std::shared_ptr<HostKvPool> create_unpinned(std::size_t block_bytes, std::size_t blocks);

    HostKvPool(const HostKvPool&)            = delete;
    HostKvPool& operator=(const HostKvPool&) = delete;
    ~HostKvPool() override;

    [[nodiscard]] std::size_t block_bytes() const noexcept { return block_bytes_; }
    [[nodiscard]] std::size_t capacity_bytes() const;
    [[nodiscard]] std::size_t used_bytes() const;
    [[nodiscard]] std::size_t entries() const;
    [[nodiscard]] std::size_t objects() const;
    /// Entries dropped for room (any owner) and objects found already stored, over the pool's life.
    [[nodiscard]] std::uint64_t evictions() const;
    [[nodiscard]] std::uint64_t shared_objects() const;

    [[nodiscard]] OwnerId add_owner();
    /// Entries of `owner` the pool dropped to make room since the last call.
    [[nodiscard]] std::vector<EntryId> take_dropped(OwnerId owner);

    // ---- writing (zero copy: the program's copies go straight into the blocks) ----

    struct Placement {
        ObjectId object = 0;
        bool created    = false; ///< false: the object was already stored (or being written)
    };
    /// The object for `ref`, existing or new, held for the caller until `add_entry` takes it over
    /// or `release_objects` gives it back. A new one needs `ceil(ref.bytes / block_bytes)` blocks,
    /// made by dropping the least recently used entries no lease holds. Null when there is no room.
    [[nodiscard]] std::optional<Placement> acquire(const kvtier::ObjectRef& ref, OwnerId writer);
    /// Host memory of block `index` of an object.
    [[nodiscard]] std::span<std::byte> block(ObjectId object, std::size_t index);
    [[nodiscard]] std::size_t block_count(ObjectId object) const;
    /// Gives back objects `acquire` handed out (a snapshot that failed half way). A new one is
    /// never completed, so entries that found it meanwhile cannot lease it.
    void release_objects(std::span<const Placement> placements);
    /// An entry over objects the caller acquired (their holds pass to it). `pinned` entries go
    /// last when the pool needs room.
    [[nodiscard]] EntryId add_entry(OwnerId owner, std::vector<ObjectId> objects, bool pinned = false);
    /// The copies into these objects are done.
    void mark_ready(std::span<const ObjectId> objects);
    /// The owner no longer indexes the entry (superseded, invalidated). Leases keep its objects.
    void drop_entry(EntryId entry);
    void touch(EntryId entry);

    /// Keeps an entry's objects for a reader. Empty when the entry is gone or one of its objects is
    /// still being written by another owner.
    class Lease {
    public:
        Lease() = default;
        Lease(const Lease&)            = delete;
        Lease& operator=(const Lease&) = delete;
        Lease(Lease&& other) noexcept : pool_(std::move(other.pool_)), entry_(other.entry_) { other.entry_ = 0; }
        Lease& operator=(Lease&& other) noexcept {
            if (this != &other) {
                reset();
                pool_  = std::move(other.pool_);
                entry_ = other.entry_;
                other.entry_ = 0;
            }
            return *this;
        }
        ~Lease() { reset(); }
        void reset() noexcept;
        [[nodiscard]] explicit operator bool() const noexcept { return entry_ != 0; }
        [[nodiscard]] EntryId entry() const noexcept { return entry_; }

    private:
        friend class HostKvPool;
        Lease(std::shared_ptr<HostKvPool> pool, EntryId entry) : pool_(std::move(pool)), entry_(entry) {}
        std::shared_ptr<HostKvPool> pool_;
        EntryId entry_ = 0;
    };
    [[nodiscard]] Lease lease(EntryId entry, OwnerId reader);
    /// The writer's own lease while its copies run: no readiness check.
    [[nodiscard]] Lease hold(EntryId entry);
    [[nodiscard]] bool alive(EntryId entry) const;
    /// Drops every entry of `owner` (a program going away).
    void drop_owner(OwnerId owner);

    // ---- kvtier::Backend: the same store for callers outside the engine (and tests) ----
    [[nodiscard]] kvtier::Caps caps() const override;
    bool register_host(void* base, std::size_t bytes) override;
    void contains(std::span<const kvtier::ObjectRef> objects, std::span<std::uint8_t> present) override;
    [[nodiscard]] kvtier::Ticket put(std::span<const kvtier::ObjectRef> objects,
                                     std::span<const kvtier::Span> spans) override;
    [[nodiscard]] kvtier::Ticket get(std::span<const kvtier::ObjectRef> objects,
                                     std::span<const kvtier::Span> spans) override;
    [[nodiscard]] int poll(kvtier::Ticket ticket) override;
    void erase(std::span<const kvtier::Key> keys) override;
    void set_listener(kvtier::Listener listener) override;

private:
    struct Arena {
        std::shared_ptr<std::byte> memory;
        std::size_t blocks = 0;
    };
    struct Object {
        kvtier::ObjectRef ref;
        std::vector<std::uint32_t> blocks;
        std::uint32_t holds = 0; ///< entries and pending acquires that hold it
        OwnerId writer      = kNoOwner;
        bool ready          = false;
        bool poisoned       = false; ///< its writer gave up before the bytes were complete
        bool live           = false;
    };
    struct Entry {
        OwnerId owner = kNoOwner;
        std::vector<ObjectId> objects;
        std::uint64_t last_use = 0;
        std::uint32_t leases   = 0;
        bool pinned            = false;
        bool alive             = true;
    };

    HostKvPool(std::size_t block_bytes) : block_bytes_(block_bytes) {}
    void add_arena(Arena arena);
    [[nodiscard]] std::byte* block_address(std::uint32_t block) const;
    bool make_room(std::size_t blocks, std::vector<kvtier::ObjectRef>& removed);
    void unhold(ObjectId object, std::vector<kvtier::ObjectRef>& removed);
    void retire_entry(EntryId id, std::vector<kvtier::ObjectRef>& removed);
    void release_lease(EntryId entry) noexcept;
    void notify(std::vector<kvtier::ObjectRef>& removed, bool stored);

    const std::size_t block_bytes_;
    mutable std::mutex mutex_;
    std::vector<Arena> arenas_;
    std::vector<std::size_t> arena_first_; ///< global block id of each arena's first block
    std::vector<std::uint32_t> free_blocks_;
    std::size_t total_blocks_ = 0;
    std::vector<Object> objects_;
    std::vector<ObjectId> free_objects_;
    std::unordered_map<kvtier::Key, ObjectId, kvtier::KeyHash> by_key_;
    std::unordered_map<EntryId, Entry> entries_;
    std::unordered_map<kvtier::Key, EntryId, kvtier::KeyHash> external_; ///< Backend::put objects
    std::unordered_map<OwnerId, std::vector<EntryId>> dropped_;
    EntryId next_entry_  = 1;
    OwnerId next_owner_  = 1;
    std::uint64_t clock_ = 0;
    std::uint64_t evictions_ = 0, shared_objects_ = 0;
    kvtier::Ticket next_ticket_ = 1;
    kvtier::Listener listener_;
};

/// A back end in a shared library (api/kv_tier.h's C table) behind the C++ interface.
class CBackend final : public kvtier::Backend {
public:
    explicit CBackend(const sinfer_kvtier_v1* table);
    ~CBackend() override;
    [[nodiscard]] kvtier::Caps caps() const override;
    bool register_host(void* base, std::size_t bytes) override;
    void contains(std::span<const kvtier::ObjectRef> objects, std::span<std::uint8_t> present) override;
    [[nodiscard]] kvtier::Ticket put(std::span<const kvtier::ObjectRef> objects,
                                     std::span<const kvtier::Span> spans) override;
    [[nodiscard]] kvtier::Ticket get(std::span<const kvtier::ObjectRef> objects,
                                     std::span<const kvtier::Span> spans) override;
    [[nodiscard]] int poll(kvtier::Ticket ticket) override;
    void erase(std::span<const kvtier::Key> keys) override;
    void set_listener(kvtier::Listener listener) override;

private:
    static void forward(void* context, const sinfer_kvtier_object* objects, std::size_t count, int stored);
    const sinfer_kvtier_v1* table_;
    std::unique_ptr<kvtier::Listener> listener_;
};

/// The in-process pool as a C table, so a test (or a library that wraps the engine's own tier)
/// can drive it through the same ABI an external back end exports.
[[nodiscard]] sinfer_kvtier_v1 host_pool_c_table(HostKvPool* pool);

} // namespace sinfer::family::detail
