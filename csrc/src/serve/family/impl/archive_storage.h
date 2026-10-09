// Host storage for archived prefix images (the prefix cache's `ArchivedSequence`).
#pragma once

#include <cuda_runtime_api.h>

#include <cstddef>
#include <iterator>
#include <map>
#include <memory>
#include <mutex>
#include <optional>

namespace sinfer::family::detail {

/// The host bytes one archived image lives in.
struct ArchiveBlock {
    std::byte* data  = nullptr;
    std::size_t size = 0;
    /// Page-locked: copies to and from it stay asynchronous on the program's stream, which
    /// orders them against every other use of the image, so archiving waits on nothing.
    bool pinned      = false;
    /// Returns the range to its arena, or frees the heap buffer.
    std::shared_ptr<void> owner;
};

inline constexpr std::size_t kArchiveAlignment = 256;

[[nodiscard]] constexpr std::size_t archive_aligned(std::size_t bytes) noexcept {
    return (bytes + kArchiveAlignment - 1) & ~(kArchiveAlignment - 1);
}

/// A heap block, for when no page-locked range fits. Not zero-filled: the copies write every
/// byte. Copies into pageable memory return only once done.
[[nodiscard]] inline ArchiveBlock heap_archive_block(std::size_t bytes) {
    std::shared_ptr<std::byte[]> buffer(new std::byte[bytes]);
    std::byte* data = buffer.get();
    return ArchiveBlock{data, bytes, false, std::shared_ptr<void>(std::move(buffer), data)};
}

/// One page-locked host allocation that archived images are carved from, first fit.
///
/// Images used to live in zero-filled pageable vectors, copied tensor by tensor with a stream
/// sync after each. On a DGX Spark that is ~180 MB for a Qwen3.8-27B lane (its recurrent state
/// twice, current and checkpoint, and the KV pages), paid on the host with the GPU idle at
/// every admission onto a lane that held a finished request.
///
/// A freed range is handed out again only to a later archive, whose copies are enqueued after
/// every copy that read or wrote the range before, on the same stream; the host never reads
/// the bytes. Nothing here therefore synchronizes.
class PinnedArchiveArena : public std::enable_shared_from_this<PinnedArchiveArena> {
public:
    /// Null when the host cannot page-lock `capacity` bytes; images then use the heap.
    [[nodiscard]] static std::shared_ptr<PinnedArchiveArena> create(std::size_t capacity) noexcept {
        void* base = nullptr;
        if (capacity == 0 || cudaHostAlloc(&base, capacity, cudaHostAllocDefault) != cudaSuccess) {
            (void)cudaGetLastError();
            return nullptr;
        }
        try {
            return std::shared_ptr<PinnedArchiveArena>(
                new PinnedArchiveArena(static_cast<std::byte*>(base), capacity));
        } catch (...) {
            (void)cudaFreeHost(base);
            return nullptr;
        }
    }

    PinnedArchiveArena(const PinnedArchiveArena&)            = delete;
    PinnedArchiveArena& operator=(const PinnedArchiveArena&) = delete;
    ~PinnedArchiveArena() { (void)cudaFreeHost(base_); }

    [[nodiscard]] std::optional<ArchiveBlock> allocate(std::size_t bytes) {
        const std::size_t size = archive_aligned(bytes == 0 ? 1 : bytes);
        std::lock_guard lock(mutex_);
        for (auto it = free_.begin(); it != free_.end(); ++it) {
            if (it->second < size) { continue; }
            const std::size_t offset = it->first;
            const std::size_t rest   = it->second - size;
            free_.erase(it);
            if (rest != 0) { free_.emplace(offset + size, rest); }
            std::byte* data = base_ + offset;
            std::shared_ptr<void> owner(
                data, [arena = shared_from_this(), offset, size](void*) { arena->release(offset, size); });
            return ArchiveBlock{data, bytes, true, std::move(owner)};
        }
        return std::nullopt;
    }

private:
    PinnedArchiveArena(std::byte* base, std::size_t capacity) : base_(base) {
        free_.emplace(0, capacity);
    }

    void release(std::size_t offset, std::size_t size) noexcept {
        std::lock_guard lock(mutex_);
        auto [it, inserted] = free_.emplace(offset, size);
        if (!inserted) { return; }
        if (auto next = std::next(it); next != free_.end() && offset + size == next->first) {
            it->second += next->second;
            free_.erase(next);
        }
        if (it != free_.begin()) {
            if (auto previous = std::prev(it); previous->first + previous->second == it->first) {
                previous->second += it->second;
                free_.erase(it);
            }
        }
    }

    std::byte* base_;
    std::mutex mutex_;
    std::map<std::size_t, std::size_t> free_; ///< offset -> bytes
};

} // namespace sinfer::family::detail
