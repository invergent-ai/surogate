#pragma once

#include "api/types.h"

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

namespace sinfer::serve {

// A slot may be rewritten only after its last admitted request releases it.
// Readers of other slots continue while an update drains its own requests.
class LoraSlots {
    using Clock = std::chrono::steady_clock;
    struct Entry {
        std::int32_t slot;
        std::size_t readers = 0;
        bool updating = false;
    };
    struct State {
        std::mutex mutex;
        std::timed_mutex writer;
        std::condition_variable changed;
        std::map<std::string, std::shared_ptr<Entry>> names;
        std::vector<std::int32_t> free;
    };
    struct Reader {
        std::shared_ptr<State> state;
        std::shared_ptr<Entry> entry;
        ~Reader() {
            {
                std::lock_guard lock(state->mutex);
                --entry->readers;
            }
            state->changed.notify_all();
        }
    };

public:
    struct Lease {
        std::int32_t slot = -1;
        std::shared_ptr<void> lifetime;
    };

    class Update {
    public:
        Update(const Update&) = delete;
        Update& operator=(const Update&) = delete;
        ~Update() {
            if (committed_) { return; }
            {
                std::lock_guard lock(state_->mutex);
                entry_->updating = false;
                if (!existing_) { state_->free.push_back(entry_->slot); }
            }
            state_->changed.notify_all();
        }
        [[nodiscard]] std::int32_t slot() const { return entry_->slot; }

        // Publish after every write and cache invalidation has completed. A
        // failed device upload removes the name instead of exposing partial data.
        void commit(bool remove = false) {
            {
                std::lock_guard lock(state_->mutex);
                if (remove) {
                    state_->names.erase(name_);
                    state_->free.push_back(entry_->slot);
                } else {
                    state_->names[name_] = entry_;
                }
                entry_->updating = false;
                committed_ = true;
            }
            state_->changed.notify_all();
        }

    private:
        friend class LoraSlots;
        Update(std::shared_ptr<State> state, std::string name, Clock::time_point deadline,
               bool must_exist)
            : state_(std::move(state)), name_(std::move(name)), writer_(state_->writer, std::defer_lock) {
            if (name_.empty()) { throw std::invalid_argument("an adapter name is required"); }
            if (!writer_.try_lock_until(deadline)) {
                throw RequestError(RequestErrorKind::QueueTimeout, "timed out waiting for an adapter update");
            }
            std::unique_lock lock(state_->mutex);
            const auto found = state_->names.find(name_);
            existing_ = found != state_->names.end();
            if (existing_) {
                entry_ = found->second;
                entry_->updating = true;
                if (!state_->changed.wait_until(lock, deadline, [&] { return entry_->readers == 0; })) {
                    entry_->updating = false;
                    state_->changed.notify_all();
                    throw RequestError(RequestErrorKind::QueueTimeout,
                                       "timed out waiting for active requests using adapter '" + name_ + "'");
                }
            } else {
                if (must_exist) { throw std::invalid_argument("adapter '" + name_ + "' is not loaded"); }
                if (state_->free.empty()) {
                    throw std::invalid_argument("all adapter slots are in use; unload one or increase --max-loras");
                }
                entry_ = std::make_shared<Entry>(state_->free.front());
                state_->free.erase(state_->free.begin());
                entry_->updating = true;
            }
        }
        std::shared_ptr<State> state_;
        std::string name_;
        std::unique_lock<std::timed_mutex> writer_;
        std::shared_ptr<Entry> entry_;
        bool existing_ = false;
        bool committed_ = false;
    };

    explicit LoraSlots(std::uint32_t capacity) : state_(std::make_shared<State>()) {
        for (std::uint32_t slot = 0; slot < capacity; ++slot) {
            state_->free.push_back(static_cast<std::int32_t>(slot));
        }
    }

    [[nodiscard]] std::int32_t find(const std::string& name) const {
        std::lock_guard lock(state_->mutex);
        const auto found = state_->names.find(name);
        return found == state_->names.end() ? -1 : found->second->slot;
    }
    [[nodiscard]] std::vector<std::string> names() const {
        std::lock_guard lock(state_->mutex);
        std::vector<std::string> result;
        for (const auto& [name, entry] : state_->names) { result.push_back(name); }
        return result;
    }
    [[nodiscard]] Lease acquire(const std::string& name, Clock::time_point deadline,
                                const std::function<bool()>& cancelled = {}) const {
        if (name.empty()) { return {}; }
        std::unique_lock lock(state_->mutex);
        for (;;) {
            if (cancelled) {
                lock.unlock();
                const bool stop = cancelled();
                lock.lock();
                if (stop) {
                    throw RequestError(RequestErrorKind::Cancelled, "cancelled while waiting for an adapter update");
                }
            }
            const auto found = state_->names.find(name);
            if (found == state_->names.end()) {
                throw RequestError(RequestErrorKind::Unavailable, "adapter '" + name + "' is no longer loaded");
            }
            const auto entry = found->second;
            if (!entry->updating) {
                auto reader = std::make_shared<Reader>(state_, entry);
                ++entry->readers;
                return {entry->slot, std::move(reader)};
            }
            if (Clock::now() >= deadline) {
                throw RequestError(RequestErrorKind::QueueTimeout, "timed out waiting for adapter '" + name + "'");
            }
            state_->changed.wait_until(lock, std::min(deadline, Clock::now() + std::chrono::milliseconds(10)));
        }
    }
    [[nodiscard]] Update update(const std::string& name, Clock::time_point deadline, bool must_exist = false) {
        return Update(state_, name, deadline, must_exist);
    }

    // Shared GRPO publishes slot zero only after draining the entire service.
    void reset_to(const std::string& name, std::int32_t slot) {
        std::lock_guard writer(state_->writer);
        std::lock_guard lock(state_->mutex);
        for (const auto& [old_name, entry] : state_->names) {
            if (entry->readers != 0) { throw std::logic_error("adapter reset requires drained requests"); }
        }
        state_->names.clear();
        state_->names.emplace(name, std::make_shared<Entry>(slot));
        state_->free.clear();
    }

private:
    std::shared_ptr<State> state_;
};

} // namespace sinfer::serve
