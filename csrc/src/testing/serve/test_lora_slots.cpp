#include "serve/lora_slots.h"

#include <atomic>
#include <cassert>
#include <chrono>
#include <future>
#include <iostream>
#include <thread>

using sinfer::RequestError;
using sinfer::RequestErrorKind;
using sinfer::serve::LoraSlots;
using Clock = std::chrono::steady_clock;
using namespace std::chrono_literals;

static auto deadline() { return Clock::now() + 3s; }

static void load(LoraSlots& slots, const std::string& name) {
    auto update = slots.update(name, deadline());
    update.commit();
}

static void wait_until_draining(LoraSlots& slots, const std::string& name) {
    const auto limit = deadline();
    while (Clock::now() < limit) {
        try {
            auto probe = slots.acquire(name, Clock::now() + 5ms);
        } catch (const RequestError& error) {
            assert(error.kind() == RequestErrorKind::QueueTimeout);
            return;
        }
        std::this_thread::yield();
    }
    assert(false && "update never blocked new readers");
}

static void replacement_preserves_running_requests() {
    LoraSlots slots(2);
    load(slots, "policy");
    load(slots, "other");
    auto running = slots.acquire("policy", deadline());
    const auto original_slot = running.slot;
    std::atomic<int> weights{1};
    std::promise<void> writing, publish;
    auto permit_publish = publish.get_future();
    auto writer = std::async(std::launch::async, [&] {
        auto update = slots.update("policy", deadline());
        assert(update.slot() == original_slot);
        weights = 2;
        writing.set_value();
        assert(permit_publish.wait_for(3s) == std::future_status::ready);
        update.commit();
    });
    wait_until_draining(slots, "policy");
    assert(weights == 1);
    assert(slots.acquire("other", deadline()).slot != original_slot);
    assert(slots.acquire("", deadline()).slot == -1);
    auto next = std::async(std::launch::async, [&] {
        auto reader = slots.acquire("policy", deadline());
        assert(reader.slot == original_slot);
        return weights.load();
    });
    assert(next.wait_for(10ms) == std::future_status::timeout);
    running.lifetime.reset();
    assert(writing.get_future().wait_for(3s) == std::future_status::ready);
    assert(next.wait_for(10ms) == std::future_status::timeout);
    publish.set_value();
    writer.get();
    assert(next.get() == 2);
}

static void unload_waits_before_recycling_slots() {
    LoraSlots slots(1);
    load(slots, "old");
    auto running = slots.acquire("old", deadline());
    auto writer = std::async(std::launch::async, [&] {
        auto update = slots.update("old", deadline(), true);
        update.commit(true);
    });
    wait_until_draining(slots, "old");
    assert(slots.find("old") == running.slot);
    running.lifetime.reset();
    writer.get();
    assert(slots.find("old") == -1);
    try {
        (void)slots.acquire("old", deadline());
        assert(false && "an unloaded adapter must never fall back to the base");
    } catch (const RequestError& error) {
        assert(error.kind() == RequestErrorKind::Unavailable);
    }
    load(slots, "new");
    assert(slots.acquire("new", deadline()).slot == 0);
}

static void timeout_and_cancel_leave_the_old_adapter_usable() {
    LoraSlots slots(1);
    load(slots, "policy");
    auto running = slots.acquire("policy", deadline());
    auto writer = std::async(std::launch::async, [&] {
        try {
            (void)slots.update("policy", Clock::now() + 80ms);
            assert(false && "a live slot must not be rewritten");
        } catch (const RequestError& error) {
            assert(error.kind() == RequestErrorKind::QueueTimeout);
        }
    });
    wait_until_draining(slots, "policy");
    try {
        (void)slots.acquire("policy", deadline(), [] { return true; });
        assert(false && "cancelled reader did not leave the wait");
    } catch (const RequestError& error) {
        assert(error.kind() == RequestErrorKind::Cancelled);
    }
    writer.get();
    assert(slots.acquire("policy", deadline(), [&] {
        assert(slots.find("policy") == running.slot);
        return false;
    }).slot == running.slot);
}

static void competing_updates_are_serialized() {
    LoraSlots slots(1);
    load(slots, "policy");
    std::future<void> second;
    std::atomic<int> writes{0};
    {
        auto first = slots.update("policy", deadline());
        second = std::async(std::launch::async, [&] {
            auto update = slots.update("policy", deadline());
            assert(writes == 1);
            writes = 2;
            update.commit();
        });
        assert(second.wait_for(10ms) == std::future_status::timeout);
        writes = 1;
        first.commit();
    }
    second.get();
    assert(writes == 2);
}

static void abandoned_update_does_not_lose_a_name_or_a_slot() {
    LoraSlots slots(1);
    { auto update = slots.update("not-published", deadline()); }
    assert(slots.names().empty());
    load(slots, "policy");
    { auto update = slots.update("policy", deadline()); }
    assert(slots.acquire("policy", deadline()).slot == 0);
    {
        auto failed_upload = slots.update("policy", deadline());
        failed_upload.commit(true);
    }
    assert(slots.names().empty());
    load(slots, "replacement");
    assert(slots.find("replacement") == 0);
}

int main() {
    replacement_preserves_running_requests();
    unload_waits_before_recycling_slots();
    timeout_and_cancel_leave_the_old_adapter_usable();
    competing_updates_are_serialized();
    abandoned_update_does_not_lose_a_name_or_a_slot();
    std::cout << "LoRA slot lifecycle tests passed\n";
}
