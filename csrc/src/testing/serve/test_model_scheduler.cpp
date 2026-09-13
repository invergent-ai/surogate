#include "serve/model_scheduler.h"

#include <cassert>
#include <future>
#include <iostream>

using namespace sinfer;
using namespace sinfer::serve;
using namespace std::chrono_literals;
using Clock = std::chrono::steady_clock;

struct Model final : ScheduledModel {
    std::atomic<bool> asleep{true};
    std::atomic<unsigned> wakes{0}, sleeps{0}, shrinks{0};
    std::atomic<std::size_t> active{0}, resumable{0};
    std::function<void()> on_wake, on_sleep, on_shrink;
    std::map<int, std::size_t> footprint{{0, 100}};
    void prepare_sleep_backup() override {}
    void sleep(bool = false) override {
        ++sleeps;
        if (on_sleep) { on_sleep(); }
        asleep = true;
    }
    void wake_up() override {
        ++wakes;
        if (on_wake) { on_wake(); }
        asleep = false;
    }
    bool is_sleeping() const override { return asleep; }
    void shrink_kv() override { ++shrinks; if (on_shrink) { on_shrink(); } }
    std::size_t active_requests() const override { return active; }
    std::size_t resumable_requests() const override { return resumable; }
    std::size_t resident_bytes(int device = -1) const override {
        if (device >= 0) {
            const auto found = footprint.find(device);
            return found == footprint.end() ? 0 : found->second;
        }
        std::size_t total = 0;
        for (const auto& [id, bytes] : footprint) { total += bytes; }
        return total;
    }
    std::vector<int> devices() const override {
        std::vector<int> out;
        for (const auto& [id, bytes] : footprint) { out.push_back(id); }
        return out;
    }
};

template<class F> void expect(RequestErrorKind kind, F&& operation) {
    try { operation(); assert(false && "expected a request error"); }
    catch (const RequestError& error) { assert(error.kind() == kind); }
}

void blocked_admission() {
    Model target, busy;
    busy.asleep = false;
    busy.active = busy.resumable = 1;
    target.active = 1; // initial wake admission is not resumable work
    ModelScheduler scheduler({{"target", &target}, {"busy", &busy, 2}}, {{0, 100}});
    const auto started = Clock::now();
    expect(RequestErrorKind::QueueTimeout, [&] {
        scheduler.ensure_awake(&target, {.deadline = started + 40ms});
    });
    assert(Clock::now() - started < 500ms);
    expect(RequestErrorKind::Cancelled, [&] {
        scheduler.ensure_awake(&target, {
            .deadline = Clock::now() + 2s,
            .cancellation = CancellationView([begin = Clock::now()] { return Clock::now() - begin > 20ms; })});
    });
    assert(target.wakes == 0 && busy.sleeps == 0);
    busy.active = busy.resumable = 0;
    busy.asleep = true;
    std::this_thread::sleep_for(600ms); // the resume ticker must not revive cancelled wake work
    assert(target.wakes == 0);
    scheduler.ensure_awake(&target, {.deadline = Clock::now() + 1s});
    assert(target.wakes == 1);
    scheduler.ensure_awake(&target, {.deadline = Clock::now() + 1s});
    assert(target.wakes == 1);
}

void transfer_does_not_block_waiters() {
    Model transferring, other;
    std::promise<void> entered, release;
    auto released = release.get_future().share();
    transferring.on_wake = [&] { entered.set_value(); released.wait(); };
    ModelScheduler scheduler({{"copy", &transferring}, {"other", &other}}, {{0, 200}});
    auto first = std::async(std::launch::async, [&] {
        expect(RequestErrorKind::QueueTimeout, [&] {
            scheduler.ensure_awake(&transferring, {.deadline = Clock::now() + 100ms});
        });
    });
    assert(entered.get_future().wait_for(1s) == std::future_status::ready);
    expect(RequestErrorKind::QueueTimeout, [&] {
        scheduler.ensure_awake(&other, {.deadline = Clock::now() + 30ms});
    });
    const auto caller = std::this_thread::get_id();
    expect(RequestErrorKind::Cancelled, [&] {
        scheduler.ensure_awake(&other, {.deadline = Clock::now() + 1s,
            .cancellation = CancellationView([begin = Clock::now(), caller] {
                assert(std::this_thread::get_id() == caller);
                return Clock::now() - begin > 20ms;
            })});
    });
    // The initiating request also leaves while its own restore is still in progress.
    assert(first.wait_for(500ms) == std::future_status::ready);
    first.get();
    assert(other.wakes == 0);
    release.set_value();
    scheduler.ensure_awake(&transferring, {.deadline = Clock::now() + 1s});
    assert(transferring.wakes == 1);
    scheduler.ensure_awake(&other, {.deadline = Clock::now() + 1s});
    assert(other.wakes == 1);
}

void cancellation_during_eviction() {
    Model target, victim;
    victim.asleep = false;
    std::atomic<bool> cancelled{false};
    std::promise<void> entered, release;
    auto released = release.get_future().share();
    victim.on_sleep = [&] { entered.set_value(); released.wait(); };
    ModelScheduler scheduler({{"target", &target}, {"victim", &victim}}, {{0, 100}});
    auto waiter = std::async(std::launch::async, [&] {
        expect(RequestErrorKind::Cancelled, [&] {
            scheduler.ensure_awake(&target, {.deadline = Clock::now() + 2s,
                .cancellation = CancellationView([&] { return cancelled.load(); })});
        });
    });
    assert(entered.get_future().wait_for(1s) == std::future_status::ready);
    cancelled = true;
    assert(waiter.wait_for(500ms) == std::future_status::ready);
    waiter.get();
    release.set_value();
    std::this_thread::sleep_for(600ms);
    assert(target.wakes == 0 && victim.sleeps == 1);
    scheduler.ensure_awake(&target, {.deadline = Clock::now() + 1s});
    assert(target.wakes == 1);
}

void wake_failure_and_parked_resume() {
    Model target;
    target.on_wake = [] { throw std::runtime_error("restore failed"); };
    ModelScheduler scheduler({{"target", &target}}, {{0, 100}});
    try {
        scheduler.ensure_awake(&target, {.deadline = Clock::now() + 1s});
        assert(false);
    } catch (const std::runtime_error& error) { assert(std::string(error.what()) == "restore failed"); }
    target.on_wake = {};
    target.active = target.resumable = 1;
    const auto until = Clock::now() + 2s;
    while (target.asleep && Clock::now() < until) { std::this_thread::sleep_for(10ms); }
    assert(!target.asleep && target.wakes == 2);
}

void infeasible_evictions_leave_neighbors_awake() {
    {
        Model target, idle;
        target.footprint = {{0, 300}};
        idle.asleep = false;
        ModelScheduler scheduler({{"target", &target}, {"idle", &idle}}, {{0, 200}});
        const auto start = Clock::now();
        try {
            scheduler.ensure_awake(&target, {.deadline = start + 1s});
            assert(false);
        } catch (const RequestError& error) {
            assert(error.kind() == RequestErrorKind::Unavailable);
            const std::string message = error.what();
            assert(message.find("device 0") != std::string::npos);
            assert(message.find("300") != std::string::npos && message.find("200") != std::string::npos);
        }
        assert(Clock::now() - start < 500ms);
        assert(idle.sleeps == 0 && idle.shrinks == 0 && !idle.asleep);
    }
    for (bool multi_device : {false, true}) {
        Model target, idle, busy;
        idle.asleep = busy.asleep = false;
        busy.active = 1;
        std::map<int, std::size_t> budget{{0, 200}};
        if (multi_device) {
            target.footprint = {{0, 100}, {1, 100}};
            busy.footprint = {{1, 100}};
            budget = {{0, 100}, {1, 100}};
        } else {
            target.footprint = {{0, 150}};
            idle.footprint = {{0, 40}};
        }
        ModelScheduler scheduler({{"target", &target}, {"idle", &idle}, {"busy", &busy, 2}}, budget);
        expect(RequestErrorKind::QueueTimeout, [&] {
            scheduler.ensure_awake(&target, {.deadline = Clock::now() + 50ms});
        });
        assert(target.wakes == 0 && busy.sleeps == 0);
        assert(idle.sleeps == 0 && idle.shrinks == 0 && !idle.asleep);
    }
}

void feasible_reclamation_uses_only_needed_victims() {
    Model target, first, second, busy;
    first.asleep = second.asleep = busy.asleep = false;
    first.footprint = second.footprint = {{0, 60}};
    busy.active = 1;
    ModelScheduler scheduler({{"target", &target}, {"first", &first}, {"second", &second}, {"busy", &busy}},
                             {{0, 200}});
    scheduler.ensure_awake(&target, {.deadline = Clock::now() + 1s});
    assert(target.wakes == 1 && first.sleeps == 1 && second.sleeps == 1 && busy.sleeps == 0);
    Model small, cached;
    small.footprint = {{0, 80}};
    cached.footprint = {{0, 180}};
    cached.asleep = false;
    cached.on_shrink = [&] { cached.footprint = {{0, 120}}; };
    ModelScheduler shrinker({{"small", &small}, {"cached", &cached}}, {{0, 200}});
    shrinker.ensure_awake(&small, {.deadline = Clock::now() + 1s});
    assert(small.wakes == 1 && cached.shrinks == 1 && cached.sleeps == 0);
}

int main() {
    setenv("SUROGATE_MM_KEEPWARM_MS", "0", 1);
    setenv("SUROGATE_MM_PREEMPT_AFTER_MS", "10", 1);
    setenv("SUROGATE_MM_MIN_DWELL_MS", "0", 1);
    blocked_admission();
    transfer_does_not_block_waiters();
    cancellation_during_eviction();
    wake_failure_and_parked_resume();
    infeasible_evictions_leave_neighbors_awake();
    feasible_reclamation_uses_only_needed_victims();
    std::cout << "model wake deadline, cancellation, transfer, recovery and resume checks passed\n";
}
