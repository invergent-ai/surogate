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
    std::atomic<unsigned> wakes{0}, sleeps{0};
    std::atomic<std::size_t> active{0}, resumable{0};
    std::function<void()> on_wake, on_sleep;
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
    void shrink_kv() override {}
    std::size_t active_requests() const override { return active; }
    std::size_t resumable_requests() const override { return resumable; }
    std::size_t resident_bytes(int = -1) const override { return 100; }
    std::vector<int> devices() const override { return {0}; }
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

int main() {
    setenv("SUROGATE_MM_KEEPWARM_MS", "0", 1);
    setenv("SUROGATE_MM_PREEMPT_AFTER_MS", "10", 1);
    setenv("SUROGATE_MM_MIN_DWELL_MS", "0", 1);
    blocked_admission();
    transfer_does_not_block_waiters();
    cancellation_during_eviction();
    wake_failure_and_parked_resume();
    std::cout << "model wake deadline, cancellation, transfer, recovery and resume checks passed\n";
}
