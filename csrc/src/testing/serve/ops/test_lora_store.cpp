// The adapter store: what a bank is, which device it lives on, and who may write it.
//
// The suite had no LoRA test at all, which is how two defects survived. One was
// that loading an adapter from a thread bound to another device killed the
// engine: every upload resolves its destination against the current device, and
// an HTTP handler is on whatever device it started on. The other was that the
// store assumed one device per engine, so a pipeline -- whose stages build
// concurrently, one thread per card -- could not have one at all.
//
// These run on real devices. With one they cover the single-device contract;
// the multi-device cases skip. Nothing here needs a model or a checkpoint.

#include "api/ops/lora_store.h"
#include "core/device.h"
#include "core/engine_context.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <iostream>
#include <thread>
#include <vector>

namespace {

int failures = 0;

void check(bool condition, const char* what) {
    if (!condition) {
        std::cerr << "FAIL: " << what << '\n';
        ++failures;
    }
}

int device_count() {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess) { return 0; }
    return count;
}

/// A bank's slot, read back from the device it lives on.
std::vector<std::uint16_t> read_slot_a(const sinfer::ops::LoraBank& bank, std::int32_t slot,
                                       int device) {
    const sinfer::ScopedDevice on_bank(device);
    std::vector<std::uint16_t> out(static_cast<std::size_t>(bank.a_stride));
    const auto* source = static_cast<const std::uint16_t*>(bank.a) +
                         static_cast<std::size_t>(slot) * bank.a_stride;
    if (cudaMemcpy(out.data(), source, out.size() * sizeof(std::uint16_t),
                   cudaMemcpyDeviceToHost) != cudaSuccess) {
        out.assign(out.size(), 0xFFFF);
    }
    return out;
}

/// One store standing in for one stage: two layers, one module each.
void build(sinfer::ops::LoraStore& store, const void* key, std::int32_t first_layer) {
    store.configure(/*slots=*/2, /*max_rank=*/8, /*max_tokens=*/16);
    store.register_module(first_layer, "down_proj",
                          sinfer::ops::LoraStore::ModuleBinding{key, 4, 4, 4});
    store.register_module(first_layer + 1, "down_proj",
                          sinfer::ops::LoraStore::ModuleBinding{key, 5, 4, 4});
    store.ensure_banks();
}

/// A is [rank,in] and B is [out,rank]; the store folds alpha/r into A.
void write(sinfer::ops::LoraStore& store, std::int32_t layer, std::int32_t slot, float scale) {
    const std::vector<std::uint16_t> a(static_cast<std::size_t>(2) * 4, 0x3F80); // bf16 1.0
    const std::vector<std::uint16_t> b(static_cast<std::size_t>(4) * 2, 0x3F80);
    store.set_module_slot(layer, "down_proj", slot, a, b, /*rank=*/2, /*in=*/4, /*out=*/4, scale);
}

void one_device() {
    sinfer::ops::LoraStoreSet set;
    int device = 0;
    check(cudaGetDevice(&device) == cudaSuccess, "cudaGetDevice");
    sinfer::ops::LoraStore& store = set.for_device(device);
    // A key needs only to be a distinct address; nothing dereferences it.
    const auto key = reinterpret_cast<const void*>(0x1000);
    build(store, key, /*first_layer=*/0);

    check(store.device() == device, "the store learns the device it was built on");
    check(store.has_bindings(), "the directory is not empty");
    check(store.covers_layer(0) && store.covers_layer(1), "both registered layers are covered");
    check(!store.covers_layer(7), "an unregistered layer is not covered");
    check(set.devices().size() == 1, "the set holds exactly the one device");
    check(set.peek(device) == &store, "peek returns the store that was created");

    const sinfer::ops::LoraBank* bank = store.find(key, 4);
    check(bank != nullptr, "layer 0's bank exists");
    check(store.find(key, 5) != nullptr, "layer 1's bank exists on the same key, another port");
    check(store.find(key, 6) == nullptr, "an unregistered port has no bank");
    if (bank == nullptr) { return; }

    write(store, /*layer=*/0, /*slot=*/1, /*scale=*/2.0F);
    const auto written = read_slot_a(*bank, 1, device);
    check(written.size() == static_cast<std::size_t>(bank->a_stride), "slot A is one stride wide");
    check(written[0] == 0x4000, "alpha/r is folded into A once (1.0 * 2 = 2.0)");
    bool tail_is_zero = true;
    for (std::size_t i = 2 * 4; i < written.size(); ++i) { tail_is_zero &= written[i] == 0; }
    check(tail_is_zero, "rows past the adapter's rank stay zero");

    // Unloading scrubs the slot rather than freeing it, so a request still in
    // flight adds nothing instead of reading another adapter's weights.
    store.clear_slot(1);
    const auto cleared = read_slot_a(*bank, 1, device);
    bool all_zero = true;
    for (const std::uint16_t value : cleared) { all_zero &= value == 0; }
    check(all_zero, "clear_slot zeroes the slot");
}

/// The defect that killed the engine: every write below happens on a thread bound
/// to a different device than the store.
void write_from_another_device(int other) {
    sinfer::ops::LoraStoreSet set;
    int home = 0;
    check(cudaGetDevice(&home) == cudaSuccess, "cudaGetDevice");
    sinfer::ops::LoraStore& store = set.for_device(home);
    const auto key = reinterpret_cast<const void*>(0x2000);
    build(store, key, /*first_layer=*/0);
    const sinfer::ops::LoraBank* bank = store.find(key, 4);
    check(bank != nullptr, "the bank exists");
    if (bank == nullptr) { return; }

    bool threw = false;
    std::thread visitor([&] {
        // Bound elsewhere, exactly as an HTTP handler thread is.
        if (cudaSetDevice(other) != cudaSuccess) { return; }
        try {
            write(store, /*layer=*/0, /*slot=*/0, /*scale=*/1.0F);
            store.clear_slot(1);
        } catch (...) { threw = true; }
    });
    visitor.join();
    check(!threw, "an upload from a thread on another device does not throw");
    const auto written = read_slot_a(*bank, 0, home);
    check(written[0] == 0x3F80, "the upload landed on the store's own device");
}

/// Two stores on two devices, which is what a pipeline's stages are.
void two_devices(int second) {
    sinfer::ops::LoraStoreSet set;
    sinfer::ops::LoraStore& first_store = set.for_device(0);
    const auto first_key = reinterpret_cast<const void*>(0x3000);
    {
        const sinfer::ScopedDevice on_first(0);
        build(first_store, first_key, /*first_layer=*/0);
    }
    sinfer::ops::LoraStore& second_store = set.for_device(second);
    const auto second_key = reinterpret_cast<const void*>(0x4000);
    {
        const sinfer::ScopedDevice on_second(second);
        build(second_store, second_key, /*first_layer=*/2);
    }

    check(&first_store != &second_store, "each device has its own store");
    check(first_store.device() == 0, "stage 0's store is on device 0");
    check(second_store.device() == second, "stage 1's store is on its own device");
    check(set.devices().size() == 2, "the set reports both devices");

    // Layer routing is what sends an adapter's module to the stage holding it.
    check(first_store.covers_layer(0) && !first_store.covers_layer(2),
          "stage 0 claims its own layers only");
    check(second_store.covers_layer(2) && !second_store.covers_layer(0),
          "stage 1 claims its own layers only");

    // Writing both stores from one thread is what loading an adapter does.
    write(first_store, /*layer=*/0, /*slot=*/0, /*scale=*/1.0F);
    write(second_store, /*layer=*/2, /*slot=*/0, /*scale=*/1.0F);
    const auto* first_bank  = first_store.find(first_key, 4);
    const auto* second_bank = second_store.find(second_key, 4);
    check(first_bank != nullptr && second_bank != nullptr, "both banks exist");
    if (first_bank == nullptr || second_bank == nullptr) { return; }
    check(read_slot_a(*first_bank, 0, 0)[0] == 0x3F80, "stage 0's adapter landed on device 0");
    check(read_slot_a(*second_bank, 0, second)[0] == 0x3F80, "stage 1's adapter landed on its own");
}

} // namespace

int main() {
    const int count = device_count();
    if (count < 1) {
        std::cerr << "no CUDA device; skipping\n";
        return 77;
    }
    one_device();
    if (count >= 2) {
        write_from_another_device(1);
        two_devices(1);
    } else {
        std::cerr << "only one device; the multi-device cases did not run\n";
    }
    if (failures != 0) {
        std::cerr << failures << " check(s) failed\n";
        return 1;
    }
    std::cout << "lora store: ok (" << count << " device(s))\n";
    return 0;
}
