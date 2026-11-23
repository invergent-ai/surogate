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
#include "core/arena.h"
#include "core/device.h"
#include "core/engine_context.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>
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

    // A live trainer publishes compact GPU tensors into padded serving banks.
    const std::vector<std::uint16_t> ones(8, 0x3F80);
    sinfer::DeviceBuffer a(ones.size() * sizeof(std::uint16_t));
    sinfer::DeviceBuffer b(ones.size() * sizeof(std::uint16_t));
    a.copy_from_host(ones.data(), a.bytes);
    b.copy_from_host(ones.data(), b.bytes);
    sinfer::DeviceAdapterModule module{0, "down_proj", a.p, b.p, 2, 4, 4, 2.0F};
    store.set_device_module(0, module);
    check(read_slot_a(*bank, 0, device) == written, "GPU publication matches scaled host upload");
    std::vector<std::uint16_t> published_b(bank->b_stride);
    CUDA_CHECK(cudaMemcpy(published_b.data(), bank->b, published_b.size() * sizeof(std::uint16_t),
                          cudaMemcpyDeviceToHost));
    bool b_matches = true;
    for (std::size_t i = 0; i < published_b.size(); ++i) {
        b_matches &= published_b[i] == (i % 8 < 2 ? 0x3F80 : 0);
    }
    check(b_matches, "GPU B rows have the correct rank padding");
    module.in = 3;
    bool rejected = false;
    try { store.set_device_module(0, module); }
    catch (const std::exception&) { rejected = true; }
    check(rejected, "GPU publication rejects an incompatible adapter shape");
    check(read_slot_a(*bank, 0, device) == written, "rejected publication leaves the slot intact");
    check(read_slot_a(*bank, 1, device) == written, "GPU publication leaves other slots intact");

    try {
        store.set_module_slot(0, "unsupported_proj", 0, ones, ones, 2, 4, 4, 1);
        check(false, "unsupported adapter module must be refused");
    } catch (const std::invalid_argument& error) {
        const std::string message = error.what();
        check(message.find("unsupported_proj") != std::string::npos &&
                  message.find("layer 0") != std::string::npos &&
                  message.find("surogate merge") != std::string::npos,
              "unsupported adapter warning names its module, layer and remedy");
    }
    check(read_slot_a(*bank, 0, device) == written, "unsupported weights leave the current adapter intact");

    // A drained slot is zeroed while its bank allocation stays reusable.
    store.clear_slot(1);
    const auto cleared = read_slot_a(*bank, 1, device);
    bool all_zero = true;
    for (const std::uint16_t value : cleared) { all_zero &= value == 0; }
    check(all_zero, "clear_slot zeroes the slot");
}

/// The prefix caches key KV by (slot, KV generation). The fault this guards (11-10): an
/// in-place swap kept the slot number, so a prompt seen under the old adapter was served from
/// the old adapter's KV and scores after the swap. Every change of a slot's bytes must move its
/// generation to a value it never had, and leave the other slots' alone.
void kv_generations() {
    sinfer::ops::LoraStoreSet set;
    int device = 0;
    check(cudaGetDevice(&device) == cudaSuccess, "cudaGetDevice");
    sinfer::ops::LoraStore& store = set.for_device(device);
    const auto key = reinterpret_cast<const void*>(0x3000);
    build(store, key, /*first_layer=*/0);

    check(store.slot_kv_generation(0) == 0 && store.slot_kv_generation(1) == 0,
          "a slot never loaded has KV generation 0");
    check(store.slot_kv_generation(-1) == 0 && store.slot_kv_generation(2) == 0,
          "out-of-range slots (the base model) have generation 0");
    std::vector<std::uint64_t> seen;
    const auto moved = [&](std::int32_t slot, std::uint64_t before_slot, std::uint64_t before_clock,
                           const char* what) {
        const std::uint64_t now = store.slot_kv_generation(slot);
        bool fresh = now != 0 && now != before_slot;
        for (const auto value : seen) { fresh &= value != now; }
        seen.push_back(now);
        check(fresh, what);
        check(store.kv_generation_clock() > before_clock, "the clock moves with every change");
    };

    std::uint64_t clock = store.kv_generation_clock();
    write(store, /*layer=*/0, /*slot=*/1, /*scale=*/2.0F);
    moved(1, 0, clock, "a load moves the slot's generation");
    check(store.slot_kv_generation(0) == 0, "a load into slot 1 leaves slot 0's generation");

    // The in-place swap: the same slot, new bytes, at a round boundary.
    const std::uint64_t loaded = store.slot_kv_generation(1);
    const std::uint64_t slot0  = store.slot_kv_generation(0);
    clock                      = store.kv_generation_clock();
    const std::string module = "down_proj";
    const std::vector<std::uint16_t> a(static_cast<std::size_t>(2) * 4, 0x4000); // bf16 2.0
    const std::vector<std::uint16_t> b(static_cast<std::size_t>(4) * 2, 0x3F80);
    std::vector<sinfer::ops::LoraStore::StagedModule> modules{{0, &module, &a, &b, 2, 4, 4, 1.0F}};
    const auto staged = store.stage_modules(1, modules);
    check(store.slot_kv_generation(1) == loaded, "staging alone does not change what the slot holds");
    store.commit_staged(*staged);
    moved(1, loaded, clock, "an in-place swap moves the slot's generation");
    check(store.slot_kv_generation(0) == slot0, "an in-place swap of slot 1 leaves slot 0's generation");

    // Swapping the same bytes in again still moves it: the KV was computed between the swaps.
    const std::uint64_t swapped = store.slot_kv_generation(1);
    clock                       = store.kv_generation_clock();
    store.commit_staged(*store.stage_modules(1, modules));
    moved(1, swapped, clock, "a repeated in-place swap moves the slot's generation again");

    const std::uint64_t before_clear = store.slot_kv_generation(1);
    clock                            = store.kv_generation_clock();
    store.clear_slot(1);
    moved(1, before_clear, clock, "an unload moves the slot's generation");

    // A trainer-published module (shared GRPO adapters).
    const std::vector<std::uint16_t> ones(8, 0x3F80);
    sinfer::DeviceBuffer da(ones.size() * sizeof(std::uint16_t));
    sinfer::DeviceBuffer db(ones.size() * sizeof(std::uint16_t));
    da.copy_from_host(ones.data(), da.bytes);
    db.copy_from_host(ones.data(), db.bytes);
    clock = store.kv_generation_clock();
    store.set_device_module(0, sinfer::DeviceAdapterModule{0, "down_proj", da.p, db.p, 2, 4, 4, 2.0F});
    moved(0, slot0, clock, "a published module moves its slot's generation");

    // The engine-facing accessors read the bound context's store for the current device.
    check(sinfer::ops::lora_slot_kv_generation(-1) == 0, "the base model's generation is 0");
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

void speculative_columns() {
    sinfer::ops::EngineOpsContext context;
    sinfer::ops::bind_ops_context(&context);
    auto& store = sinfer::ops::lora_store_for_current_device();
    build(store, reinterpret_cast<const void*>(0x4000), 0);
    sinfer::ops::lora_set_active(true);
    sinfer::DeviceBuffer row_storage(3 * sizeof(std::int32_t));
    sinfer::DeviceBuffer column_storage(48 * sizeof(std::int32_t));
    sinfer::Tensor rows(row_storage.p, sinfer::DType::I32, {3});
    sinfer::Tensor storage(column_storage.p, sinfer::DType::I32, {48});
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    for (int width : {1, 4, 16}) {
        cudaGraph_t graph;
        cudaGraphExec_t executable;
        CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
        auto columns = sinfer::ops::speculative_lora_columns(rows, storage, width, 3, stream);
        CUDA_CHECK(cudaStreamEndCapture(stream, &graph));
        CUDA_CHECK(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0));
        for (const std::vector<std::int32_t> selected : {std::vector<std::int32_t>{-1, 0, 1}, {1, -1, 0}}) {
            row_storage.copy_from_host(selected.data(), row_storage.bytes);
            CUDA_CHECK(cudaGraphLaunch(executable, stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));
            std::vector<std::int32_t> actual(width * 3);
            CUDA_CHECK(cudaMemcpy(actual.data(), columns.data, columns.bytes(), cudaMemcpyDeviceToHost));
            for (int i = 0; i < width * 3; ++i) {
                check(actual[i] == selected[i / width], "speculative graph reads each row's current adapter");
            }
        }
        CUDA_CHECK(cudaGraphExecDestroy(executable));
        CUDA_CHECK(cudaGraphDestroy(graph));
    }
    {
        sinfer::ops::ScopedLoraColumns outer(rows);
        const auto* selected = sinfer::ops::lora_current_round().slots;
        {
            sinfer::ops::ScopedLoraColumns inner(rows.slice(0, 1, 1));
            check(sinfer::ops::lora_current_round().slots->data != selected->data,
                  "a subset head selects its own columns");
        }
        check(sinfer::ops::lora_current_round().slots == selected, "a subset head restores the enclosing selection");
    }
    check(!sinfer::ops::lora_current_round().valid(), "the completed round leaves no adapter selection");
    CUDA_CHECK(cudaStreamDestroy(stream));
    sinfer::ops::bind_ops_context(nullptr);
}

/// Per-bank rank (ModuleBinding::rank): a store at max_rank 8 holds a module at the default rank
/// (8) and an Inkling-expert-like fused pair at rank 2 (gate / up interleaved in one checkpoint B, one A).
/// The small banks are (slots, 2) wide; an adapter of rank 2 lands in them bit for bit through every write
/// path (host upload, in-place staging, device publication); rank 3 is refused for them only.
void mixed_ranks() {
    using Binding = sinfer::ops::LoraStore::ModuleBinding;
    sinfer::ops::LoraStoreSet set;
    int device = 0;
    check(cudaGetDevice(&device) == cudaSuccess, "cudaGetDevice");
    sinfer::ops::LoraStore& store = set.for_device(device);
    const auto key = reinterpret_cast<const void*>(0x3000);
    store.configure(/*slots=*/2, /*max_rank=*/8, /*max_tokens=*/16);
    store.register_module(0, "attn.big", Binding{key, 1, 6, 5});
    Binding gate{key, 2, 6, 3};
    gate.source_out = 6; gate.row_offset = 0; gate.row_group = 1; gate.row_stride = 2; gate.rank = 2;
    Binding up = gate;
    up.port = 3; up.row_offset = 1;
    store.register_module(0, "mlp.experts.0.gate_up_proj", gate);
    store.register_module(0, "mlp.experts.0.gate_up_proj", up);
    Binding gate_bank{key, 2, 6, 3};
    gate_bank.rank = 2;
    Binding up_bank{key, 3, 6, 3};
    up_bank.rank = 2;
    store.register_bank_table(key, {Binding{}, gate_bank, up_bank});
    store.ensure_banks();

    const sinfer::ops::LoraBank* big   = store.find(key, 1);
    const sinfer::ops::LoraBank* small = store.find(key, 2);
    check(big != nullptr && small != nullptr && store.find(key, 3) != nullptr, "mixed: all three banks exist");
    if (big == nullptr || small == nullptr) { return; }
    check(big->rank == 8 && big->a_stride == 8 * 6 && big->b_stride == 5 * 8, "mixed: default bank at max_rank");
    check(small->rank == 2 && small->a_stride == 2 * 6 && small->b_stride == 3 * 2, "mixed: small bank at its rank");
    const sinfer::ops::LoraBank* host = store.host_bank_table(key);
    check(host != nullptr && host[1].rank == 2 && host[2].rank == 2, "mixed: the host table carries the bank rank");

    // A [2, 6] and B [6, 2] with distinct bf16 values (1.0 + i / 128, exact), scale 1: the bank holds them as is.
    std::vector<std::uint16_t> a(12), b(12);
    for (int i = 0; i < 12; ++i) {
        a[i] = static_cast<std::uint16_t>(0x3F80 + i);
        b[i] = static_cast<std::uint16_t>(0x4000 + i);
    }
    const auto expect_small = [&](std::int32_t slot, const char* what) {
        const sinfer::ScopedDevice on(device);
        for (std::int32_t port = 2; port <= 3; ++port) {
            const sinfer::ops::LoraBank* bank = store.find(key, port);
            std::vector<std::uint16_t> got_a(bank->a_stride), got_b(bank->b_stride);
            CUDA_CHECK(cudaMemcpy(got_a.data(), static_cast<const std::uint16_t*>(bank->a) + slot * bank->a_stride,
                                  got_a.size() * 2, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(got_b.data(), static_cast<const std::uint16_t*>(bank->b) + slot * bank->b_stride,
                                  got_b.size() * 2, cudaMemcpyDeviceToHost));
            bool ok = got_a == a;
            for (int row = 0; row < 3; ++row) {
                for (int r = 0; r < 2; ++r) { ok &= got_b[row * 2 + r] == b[(2 * row + (port - 2)) * 2 + r]; }
            }
            check(ok, what);
            const sinfer::ops::LoraBank& mirror = store.host_bank_table(key)[port - 1];
            const auto* ha = static_cast<const std::uint16_t*>(mirror.a) + slot * mirror.a_stride;
            const auto* hb = static_cast<const std::uint16_t*>(mirror.b) + slot * mirror.b_stride;
            check(std::equal(got_a.begin(), got_a.end(), ha) && std::equal(got_b.begin(), got_b.end(), hb),
                  "mixed: the host mirror equals the device bank");
        }
    };
    store.set_module_slot(0, "mlp.experts.0.gate_up_proj", 0, a, b, 2, 6, 6, 1.0F);
    expect_small(0, "mixed: host upload lands bitwise in the rank-2 banks (gate even rows, up odd rows)");

    const std::string name = "mlp.experts.0.gate_up_proj";
    const auto staged = store.stage_modules(1, {sinfer::ops::LoraStore::StagedModule{0, &name, &a, &b, 2, 6, 6, 1.0F}});
    store.commit_staged(*staged);
    expect_small(1, "mixed: an in-place staged swap writes the same bits as the upload");

    sinfer::DeviceBuffer da(a.size() * 2), db(b.size() * 2);
    da.copy_from_host(a.data(), da.bytes);
    db.copy_from_host(b.data(), db.bytes);
    store.clear_slot(0);
    store.set_device_module(0, sinfer::DeviceAdapterModule{0, name, da.p, db.p, 2, 6, 6, 1.0F});
    expect_small(0, "mixed: a device publication writes the same bits");

    std::vector<std::uint16_t> a3(18, 0x3F80), b3(18, 0x3F80), bbig(15, 0x3F80);
    bool refused = false;
    try {
        store.set_module_slot(0, name, 0, a3, b3, 3, 6, 6, 1.0F);
    } catch (const std::invalid_argument&) { refused = true; }
    check(refused, "mixed: rank 3 is refused for a rank-2 bank");
    expect_small(0, "mixed: a refused load leaves the slot as it was");
    bool loaded = true;
    try {
        store.set_module_slot(0, "attn.big", 0, a3, bbig, 3, 6, 5, 1.0F);
    } catch (const std::exception&) { loaded = false; }
    check(loaded, "mixed: rank 3 loads into the default bank");

    // Two aliases of one bank must agree on its rank.
    sinfer::ops::LoraStoreSet other;
    sinfer::ops::LoraStore& clash = other.for_device(device);
    clash.configure(1, 8, 4);
    Binding r2{key, 9, 4, 4};
    r2.rank = 2;
    clash.register_module(0, "x", r2);
    clash.register_module(0, "y", Binding{key, 9, 4, 4});
    bool clashed = false;
    try { clash.ensure_banks(); } catch (const std::logic_error&) { clashed = true; }
    check(clashed, "mixed: aliases that disagree on the rank are refused");
}

} // namespace

int main() {
    const int count = device_count();
    if (count < 1) {
        std::cerr << "no CUDA device; skipping\n";
        return 77;
    }
    one_device();
    kv_generations();
    speculative_columns();
    mixed_ranks();
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
