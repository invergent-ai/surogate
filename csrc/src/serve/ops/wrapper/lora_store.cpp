// The resident adapter banks (api/ops/lora_store.h).

#include "api/ops/lora_store.h"
#include "api/ops/scale.h"

#include "core/device.h"
#include "core/engine_context.h"
#include "ops/kernel/lora_fused_limits.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <stdexcept>
#include <vector>

namespace sinfer::ops {
namespace {

float bf16_to_float(std::uint16_t bits) {
    const std::uint32_t word = static_cast<std::uint32_t>(bits) << 16U;
    float value              = 0.0F;
    std::memcpy(&value, &word, sizeof(value));
    return value;
}

std::uint16_t float_to_bf16(float value) {
    std::uint32_t word = 0;
    std::memcpy(&word, &value, sizeof(word));
    word += 0x7FFFU + ((word >> 16U) & 1U);
    return static_cast<std::uint16_t>(word >> 16U);
}

void* upload_zeroed(std::size_t elements) {
    void* device = nullptr;
    const std::size_t bytes = elements * sizeof(std::uint16_t);
    CUDA_CHECK(cudaMalloc(&device, bytes));
    CUDA_CHECK(cudaMemset(device, 0, bytes));
    return device;
}

} // namespace

int LoraStore::current_device() {
    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    return device;
}

LoraStore::~LoraStore() {
    // A destructor cannot throw, so the device is bound bare here rather than
    // through ScopedDevice. Freeing an allocation while another device is current
    // is what would leak it.
    int previous             = 0;
    const bool have_previous = cudaGetDevice(&previous) == cudaSuccess;
    if (device_ >= 0) { (void)cudaSetDevice(device_); }
    for (auto& [key, bank] : banks_) {
        if (!bank.raw) { continue; }
        cudaFree(bank.a);
        cudaFree(bank.b);
    }
    if (raw_round_state_) {
        cudaFree(scratch_);
        cudaFree(uniform_cell_);
    }
    if (have_previous) { (void)cudaSetDevice(previous); }
}

void LoraStore::configure(std::int32_t slots, std::int32_t max_rank, std::int32_t max_tokens) {
    if (slots <= 0 || max_rank <= 0 || max_tokens <= 0) {
        throw std::invalid_argument("lora_store: slots, max_rank and max_tokens must be positive");
    }
    if (!banks_.empty()) {
        throw std::logic_error("lora_store: configure must precede any slot");
    }
    slots_          = slots;
    max_rank_       = max_rank;
    scratch_tokens_ = max_tokens;
    // The device memory -- scratch, slot cell, banks -- is allocated by
    // ensure_banks, in one arena, so that under sleep mode the whole adapter
    // estate is a single Offload region of the engine. Direct users that never
    // register a directory (tools, tests) get raw allocations lazily below.
}

void LoraStore::ensure_raw_round_state() {
    if (scratch_ != nullptr) { return; }
    if (device_ < 0) { device_ = current_device(); }
    // Wide enough for a split site's low vectors: up to three pairs share one
    // launch, each padded to max_rank.
    scratch_ = upload_zeroed(static_cast<std::size_t>(kLoraFusedPairLimit) * max_rank_ *
                             scratch_tokens_);
    CUDA_CHECK(cudaMalloc(&uniform_cell_, sizeof(std::int32_t)));
    raw_round_state_        = true;
    const std::int32_t none = -1;
    CUDA_CHECK(cudaMemcpy(uniform_cell_, &none, sizeof(none), cudaMemcpyHostToDevice));
}

Tensor LoraStore::scratch(std::int32_t tokens) const {
    if (scratch_ == nullptr || tokens <= 0 || tokens > scratch_tokens_) { return Tensor{}; }
    // The view is the widest allocation, not the round's width: a captured graph
    // must see one shape, and the kernels bound their work by the ids they read.
    return Tensor(scratch_, DType::BF16, {kLoraFusedPairLimit * max_rank_ * scratch_tokens_});
}

void LoraStore::set_slot(const void* base_key, std::int32_t port, std::int32_t slot,
                         const std::vector<std::uint16_t>& a, const std::vector<std::uint16_t>& b,
                         std::int32_t rank, std::int32_t in_dim, std::int32_t out_dim,
                         float scale) {
    if (slots_ == 0) { throw std::logic_error("lora_store: configure() first"); }
    if (base_key == nullptr || slot < 0 || slot >= slots_) {
        throw std::invalid_argument("lora_store: slot is outside the configured range");
    }
    if (rank <= 0 || rank > max_rank_ || in_dim <= 0 || out_dim <= 0) {
        throw std::invalid_argument("lora_store: adapter geometry is outside the bank");
    }
    if (a.size() != static_cast<std::size_t>(rank) * in_dim ||
        b.size() != static_cast<std::size_t>(out_dim) * rank) {
        throw std::invalid_argument("lora_store: A must be [rank,in] and B [out,rank]");
    }

    const Key key{base_key, port};
    auto found = banks_.find(key);
    if (found == banks_.end()) {
        if (banks_built_) {
            throw std::logic_error(
                "lora_store: this projection has no bank and the banks are built -- banks are "
                "created by ensure_banks before serving so captured graphs and the lock-free "
                "find() stay valid; register the module in the target's directory");
        }
        ensure_raw_round_state();
        Bank bank;
        bank.raw = true;
        const std::size_t a_elements =
            static_cast<std::size_t>(slots_) * max_rank_ * in_dim;
        const std::size_t b_elements =
            static_cast<std::size_t>(slots_) * out_dim * max_rank_;
        bank.a             = upload_zeroed(a_elements);
        bank.b             = upload_zeroed(b_elements);
        bank.view.a        = bank.a;
        bank.view.b        = bank.b;
        bank.view.a_stride = static_cast<std::int64_t>(max_rank_) * in_dim;
        bank.view.b_stride = static_cast<std::int64_t>(out_dim) * max_rank_;
        bank.view.rank     = max_rank_;
        bank.view.n        = out_dim;
        bank.view.k        = in_dim;
        found              = banks_.emplace(key, bank).first;
    }
    Bank& bank = found->second;
    if (bank.view.k != in_dim || bank.view.n != out_dim) {
        throw std::invalid_argument("lora_store: two adapters disagree on this projection's shape");
    }

    // A padded to max_rank rows, with alpha/r folded in; rows past `rank` stay
    // zero so a short adapter contributes nothing through them.
    std::vector<std::uint16_t> a_padded(static_cast<std::size_t>(max_rank_) * in_dim, 0);
    for (std::int32_t r = 0; r < rank; ++r) {
        for (std::int32_t i = 0; i < in_dim; ++i) {
            const float value = bf16_to_float(a[static_cast<std::size_t>(r) * in_dim + i]) * scale;
            a_padded[static_cast<std::size_t>(r) * in_dim + i] = float_to_bf16(value);
        }
    }
    // B padded to max_rank columns, likewise.
    std::vector<std::uint16_t> b_padded(static_cast<std::size_t>(out_dim) * max_rank_, 0);
    for (std::int32_t row = 0; row < out_dim; ++row) {
        for (std::int32_t r = 0; r < rank; ++r) {
            b_padded[static_cast<std::size_t>(row) * max_rank_ + r] =
                b[static_cast<std::size_t>(row) * rank + r];
        }
    }

    // The caller may be on another device -- an adapter upload runs on an HTTP
    // handler thread, and under a pipeline this store is one stage's of several.
    // A copy resolves its destination against the current device, not against the
    // pointer, so bind this store's device for the duration.
    const ScopedDevice on_store(device_);
    CUDA_CHECK(cudaMemcpy(static_cast<std::uint16_t*>(bank.a) + static_cast<std::size_t>(slot) *
                                                                    bank.view.a_stride,
                          a_padded.data(), a_padded.size() * sizeof(std::uint16_t),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(static_cast<std::uint16_t*>(bank.b) + static_cast<std::size_t>(slot) *
                                                                    bank.view.b_stride,
                          b_padded.data(), b_padded.size() * sizeof(std::uint16_t),
                          cudaMemcpyHostToDevice));
    if (std::getenv("SUROGATE_SERVE_LORA_DEBUG") != nullptr) {
        double a_abs = 0.0, b_abs = 0.0;
        for (std::size_t i = 0; i < a_padded.size(); ++i) {
            a_abs = std::max(a_abs, std::abs(static_cast<double>(bf16_to_float(a_padded[i]))));
        }
        for (std::size_t i = 0; i < b_padded.size(); ++i) {
            b_abs = std::max(b_abs, std::abs(static_cast<double>(bf16_to_float(b_padded[i]))));
        }
        std::fprintf(stderr, "lora-slot: slot=%d rank=%d in=%d out=%d scale=%.3f |A|max=%.5f |B|max=%.5f\n",
                     slot, rank, in_dim, out_dim, static_cast<double>(scale), a_abs, b_abs);
    }
}

// The directory stays open across ensure_banks calls: a pipeline registers one
// stage's layers, builds their banks on that stage's device, and then registers
// the next stage's. What must not happen is a bank appearing once the engine
// serves -- `set_slot` refuses that, and every registration here runs inside the
// engine's construction window.
void LoraStore::register_module(std::int32_t layer, std::string module, ModuleBinding binding) {
    if (binding.key == nullptr || binding.in <= 0 || binding.out <= 0) {
        throw std::invalid_argument("lora_store: a module binding needs a key and positive shapes");
    }
    directory_.emplace(std::make_pair(layer, std::move(module)), binding);
}

bool LoraStore::covers_layer(std::int32_t layer) const {
    const auto bound = directory_.lower_bound(std::make_pair(layer, std::string{}));
    if (bound != directory_.end() && bound->first.first == layer) { return true; }
    // A layer this store holds but adapts nowhere -- a linear-attention layer
    // whose attention names are refused -- still belongs to it. Saying so is what
    // lets the refusal explain itself instead of the layer looking absent.
    const auto refused = layer_refusals_.lower_bound(std::make_pair(layer, std::string{}));
    return refused != layer_refusals_.end() && refused->first.first == layer;
}

void LoraStore::register_refusal(std::string module, std::string reason) {
    refusals_.emplace(std::move(module), std::move(reason));
}

void LoraStore::register_layer_refusal(std::int32_t layer, std::string module, std::string reason) {
    layer_refusals_.emplace(std::make_pair(layer, std::move(module)), std::move(reason));
}

void LoraStore::ensure_banks() {
    if (slots_ == 0) { throw std::logic_error("lora_store: configure() first"); }
    // Everything this call creates goes into one owning arena: the banks of the
    // modules registered since the last call, and this device's round state if it
    // has none yet. Allocated here -- inside the engine's construction window --
    // so under sleep mode it registers as one sleepable Offload region of that
    // engine, and a slept model's adapters leave VRAM with it.
    //
    // A pipeline calls this once per stage, with that stage's device current and
    // that stage's layers just registered, so a layer's banks land on the device
    // that holds the layer. Earlier arenas are kept rather than replaced: their
    // banks are already pointed at by the directory and by captured graphs.
    if (device_ < 0) { device_ = current_device(); }
    const bool need_round_state = scratch_ == nullptr;

    std::size_t total = 0;
    if (need_round_state) {
        total += static_cast<std::size_t>(kLoraFusedPairLimit) * max_rank_ * scratch_tokens_ *
                     sizeof(std::uint16_t) +
                 512; // the cell, and the alignment each of the two allocations may waste
    }
    struct Planned {
        Key key;
        std::size_t a_bytes;
        std::size_t b_bytes;
        const ModuleBinding* binding;
    };
    std::vector<Planned> planned;
    for (const auto& [where, binding] : directory_) {
        const Key key{binding.key, binding.port};
        if (banks_.find(key) != banks_.end()) { continue; }
        bool queued = false;
        for (const auto& item : planned) {
            if (item.key == key) { queued = true; break; }
        }
        if (queued) { continue; }
        const std::size_t a_bytes =
            static_cast<std::size_t>(slots_) * max_rank_ * binding.in * sizeof(std::uint16_t);
        const std::size_t b_bytes =
            static_cast<std::size_t>(slots_) * binding.out * max_rank_ * sizeof(std::uint16_t);
        planned.push_back({key, a_bytes, b_bytes, &binding});
        total += a_bytes + b_bytes + 512;
    }
    if (planned.empty() && !need_round_state) { return; }

    auto arena = std::make_unique<DeviceArena>(total);
    CUDA_CHECK(cudaMemset(arena->base(), 0, arena->capacity()));
    if (need_round_state) {
        scratch_      = arena->alloc_bytes(static_cast<std::size_t>(kLoraFusedPairLimit) *
                                     max_rank_ * scratch_tokens_ * sizeof(std::uint16_t))
                       .data;
        uniform_cell_ = static_cast<std::int32_t*>(arena->alloc_bytes(sizeof(std::int32_t)).data);
        const std::int32_t none = -1;
        CUDA_CHECK(cudaMemcpy(uniform_cell_, &none, sizeof(none), cudaMemcpyHostToDevice));
    }
    for (const Planned& item : planned) {
        const ModuleBinding& binding = *item.binding;
        Bank bank;
        bank.a             = arena->alloc_bytes(item.a_bytes).data;
        bank.b             = arena->alloc_bytes(item.b_bytes).data;
        bank.view.a        = bank.a;
        bank.view.b        = bank.b;
        bank.view.a_stride = static_cast<std::int64_t>(max_rank_) * binding.in;
        bank.view.b_stride = static_cast<std::int64_t>(binding.out) * max_rank_;
        bank.view.rank     = max_rank_;
        bank.view.n        = binding.out;
        bank.view.k        = binding.in;
        banks_.emplace(item.key, bank);
    }
    storage_.push_back(std::move(arena));
    banks_built_ = true;
}

void LoraStore::validate_module(std::int32_t layer, const std::string& module,
                                const std::vector<std::uint16_t>& a,
                                const std::vector<std::uint16_t>& b, std::int32_t rank,
                                std::int32_t in_dim, std::int32_t out_dim, float scale) const {
    if (rank <= 0 || rank > max_rank_ || in_dim <= 0 || out_dim <= 0 || !std::isfinite(scale) ||
        a.size() != static_cast<std::size_t>(rank) * in_dim ||
        b.size() != static_cast<std::size_t>(out_dim) * rank) {
        throw std::invalid_argument("invalid adapter tensor geometry or scale for '" + module + "'");
    }
    const auto refused = refusals_.find(module);
    if (refused != refusals_.end()) {
        throw std::invalid_argument("adapter module '" + module + "': " + refused->second);
    }
    const auto layer_refused = layer_refusals_.find({layer, module});
    if (layer_refused != layer_refusals_.end()) {
        throw std::invalid_argument("adapter module '" + module + "' on layer " +
                                    std::to_string(layer) + ": " + layer_refused->second);
    }
    const auto found = directory_.find({layer, module});
    if (found == directory_.end()) {
        throw std::invalid_argument(
            "adapter module '" + module + "' on layer " + std::to_string(layer) +
            " is not applied by this target; an adapter only partly applied is neither the base "
            "model nor the fine-tune");
    }
    const ModuleBinding& binding = found->second;
    if (in_dim != binding.in || out_dim != binding.out) {
        throw std::invalid_argument(
            "adapter module '" + module + "' on layer " + std::to_string(layer) + " is [" +
            std::to_string(out_dim) + "," + std::to_string(in_dim) + "] but this model's is [" +
            std::to_string(binding.out) + "," + std::to_string(binding.in) +
            "] -- the adapter was trained against a different model");
    }
}

void LoraStore::set_module_slot(std::int32_t layer, const std::string& module, std::int32_t slot,
                                const std::vector<std::uint16_t>& a,
                                const std::vector<std::uint16_t>& b, std::int32_t rank,
                                std::int32_t in_dim, std::int32_t out_dim, float scale) {
    validate_module(layer, module, a, b, rank, in_dim, out_dim, scale);
    const auto& binding = directory_.at({layer, module});
    set_slot(binding.key, binding.port, slot, a, b, rank, in_dim, out_dim, scale);
}

void LoraStore::validate_device_module(const DeviceAdapterModule& m) const {
    const auto found = directory_.find({m.layer, m.module});
    if (found == directory_.end() || m.in != found->second.in || m.out != found->second.out ||
        m.rank <= 0 || m.rank > max_rank_ || !std::isfinite(m.scale) || !m.a || !m.b) {
        throw std::invalid_argument("invalid device adapter module: " + m.module);
    }
    if (!banks_.contains(Key{found->second.key, found->second.port})) {
        throw std::logic_error("device adapter bank was not created at startup");
    }
    for (const void* pointer : {m.a, m.b}) {
        cudaPointerAttributes attrs{};
        CUDA_CHECK(cudaPointerGetAttributes(&attrs, pointer));
        if (attrs.type != cudaMemoryTypeDevice || attrs.device != device_) {
            throw std::invalid_argument("device adapter must be on the serving device");
        }
    }
}

void LoraStore::set_device_module(std::int32_t slot, const DeviceAdapterModule& m) {
    validate_device_module(m);
    if (slot < 0 || slot >= slots_) { throw std::invalid_argument("invalid adapter slot"); }
    const ScopedDevice on_store(device_);
    const auto& binding = directory_.at({m.layer, m.module});
    auto& bank = banks_.at(Key{binding.key, binding.port});
    auto* a = static_cast<std::uint16_t*>(bank.a) + slot * bank.view.a_stride;
    auto* b = static_cast<std::uint16_t*>(bank.b) + slot * bank.view.b_stride;
    CUDA_CHECK(cudaMemset(a, 0, bank.view.a_stride * 2));
    CUDA_CHECK(cudaMemset(b, 0, bank.view.b_stride * 2));
    CUDA_CHECK(cudaMemcpy(a, m.a, static_cast<std::size_t>(m.rank) * m.in * 2, cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy2D(b, max_rank_ * 2, m.b, m.rank * 2, m.rank * 2, m.out, cudaMemcpyDeviceToDevice));
    Tensor scaled(a, DType::BF16, {m.rank, m.in});
    ops::scale(scaled, m.scale, nullptr);
    CUDA_CHECK(cudaStreamSynchronize(nullptr));
}

void LoraStore::clear_slot(std::int32_t slot) {
    if (slot < 0 || slot >= slots_) { return; }
    const ScopedDevice on_store(device_);
    for (auto& [key, bank] : banks_) {
        CUDA_CHECK(cudaMemset(static_cast<std::uint16_t*>(bank.a) +
                                  static_cast<std::size_t>(slot) * bank.view.a_stride,
                              0, static_cast<std::size_t>(bank.view.a_stride) * sizeof(std::uint16_t)));
        CUDA_CHECK(cudaMemset(static_cast<std::uint16_t*>(bank.b) +
                                  static_cast<std::size_t>(slot) * bank.view.b_stride,
                              0, static_cast<std::size_t>(bank.view.b_stride) * sizeof(std::uint16_t)));
    }
}

void LoraStore::write_uniform_slot(std::int32_t slot, cudaStream_t stream) const {
    std::int32_t* cell = uniform_cell_;
    if (cell == nullptr) { return; }
    // Stream-ordered so it lands before the round that reads it, and captured as a
    // node when the round is being recorded.
    CUDA_CHECK(cudaMemcpyAsync(cell, &slot, sizeof(slot), cudaMemcpyHostToDevice, stream));
}

LoraStore& LoraStoreSet::for_device(int device) {
    if (device < 0 || device >= kMaxDevices) {
        throw std::invalid_argument("lora_store: device ordinal " + std::to_string(device) +
                                    " is outside the " + std::to_string(kMaxDevices) +
                                    " this engine indexes");
    }
    // The common case by far: the store exists and this is one acquire load.
    LoraStore* existing = stores_[static_cast<std::size_t>(device)].load(std::memory_order_acquire);
    if (existing != nullptr) { return *existing; }
    const std::lock_guard<std::mutex> lock(mutex_);
    existing = stores_[static_cast<std::size_t>(device)].load(std::memory_order_relaxed);
    if (existing != nullptr) { return *existing; }
    owned_.push_back(std::make_unique<LoraStore>());
    LoraStore* created = owned_.back().get();
    stores_[static_cast<std::size_t>(device)].store(created, std::memory_order_release);
    return *created;
}

LoraStore* LoraStoreSet::peek(int device) const noexcept {
    if (device < 0 || device >= kMaxDevices) { return nullptr; }
    return stores_[static_cast<std::size_t>(device)].load(std::memory_order_acquire);
}

std::vector<int> LoraStoreSet::devices() const {
    std::vector<int> out;
    for (int device = 0; device < kMaxDevices; ++device) {
        if (stores_[static_cast<std::size_t>(device)].load(std::memory_order_acquire) != nullptr) {
            out.push_back(device);
        }
    }
    return out;
}

LoraStore& lora_store_for_current_device() {
    return engine_slot<LoraStoreSet>().for_device(LoraStore::current_device());
}

namespace {
thread_local LoraRound t_round;
} // namespace

bool lora_active() {
    const LoraStore* store = engine_slot<LoraStoreSet>().peek(LoraStore::current_device());
    return store != nullptr && store->active();
}
void lora_set_active(bool active) { lora_store_for_current_device().set_active(active); }

void lora_set_round(const LoraRound& round) { t_round = round; }
void lora_clear_round() { t_round = LoraRound{}; }
const LoraRound& lora_current_round() { return t_round; }

} // namespace sinfer::ops
