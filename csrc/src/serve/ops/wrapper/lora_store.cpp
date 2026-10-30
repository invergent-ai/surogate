#include "api/ops/lora_replacement.h"
// The resident adapter banks (api/ops/lora_store.h).

#include "api/ops/lora_store.h"
#include "api/ops/scale.h"

#include "core/device.h"
#include "core/engine_context.h"
#include "ops/kernel/lora_fused_limits.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <set>
#include <thread>
#include <tuple>
#include <stdexcept>
#include <unordered_set>
#include <vector>

namespace sinfer::ops {
namespace {

struct StoreContextScope {
    const void* previous = current_ops_owner();
    explicit StoreContextScope(const void* owner) { bind_ops_context(static_cast<EngineOpsContext*>(const_cast<void*>(owner))); }
    ~StoreContextScope() { bind_ops_context(static_cast<EngineOpsContext*>(const_cast<void*>(previous))); }
};

// SUROGATE_SERVE_LORA_MODULES="fragment,...": the layer modules whose name contains none of the fragments get no
// bank and are refused on their layer (globals and the sparse-MoE bank tables keep theirs). A projection without a
// bank keeps its fused base route and launches no delta, so adapter rounds stop paying for projections no served
// adapter covers -- e.g. "experts.,shared_expert" for adapters on the MoE experts only. Unset: every module.
const std::vector<std::string>& served_module_fragments() {
    static const std::vector<std::string> fragments = [] {
        std::vector<std::string> out;
        const char* raw = std::getenv("SUROGATE_SERVE_LORA_MODULES");
        for (std::string rest = raw != nullptr ? raw : ""; !rest.empty();) {
            const std::size_t comma = rest.find(',');
            if (comma != 0) { out.push_back(rest.substr(0, comma)); }
            rest = comma == std::string::npos ? std::string{} : rest.substr(comma + 1);
        }
        return out;
    }();
    return fragments;
}

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

std::size_t LoraStore::device_bytes() const noexcept {
    std::size_t bytes = 0;
    for (const auto& arena : storage_) { bytes += arena->capacity(); }
    return bytes;
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
        cudaFree(bank.affine);
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
    owner_ = current_ops_owner();
    slots_          = slots;
    max_rank_       = max_rank;
    scratch_tokens_ = max_tokens;
    versions_       = std::make_unique<std::atomic<std::uint32_t>[]>(static_cast<std::size_t>(slots));
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

    if (!bank.host_a.empty()) {
        std::copy(a_padded.begin(), a_padded.end(), bank.host_a.begin() + slot * bank.view.a_stride);
        std::copy(b_padded.begin(), b_padded.end(), bank.host_b.begin() + slot * bank.view.b_stride);
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
void LoraStore::register_base_weight(const Weight& weight) {
    if (weight.qdata && weight.n > 0 && weight.k > 0) { base_weights_.emplace(weight.qdata, weight); }
}
void LoraStore::register_base(const void* key, std::int32_t port, std::vector<LoraBaseView> base,
                               Tensor bias, bool bias_in_output) {
    bases_[Key{key, port}] = Base{std::move(base), bias, bias_in_output};
}
LoraStore::Base LoraStore::base_for(const ModuleBinding& binding) const {
    if (const auto it = bases_.find(Key{binding.key, binding.port}); it != bases_.end()) { return it->second; }
    const auto found = base_weights_.find(binding.key);
    if (found == base_weights_.end()) { throw std::invalid_argument("adapter base projection was not registered"); }
    const auto& weight = found->second;
    if (weight.k != binding.in) { throw std::invalid_argument("adapter base projection input shape disagrees"); }
    if (weight.n == binding.out) { return Base{{LoraBaseView{weight, 0, 0, binding.out}}}; }
    std::map<int, int> ports;
    for (const auto& [where, parts] : directory_) {
        for (const auto& part : parts) {
            if (part.key == binding.key && part.in == weight.k) { ports.emplace(part.port, part.out); }
        }
    }
    // Fused attention is Q,K,gate,V; dense MLPs are gate,up; GDN controls are a,b.
    constexpr int order[] = {0, 1, 7, 2, 5, 6, 8, 9, 10, 11};
    int rows = 0, first = -1;
    for (int port : order) {
        if (!ports.contains(port)) { continue; }
        if (port == binding.port) { first = rows; }
        rows += ports.at(port);
    }
    if (rows != weight.n || first < 0) { throw std::invalid_argument("adapter base projection needs an explicit row mapping"); }
    return Base{{LoraBaseView{weight, first, 0, binding.out}}};
}

std::shared_ptr<DeviceArena> LoraStore::stage_replacement(std::int32_t layer, const std::string& module,
    const std::vector<std::uint16_t>& weight) const {
    if (weight.empty()) { return {}; }
    const auto& parts = module_parts(layer, module);
    if (parts.size() != 1 || !replacements_.contains(Key{parts.front().key, parts.front().port}) ||
        weight.size() != static_cast<std::size_t>(parts.front().in) * parts.front().out) {
        throw std::invalid_argument("saved embedding/head shape disagrees with the served vocabulary");
    }
    const ScopedDevice device(device_ < 0 ? current_device() : device_);
    const StoreContextScope owner(owner_);
    auto staged = std::make_shared<DeviceArena>(weight.size() * 2);
    CUDA_CHECK(cudaMemcpy(staged->base(), weight.data(), weight.size() * 2, cudaMemcpyHostToDevice));
    return staged;
}

std::vector<std::vector<float>> LoraStore::prepare_affine(std::int32_t layer,
    const std::string& module, const std::vector<std::uint16_t>& a,
    const std::vector<std::uint16_t>& b, std::int32_t rank, float scale,
    const std::vector<float>& magnitude, const std::vector<float>& bias,
    const std::vector<float>& lora_bias, const std::vector<std::uint16_t>& base_weight, std::shared_ptr<DeviceArena> prepared) const {
    const auto& parts = module_parts(layer, module);
    const int out = parts.front().checkpoint_out();
    if (!base_weight.empty() && (parts.size() != 1 ||
        !replacements_.contains(Key{parts.front().key, parts.front().port}) ||
        base_weight.size() != static_cast<std::size_t>(parts.front().in) * out)) {
        throw std::invalid_argument("saved embedding/head shape disagrees with the served vocabulary");
    }
    for (const auto* values : {&magnitude, &bias, &lora_bias}) {
        if (!values->empty() && values->size() != static_cast<std::size_t>(out)) {
            throw std::invalid_argument("adapter magnitude or bias shape disagrees with '" + module + "'");
        }
        for (float value : *values) {
            if (!std::isfinite(value)) { throw std::invalid_argument("adapter magnitude or bias is nonfinite"); }
        }
    }
    std::vector<std::vector<float>> result;
    if (magnitude.empty() && bias.empty() && lora_bias.empty()) { return result; }
    const ScopedDevice device(device_ < 0 ? current_device() : device_);
    const StoreContextScope owner(owner_);
    for (const auto& part : parts) {
        std::vector<float> affine(part.out * 2, 0.0F), norms, original(part.out, 0.0F);
        auto base = base_for(part);
        if (!magnitude.empty()) {
            std::vector<std::uint16_t> selected(static_cast<std::size_t>(part.out) * rank);
            for (int row = 0; row < part.out; ++row) {
                std::copy_n(b.data() + static_cast<std::size_t>(part.source_row(row)) * rank,
                            rank, selected.data() + static_cast<std::size_t>(row) * rank);
            }
            auto replacement = prepared;
            if (!base_weight.empty()) {
                if (!replacement) { replacement = stage_replacement(layer, module, base_weight); }
                const bool transpose = part.port == kLoraEmbeddingPort;
                Weight weight;
                weight.payload = weight.qdata = replacement->base();
                weight.payload_bytes = base_weight.size() * 2;
                weight.qtype = QType::BF16_CTRL; weight.layout = QuantLayout::Contiguous; weight.ndim = 2;
                weight.n = weight.shape[0] = weight.padded_shape[0] = transpose ? part.in : part.out;
                weight.k = weight.shape[1] = weight.padded_shape[1] = transpose ? part.out : part.in;
                base.parts = {{weight, 0, 0, part.out, transpose}};
            }
            norms = lora_weight_norms(base.parts, a, selected, rank, part.in, part.out, scale);
        }
        if (base.bias.data && (!bias.empty() || (!magnitude.empty() && base.bias_in_output))) {
            if (base.bias.dtype == DType::FP32) {
                CUDA_CHECK(cudaMemcpy(original.data(), base.bias.data, original.size() * 4, cudaMemcpyDefault));
            } else {
                std::vector<std::uint16_t> raw(part.out);
                CUDA_CHECK(cudaMemcpy(raw.data(), base.bias.data, raw.size() * 2, cudaMemcpyDefault));
                for (int row = 0; row < part.out; ++row) { original[row] = bf16_to_float(raw[row]); }
            }
        }
        for (int row = 0; row < part.out; ++row) {
            const int src = part.source_row(row);
            const float factor = magnitude.empty() ? 1.0F : magnitude[src] / norms[row];
            if (!std::isfinite(factor)) { throw std::invalid_argument("DoRA magnitude ratio is nonfinite"); }
            affine[row] = factor - 1.0F;
            affine[part.out + row] = (lora_bias.empty() ? 0.0F : scale * factor * lora_bias[src]) +
                (bias.empty() ? 0.0F : bias[src] - original[row]) -
                (base.bias_in_output ? (factor - 1.0F) * original[row] : 0.0F);
        }
        result.push_back(std::move(affine));
    }
    return result;
}

void LoraStore::register_module(std::int32_t layer, std::string module, ModuleBinding binding) {
    if (binding.key == nullptr || binding.in <= 0 || binding.out <= 0 ||
        binding.row_offset < 0 || binding.row_group < 0 || binding.row_stride < binding.row_group ||
        binding.source_row(binding.out - 1) >= binding.checkpoint_out()) {
        throw std::invalid_argument("lora_store: a module binding needs a key and positive shapes");
    }
    auto& parts = directory_[{layer, std::move(module)}];
    for (const auto& part : parts) {
        if (part.in != binding.in || part.checkpoint_out() != binding.checkpoint_out() ||
            (part.key == binding.key && part.port == binding.port)) {
            throw std::invalid_argument("lora_store: inconsistent or duplicate module parts");
        }
    }
    parts.push_back(binding);
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
    if (const auto& fragments = served_module_fragments(); !fragments.empty()) {
        std::unordered_set<Key, KeyHash> tabled;
        for (const auto& [key, bindings] : table_bindings_) {
            for (const auto& binding : bindings) {
                if (binding.key) { tabled.insert(Key{binding.key, binding.port}); }
            }
        }
        for (auto it = directory_.begin(); it != directory_.end();) {
            const auto& [where, parts] = *it;
            const bool served = where.first < 0 ||
                std::any_of(fragments.begin(), fragments.end(),
                            [&](const std::string& f) { return where.second.find(f) != std::string::npos; }) ||
                std::any_of(parts.begin(), parts.end(), [&](const ModuleBinding& b) {
                    return tabled.contains(Key{b.key, b.port}) || banks_.contains(Key{b.key, b.port});
                });
            if (served) { ++it; continue; }
            layer_refusals_.emplace(where, "not served by this engine (SUROGATE_SERVE_LORA_MODULES)");
            it = directory_.erase(it);
        }
    }

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
    std::unordered_map<Key, const ModuleBinding*, KeyHash> queued;
    for (const auto& [where, parts] : directory_) {
      for (const auto& binding : parts) {
        const Key key{binding.key, binding.port};
        if (banks_.find(key) != banks_.end()) { continue; }
        const auto [existing, inserted] = queued.emplace(key, &binding);
        if (!inserted) {
            if (existing->second->in != binding.in || existing->second->out != binding.out) {
                throw std::logic_error("LoRA aliases disagree on the projection shape");
            }
            continue;
        }
        const std::size_t a_bytes =
            static_cast<std::size_t>(slots_) * max_rank_ * binding.in * sizeof(std::uint16_t);
        const std::size_t b_bytes =
            static_cast<std::size_t>(slots_) * binding.out * max_rank_ * sizeof(std::uint16_t);
        planned.push_back({key, a_bytes, b_bytes, &binding});
        total += a_bytes + b_bytes + static_cast<std::size_t>(slots_) * binding.out * 8 + 768;
        if (replacements_.contains(key)) { total += slots_ * sizeof(void*) + 256; }
      }
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
        bank.affine = arena->alloc_bytes(static_cast<std::size_t>(slots_) * binding.out * 8).data;
        bank.view.gain = static_cast<const float*>(bank.affine);
        bank.view.bias = bank.view.gain + static_cast<std::size_t>(slots_) * binding.out;
        if (auto found = replacements_.find(item.key); found != replacements_.end()) {
            found->second.table = static_cast<void**>(arena->alloc_bytes(slots_ * sizeof(void*)).data);
            found->second.values.resize(slots_);
        }
        banks_.emplace(item.key, bank);
    }
    storage_.push_back(std::move(arena));
    banks_built_ = true;
    for (const auto& [key, bindings] : table_bindings_) {
        if (tables_.contains(key)) { continue; }
        std::vector<LoraBank> views;
        auto& host = host_tables_[key];
        for (const auto& binding : bindings) {
            LoraBank cpu;
            if (binding.key) {
                auto& bank = banks_.at(Key{binding.key, binding.port});
                if (bank.host_a.empty()) {
                    bank.host_a.resize(slots_ * bank.view.a_stride);
                    bank.host_b.resize(slots_ * bank.view.b_stride);
                    bank.host_affine.resize(static_cast<std::size_t>(slots_) * bank.view.n * 2);
                }
                cpu = bank.view;
                cpu.a = bank.host_a.data();
                cpu.b = bank.host_b.data();
                cpu.gain = bank.host_affine.data();
                cpu.bias = cpu.gain + static_cast<std::size_t>(slots_) * bank.view.n;
            }
            host.push_back(cpu);
            views.push_back(binding.key ? banks_.at(Key{binding.key, binding.port}).view : LoraBank{});
        }
        auto table = std::make_unique<DeviceArena>(views.size() * sizeof(LoraBank));
        CUDA_CHECK(cudaMemcpy(table->base(), views.data(), views.size() * sizeof(LoraBank), cudaMemcpyHostToDevice));
        tables_.emplace(key, static_cast<const LoraBank*>(table->base()));
        storage_.push_back(std::move(table));
    }
}

void LoraStore::register_bank_table(const void* key, const std::vector<ModuleBinding>& bindings) {
    if (banks_built_) { throw std::logic_error("LoRA bank tables must be registered before capture"); }
    table_bindings_.emplace(key, bindings);
}

const LoraBank* LoraStore::bank_table(const void* key) const noexcept {
    const auto found = tables_.find(key);
    return found == tables_.end() ? nullptr : found->second;
}

const LoraBank* LoraStore::host_bank_table(const void* key) const noexcept {
    const auto found = host_tables_.find(key);
    return found == host_tables_.end() ? nullptr : found->second.data();
}

const LoraStore::ModuleParts& LoraStore::module_parts(std::int32_t layer, const std::string& module) const {
    std::string name = module;
    if (const auto found = directory_.find({layer, name}); found != directory_.end()) {
        return found->second;
    }
    // Keep nested names such as mlp.experts.7.down_proj distinct from the dense
    // down_proj. Only the conventional, single-component namespaces alias a
    // target's legacy directory entries.
    for (const std::string_view prefix : {"self_attn.", "mlp.", "attention.", "feed_forward."}) {
        if (name.starts_with(prefix) && name.find('.', prefix.size()) == std::string::npos) {
            name.erase(0, prefix.size());
            break;
        }
    }
    if (const auto found = directory_.find({layer, name}); found != directory_.end()) {
        return found->second;
    }
    const std::string merge_help =
        ". Check that the adapter matches this base model. For unsupported serving modules, "
        "merge the adapter into its base checkpoint with `surogate merge`, then convert and serve the merged checkpoint.";
    const auto refused = refusals_.find(name);
    if (refused != refusals_.end()) {
        throw std::invalid_argument("unsupported adapter module '" + module + "': " + refused->second + merge_help);
    }
    const auto layer_refused = layer_refusals_.find({layer, name});
    if (layer_refused != layer_refusals_.end()) {
        throw std::invalid_argument("unsupported adapter module '" + module + "' on layer " +
                                    std::to_string(layer) + ": " + layer_refused->second + merge_help);
    }
    throw std::invalid_argument(
            "adapter module '" + module + "' on layer " + std::to_string(layer) +
            " is unsupported by this serving model" + merge_help);
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
    const auto& binding = module_parts(layer, module).front();
    if (in_dim != binding.in || out_dim != binding.checkpoint_out()) {
        throw std::invalid_argument(
            "adapter module '" + module + "' on layer " + std::to_string(layer) + " is [" +
            std::to_string(out_dim) + "," + std::to_string(in_dim) + "] but this model's is [" +
            std::to_string(binding.checkpoint_out()) + "," + std::to_string(binding.in) +
            "] -- the adapter was trained against a different model");
    }
}

void LoraStore::set_module_slot(std::int32_t layer, const std::string& module, std::int32_t slot,
                                const std::vector<std::uint16_t>& a,
                                const std::vector<std::uint16_t>& b, std::int32_t rank,
                                std::int32_t in_dim, std::int32_t out_dim, float scale,
                                const std::vector<float>& magnitude, const std::vector<float>& bias,
                                const std::vector<float>& lora_bias, const std::vector<std::uint16_t>& base_weight, std::shared_ptr<DeviceArena> prepared_replacement) {
    validate_module(layer, module, a, b, rank, in_dim, out_dim, scale);
    if (!base_weight.empty() && !prepared_replacement) { prepared_replacement = stage_replacement(layer, module, base_weight); }
    const auto affine = prepare_affine(layer, module, a, b, rank, scale, magnitude, bias, lora_bias, base_weight, prepared_replacement);
    const ScopedDevice device(device_ < 0 ? current_device() : device_);
    const StoreContextScope owner(owner_);
    std::size_t index = 0;
    for (const auto& binding : module_parts(layer, module)) {
        if (binding.out == out_dim && binding.row_offset == 0 && binding.row_group == 0) {
            set_slot(binding.key, binding.port, slot, a, b, rank, in_dim, out_dim, scale);
        } else {
            std::vector<std::uint16_t> selected(static_cast<std::size_t>(binding.out) * rank);
            for (std::int32_t row = 0; row < binding.out; ++row) {
                std::copy_n(b.data() + static_cast<std::size_t>(binding.source_row(row)) * rank,
                            rank, selected.data() + static_cast<std::size_t>(row) * rank);
            }
            set_slot(binding.key, binding.port, slot, a, selected, rank, in_dim, binding.out, scale);
        }
        auto& bank = banks_.at(Key{binding.key, binding.port});
        if (bank.affine) {
            std::vector<float> neutral(binding.out * 2, 0.0F);
            const auto& values = affine.empty() ? neutral : affine[index];
            auto* dst = static_cast<float*>(bank.affine);
            const auto stride = static_cast<std::size_t>(slots_) * binding.out;
            for (int half = 0; half < 2; ++half) {
                CUDA_CHECK(cudaMemcpy(dst + half * stride + slot * binding.out,
                    values.data() + half * binding.out, binding.out * 4, cudaMemcpyHostToDevice));
                if (!bank.host_affine.empty()) {
                    std::copy_n(values.data() + half * binding.out, binding.out,
                        bank.host_affine.data() + half * stride + slot * binding.out);
                }
            }
        }
        if (auto found = replacements_.find(Key{binding.key, binding.port}); found != replacements_.end()) {
            auto& replacement = found->second;
            auto uploaded = prepared_replacement;
            void* pointer = uploaded ? uploaded->base() : nullptr;
            CUDA_CHECK(cudaMemcpy(replacement.table + slot, &pointer, sizeof(pointer), cudaMemcpyHostToDevice));
            replacement.values[slot] = std::move(uploaded);
        }
        ++index;
    }
}


void LoraStore::validate_device_module(const DeviceAdapterModule& m) const {
    const auto& parts = module_parts(m.layer, m.module);
    if (m.in != parts.front().in || m.out != parts.front().checkpoint_out() ||
        m.rank <= 0 || m.rank > max_rank_ || !std::isfinite(m.scale) || !m.a || !m.b) {
        throw std::invalid_argument("invalid device adapter module: " + m.module);
    }
    for (const auto& binding : parts) {
        if (!banks_.contains(Key{binding.key, binding.port})) {
            throw std::logic_error("device adapter bank was not created at startup");
        }
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
    for (const auto& binding : module_parts(m.layer, m.module)) {
        auto& bank = banks_.at(Key{binding.key, binding.port});
        auto* a = static_cast<std::uint16_t*>(bank.a) + slot * bank.view.a_stride;
        auto* b = static_cast<std::uint16_t*>(bank.b) + slot * bank.view.b_stride;
        CUDA_CHECK(cudaMemset(a, 0, bank.view.a_stride * 2));
        CUDA_CHECK(cudaMemset(b, 0, bank.view.b_stride * 2));
        CUDA_CHECK(cudaMemcpy(a, m.a, static_cast<std::size_t>(m.rank) * m.in * 2, cudaMemcpyDeviceToDevice));
        const auto group = binding.row_group ? binding.row_group : binding.out;
        for (std::int32_t row = 0; row < binding.out; row += group) {
            const auto* source = static_cast<const std::uint16_t*>(m.b) +
                                 static_cast<std::size_t>(binding.source_row(row)) * m.rank;
            CUDA_CHECK(cudaMemcpy2D(b + static_cast<std::size_t>(row) * max_rank_, max_rank_ * 2,
                                    source, m.rank * 2, m.rank * 2,
                                    std::min(group, binding.out - row), cudaMemcpyDeviceToDevice));
        }
        Tensor scaled(a, DType::BF16, {m.rank, m.in});
        ops::scale(scaled, m.scale, nullptr);
        if (bank.affine) {
            for (int half = 0; half < 2; ++half) {
                const auto offset = (static_cast<std::size_t>(half) * slots_ + slot) * bank.view.n;
                CUDA_CHECK(cudaMemset(static_cast<float*>(bank.affine) + offset, 0, bank.view.n * 4));
                if (!bank.host_affine.empty()) { std::fill_n(bank.host_affine.data() + offset, bank.view.n, 0.0F); }
            }
        }
        if (auto found = replacements_.find(Key{binding.key, binding.port}); found != replacements_.end()) {
            void* empty = nullptr;
            CUDA_CHECK(cudaMemcpy(found->second.table + slot, &empty, sizeof(empty), cudaMemcpyHostToDevice));
            found->second.values[slot].reset();
        }
        if (!bank.host_a.empty()) {
            CUDA_CHECK(cudaMemcpy(bank.host_a.data() + slot * bank.view.a_stride, a,
                                   bank.view.a_stride * 2, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(bank.host_b.data() + slot * bank.view.b_stride, b,
                                   bank.view.b_stride * 2, cudaMemcpyDeviceToHost));
        }
    }
    CUDA_CHECK(cudaStreamSynchronize(nullptr));
}

void LoraStore::set_slot_identity(std::int32_t slot, const std::array<std::uint8_t, 16>& digest) {
    std::lock_guard lock(identity_mutex_);
    identities_[slot] = digest;
}

std::array<std::uint8_t, 16> LoraStore::slot_identity(std::int32_t slot) const {
    std::lock_guard lock(identity_mutex_);
    const auto found = identities_.find(slot);
    return found == identities_.end() ? std::array<std::uint8_t, 16>{} : found->second;
}

void LoraStore::clear_slot(std::int32_t slot) {
    {
        std::lock_guard lock(identity_mutex_);
        identities_.erase(slot);
    }
    if (slot < 0 || slot >= slots_) { return; }
    const ScopedDevice on_store(device_);
    for (auto& [key, replacement] : replacements_) {
        void* empty = nullptr;
        CUDA_CHECK(cudaMemcpy(replacement.table + slot, &empty, sizeof(empty), cudaMemcpyHostToDevice));
        replacement.values[slot].reset();
    }
    for (auto& [key, bank] : banks_) {
        if (bank.affine) {
            for (int half = 0; half < 2; ++half) {
                const auto offset = (static_cast<std::size_t>(half) * slots_ + slot) * bank.view.n;
                CUDA_CHECK(cudaMemset(static_cast<float*>(bank.affine) + offset, 0, bank.view.n * 4));
                if (!bank.host_affine.empty()) { std::fill_n(bank.host_affine.data() + offset, bank.view.n, 0.0F); }
            }
        }
        if (!bank.host_a.empty()) {
            std::fill_n(bank.host_a.data() + slot * bank.view.a_stride, bank.view.a_stride, 0);
            std::fill_n(bank.host_b.data() + slot * bank.view.b_stride, bank.view.b_stride, 0);
        }
        CUDA_CHECK(cudaMemset(static_cast<std::uint16_t*>(bank.a) +
                                  static_cast<std::size_t>(slot) * bank.view.a_stride,
                              0, static_cast<std::size_t>(bank.view.a_stride) * sizeof(std::uint16_t)));
        CUDA_CHECK(cudaMemset(static_cast<std::uint16_t*>(bank.b) +
                                  static_cast<std::size_t>(slot) * bank.view.b_stride,
                              0, static_cast<std::size_t>(bank.view.b_stride) * sizeof(std::uint16_t)));
    }
}

LoraStore::StagedSlot::~StagedSlot() {
    if (host_ != nullptr) { (void)cudaFreeHost(host_); }
}

std::unique_ptr<LoraStore::StagedSlot>
LoraStore::stage_modules(std::int32_t slot, const std::vector<StagedModule>& modules) const {
    if (slots_ == 0 || slot < 0 || slot >= slots_) {
        throw std::invalid_argument("lora_store: slot is outside the configured range");
    }
    // Which bank each module part fills: the same resolution set_module_slot makes.
    struct Fill {
        const StagedModule* module   = nullptr;
        const ModuleBinding* binding = nullptr;
    };
    std::unordered_map<Key, Fill, KeyHash> fills;
    for (const StagedModule& m : modules) {
        validate_module(m.layer, *m.module, *m.a, *m.b, m.rank, m.in_dim, m.out_dim, m.scale);
        for (const auto& binding : module_parts(m.layer, *m.module)) {
            const Key key{binding.key, binding.port};
            if (!banks_.contains(key)) {
                throw std::logic_error("in-place adapter swap: module '" + *m.module + "' has no bank");
            }
            if (!fills.emplace(key, Fill{&m, &binding}).second) {
                throw std::invalid_argument("adapter modules overlap at '" + *m.module + "'");
            }
        }
    }

    // The slot's regions of every bank. A covered bank sends A, B and its (neutral, zero) affine
    // rows from the image, so with one slot a run of covered banks -- adjacent in the arena -- is
    // one copy; an uncovered bank's regions are cleared on the device.
    struct Planned {
        char* device      = nullptr;
        std::size_t bytes = 0;
        const Bank* bank  = nullptr;
        const Fill* fill  = nullptr;
        int part          = 0; // 0 A, 1 B, 2 affine
    };
    auto staged   = std::make_unique<StagedSlot>();
    staged->slot_ = slot;
    std::vector<Planned> planned;
    for (const auto& [key, bank] : banks_) {
        const std::size_t a_bytes = static_cast<std::size_t>(bank.view.a_stride) * sizeof(std::uint16_t);
        const std::size_t b_bytes = static_cast<std::size_t>(bank.view.b_stride) * sizeof(std::uint16_t);
        char* a = static_cast<char*>(bank.a) + static_cast<std::size_t>(slot) * a_bytes;
        char* b = static_cast<char*>(bank.b) + static_cast<std::size_t>(slot) * b_bytes;
        const auto found = fills.find(key);
        const Fill* fill = found == fills.end() ? nullptr : &found->second;
        const auto place = [&](char* device, std::size_t bytes, int part) {
            if (fill != nullptr) {
                planned.push_back({device, bytes, &bank, fill, part});
            } else {
                staged->zeros_.push_back({device, 0, bytes});
            }
        };
        place(a, a_bytes, 0);
        place(b, b_bytes, 1);
        if (bank.affine) {
            for (int half = 0; half < 2; ++half) {
                const std::size_t offset =
                    (static_cast<std::size_t>(half) * slots_ + slot) * static_cast<std::size_t>(bank.view.n);
                place(reinterpret_cast<char*>(static_cast<float*>(bank.affine) + offset),
                      static_cast<std::size_t>(bank.view.n) * sizeof(float), 2);
            }
        }
    }
    const auto by_device = [](const auto& x, const auto& y) { return x.device < y.device; };
    std::sort(planned.begin(), planned.end(), by_device);
    std::size_t total = 0;
    for (const Planned& item : planned) { total += item.bytes; }
    // Pinned, so the commit's copies run at the bus's speed and asynchronously.
    CUDA_CHECK(cudaHostAlloc(&staged->host_, std::max<std::size_t>(total, 1), cudaHostAllocPortable));
    staged->bytes_ = total;

    std::unordered_map<Key, std::pair<std::size_t, std::size_t>, KeyHash> placed; // A and B offsets
    std::size_t offset = 0;
    for (const Planned& item : planned) {
        char* out_bytes = static_cast<char*>(staged->host_) + offset;
        const StagedModule& m = *item.fill->module;
        const Bank& bank      = *item.bank;
        if (item.part == 0) {
            // A padded to max_rank rows with alpha/r folded in, as set_slot writes it.
            auto* out              = reinterpret_cast<std::uint16_t*>(out_bytes);
            const std::int32_t in  = bank.view.k;
            for (std::int32_t r = 0; r < max_rank_; ++r) {
                for (std::int32_t i = 0; i < in; ++i) {
                    out[static_cast<std::size_t>(r) * in + i] =
                        r < m.rank ? float_to_bf16(bf16_to_float((*m.a)[static_cast<std::size_t>(r) * in + i]) *
                                                   m.scale)
                                   : std::uint16_t{0};
                }
            }
        } else if (item.part == 1) {
            // B padded to max_rank columns, the binding's rows of the checkpoint's B.
            auto* out                      = reinterpret_cast<std::uint16_t*>(out_bytes);
            const ModuleBinding& binding   = *item.fill->binding;
            for (std::int32_t row = 0; row < bank.view.n; ++row) {
                const std::size_t source = static_cast<std::size_t>(binding.source_row(row)) * m.rank;
                for (std::int32_t r = 0; r < max_rank_; ++r) {
                    out[static_cast<std::size_t>(row) * max_rank_ + r] =
                        r < m.rank ? (*m.b)[source + r] : std::uint16_t{0};
                }
            }
        } else {
            std::memset(out_bytes, 0, item.bytes); // plain LoRA: neutral affine rows
        }
        if (item.part < 2) {
            const Key key{item.fill->binding->key, item.fill->binding->port};
            auto& where = placed[key];
            (item.part == 0 ? where.first : where.second) = offset;
        }
        if (!staged->copies_.empty() &&
            static_cast<char*>(staged->copies_.back().device) + staged->copies_.back().bytes == item.device) {
            staged->copies_.back().bytes += item.bytes;
        } else {
            staged->copies_.push_back({item.device, offset, item.bytes});
        }
        offset += item.bytes;
    }
    std::sort(staged->zeros_.begin(), staged->zeros_.end(), by_device);
    std::vector<StagedSlot::Region> zeros;
    for (const auto& region : staged->zeros_) {
        if (!zeros.empty() && static_cast<char*>(zeros.back().device) + zeros.back().bytes == region.device) {
            zeros.back().bytes += region.bytes;
        } else {
            zeros.push_back(region);
        }
    }
    staged->zeros_ = std::move(zeros);
    for (const auto& [key, bank] : banks_) {
        if (bank.host_a.empty() && bank.host_affine.empty()) { continue; }
        const auto found = placed.find(key);
        staged->mirrors_.push_back({key.weight, key.port,
                                    found == placed.end() ? SIZE_MAX : found->second.first,
                                    found == placed.end() ? SIZE_MAX : found->second.second});
    }
    return staged;
}

void LoraStore::commit_staged(const StagedSlot& staged) {
    const std::int32_t slot = staged.slot_;
    if (slot < 0 || slot >= slots_) { throw std::invalid_argument("lora_store: staged slot is out of range"); }
    const auto t0 = std::chrono::steady_clock::now();
    const ScopedDevice on_store(device_ < 0 ? current_device() : device_);
    cudaStream_t stream = nullptr;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    try {
        for (const auto& region : staged.zeros_) {
            CUDA_CHECK(cudaMemsetAsync(region.device, 0, region.bytes, stream));
        }
        for (const auto& region : staged.copies_) {
            CUDA_CHECK(cudaMemcpyAsync(region.device, static_cast<const char*>(staged.host_) + region.offset,
                                       region.bytes, cudaMemcpyHostToDevice, stream));
        }
    } catch (...) {
        (void)cudaStreamSynchronize(stream);
        (void)cudaStreamDestroy(stream);
        throw;
    }
    const auto t_enqueued = std::chrono::steady_clock::now();
    // The host mirrors (the expert cache's CPU copies) are refreshed while the copies run: the
    // image is read-only to both, and several threads keep this under the bus transfer.
    std::size_t mirror_bytes = 0;
    {
        const char* image = static_cast<const char*>(staged.host_);
        const auto refresh = [&](std::size_t first, std::size_t last) {
            for (std::size_t i = first; i < last; ++i) {
                const auto& mirror = staged.mirrors_[i];
                Bank& bank         = banks_.at(Key{mirror.weight, mirror.port});
                if (!bank.host_a.empty()) {
                    auto* a = bank.host_a.data() + static_cast<std::size_t>(slot) * bank.view.a_stride;
                    auto* b = bank.host_b.data() + static_cast<std::size_t>(slot) * bank.view.b_stride;
                    if (mirror.a == SIZE_MAX) {
                        std::fill_n(a, bank.view.a_stride, std::uint16_t{0});
                        std::fill_n(b, bank.view.b_stride, std::uint16_t{0});
                    } else {
                        std::memcpy(a, image + mirror.a,
                                    static_cast<std::size_t>(bank.view.a_stride) * sizeof(std::uint16_t));
                        std::memcpy(b, image + mirror.b,
                                    static_cast<std::size_t>(bank.view.b_stride) * sizeof(std::uint16_t));
                    }
                }
                if (!bank.host_affine.empty()) {
                    for (int half = 0; half < 2; ++half) {
                        std::fill_n(bank.host_affine.data() +
                                        (static_cast<std::size_t>(half) * slots_ + slot) * bank.view.n,
                                    bank.view.n, 0.0F);
                    }
                }
            }
        };
        for (const auto& mirror : staged.mirrors_) {
            const Bank& bank = banks_.at(Key{mirror.weight, mirror.port});
            mirror_bytes += (static_cast<std::size_t>(bank.view.a_stride) + bank.view.b_stride) * sizeof(std::uint16_t);
        }
        const std::size_t count   = staged.mirrors_.size();
        const std::size_t workers = count < 64 ? 1 : 8;
        std::vector<std::thread> threads;
        for (std::size_t w = 1; w < workers; ++w) {
            threads.emplace_back(refresh, count * w / workers, count * (w + 1) / workers);
        }
        refresh(0, count / workers);
        for (auto& thread : threads) { thread.join(); }
    }
    const auto t_mirrored = std::chrono::steady_clock::now();
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaStreamDestroy(stream));
    // Plain LoRA replaces no base weight: every replacement pointer of the slot goes back to none.
    for (auto& [key, replacement] : replacements_) {
        void* empty = nullptr;
        CUDA_CHECK(cudaMemcpy(replacement.table + slot, &empty, sizeof(empty), cudaMemcpyHostToDevice));
        replacement.values[slot].reset();
    }
    const auto ms = [](auto a, auto b) { return std::chrono::duration<double, std::milli>(b - a).count(); };
    std::fprintf(stderr,
                 "lora: in-place commit slot %d on device %d: %zu copies (%.1f MiB), %zu clears, %zu mirrors "
                 "(%.1f MiB): enqueue %.1f ms, mirrors %.1f ms, copies done %.1f ms\n",
                 slot, device_, staged.copies_.size(), static_cast<double>(staged.bytes_) / (1 << 20),
                 staged.zeros_.size(), staged.mirrors_.size(), static_cast<double>(mirror_bytes) / (1 << 20),
                 ms(t0, t_enqueued), ms(t_enqueued, t_mirrored), ms(t0, std::chrono::steady_clock::now()));
}

std::uint32_t LoraStore::slot_version(std::int32_t slot) const noexcept {
    return versions_ && slot >= 0 && slot < slots_ ? versions_[slot].load(std::memory_order_relaxed) : 0U;
}

void LoraStore::set_slot_version(std::int32_t slot, std::uint32_t version) noexcept {
    if (versions_ && slot >= 0 && slot < slots_) { versions_[slot].store(version, std::memory_order_relaxed); }
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
thread_local bool t_base_round = false;
} // namespace

bool lora_enabled() {
    const LoraStore* store = engine_slot<LoraStoreSet>().peek(LoraStore::current_device());
    return store != nullptr && store->active();
}
bool lora_active() { return !t_base_round && lora_enabled(); }
void lora_set_active(bool active) { lora_store_for_current_device().set_active(active); }

ScopedLoraBaseRound::ScopedLoraBaseRound(bool base) noexcept : previous_(t_base_round) {
    t_base_round = base;
}
ScopedLoraBaseRound::~ScopedLoraBaseRound() { t_base_round = previous_; }
bool lora_base_round() noexcept { return t_base_round; }

void lora_set_round(const LoraRound& round) { t_round = round; }
void lora_clear_round() { t_round = LoraRound{}; }
const LoraRound& lora_current_round() { return t_round; }

} // namespace sinfer::ops

namespace sinfer::ops {
void LoraStore::register_replacement(const void* key, std::int32_t port) {
    replacements_.try_emplace(Key{key, port});
}
const void* const* LoraStore::replacement_table(const void* key, std::int32_t port) const {
    const auto it = replacements_.find(Key{key, port});
    return it == replacements_.end() ? nullptr : it->second.table;
}
void LoraStore::register_auto(const void* key, AutoProjection projection) {
    auto& entries = auto_[key];
    for (const auto& entry : entries) { if (entry.port == projection.port) { return; } }
    entries.push_back(projection);
}
const std::vector<LoraStore::AutoProjection>* LoraStore::auto_projections(const void* key) const {
    const auto it = auto_.find(key);
    return it == auto_.end() ? nullptr : &it->second;
}
void lora_auto_linear(const Weight& base, const Tensor& x, Tensor& out, cudaStream_t stream) {
    if (!lora_active() || !lora_current_round().valid()) { return; }
    auto& store = lora_store_for_current_device();
    const auto* entries = store.auto_projections(base.qdata);
    if (!entries) { return; }
    const auto& round = lora_current_round();
    for (const auto& entry : *entries) {
        if (entry.port >= kLoraBiasPort) { continue; }
        const auto* bank = store.find(base.qdata, entry.port);
        if (!bank) { continue; }
        if (const auto* replacement = store.replacement_table(base.qdata, entry.port)) {
            lora_replace_linear(x, out, replacement, round.slots ? *round.slots : Tensor{}, round.uniform_cell, stream);
        }
        for (int offset = 0; offset < x.ne[1]; offset += store.scratch_tokens()) {
            const int tokens = std::min(store.scratch_tokens(), x.ne[1] - offset);
            auto input = x.slice(1, offset, tokens);
            Tensor output(static_cast<std::byte*>(out.data) + offset * out.nb[1] + entry.offset * 2,
                          DType::BF16, {entry.rows, tokens});
            output.nb[1] = out.nb[1];
            auto scratch = store.scratch(tokens);
            auto slots = round.slots ? round.slots->slice(0, offset, tokens) : Tensor{};
            const LoraBank* banks[]{bank}; Tensor* outputs[]{&output};
            lora_delta_fused(input, banks, outputs, 1, slots, round.uniform_cell, scratch, stream);
        }
    }
}
void lora_auto_bias(const Tensor& base_bias, Tensor& out, cudaStream_t stream) {
    if (!lora_active() || !lora_current_round().valid()) { return; }
    auto& store = lora_store_for_current_device();
    const auto* entries = store.auto_projections(base_bias.data);
    if (!entries) { return; }
    const auto& round = lora_current_round();
    for (const auto& entry : *entries) {
        if (entry.port < kLoraBiasPort) { continue; }
        const auto* bank = store.find(base_bias.data, entry.port);
        Tensor output(static_cast<std::byte*>(out.data) + entry.offset * 2, DType::BF16,
                      {entry.rows, static_cast<int>(out.numel() / out.ne[0])});
        output.nb[1] = out.nb[1];
        lora_shift(*bank, round.slots ? *round.slots : Tensor{}, round.uniform_cell, output, stream);
    }
}
void lora_auto_embedding(const Weight& base, const Tensor& ids, Tensor& out, cudaStream_t stream) {
    if (!lora_active() || !lora_current_round().valid()) { return; }
    auto& store = lora_store_for_current_device();
    const auto* bank = store.find(base.qdata, kLoraEmbeddingPort);
    if (!bank) { return; }
    const auto& round = lora_current_round();
    if (const auto* replacement = store.replacement_table(base.qdata, kLoraEmbeddingPort)) {
        lora_replace_embedding(ids, out, replacement, round.slots ? *round.slots : Tensor{}, round.uniform_cell, stream);
    }
    lora_embedding(ids, *bank, round.slots ? *round.slots : Tensor{}, round.uniform_cell, out, stream);
}
} // namespace sinfer::ops
