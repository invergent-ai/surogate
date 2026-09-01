// The resident adapter banks (api/ops/lora_store.h).

#include "api/ops/lora_store.h"

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

LoraStore::~LoraStore() {
    for (auto& [key, bank] : banks_) {
        cudaFree(bank.a);
        cudaFree(bank.b);
    }
    cudaFree(scratch_);
    cudaFree(uniform_cell_);
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
    // Wide enough for a split site's low vectors: up to three pairs share one
    // launch, each padded to max_rank.
    scratch_        = upload_zeroed(static_cast<std::size_t>(kLoraFusedPairLimit) * max_rank *
                               max_tokens);
    CUDA_CHECK(cudaMalloc(&uniform_cell_, sizeof(std::int32_t)));
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
        if (frozen_) {
            throw std::logic_error(
                "lora_store: this projection has no bank and the store is frozen -- banks are "
                "created by ensure_banks before serving so captured graphs and the lock-free "
                "find() stay valid; register the module in the target's directory");
        }
        Bank bank;
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

void LoraStore::register_module(std::int32_t layer, std::string module, ModuleBinding binding) {
    if (frozen_) { throw std::logic_error("lora_store: the directory is frozen after ensure_banks"); }
    if (binding.key == nullptr || binding.in <= 0 || binding.out <= 0) {
        throw std::invalid_argument("lora_store: a module binding needs a key and positive shapes");
    }
    directory_.emplace(std::make_pair(layer, std::move(module)), binding);
}

void LoraStore::register_refusal(std::string module, std::string reason) {
    if (frozen_) { throw std::logic_error("lora_store: the directory is frozen after ensure_banks"); }
    refusals_.emplace(std::move(module), std::move(reason));
}

void LoraStore::register_layer_refusal(std::int32_t layer, std::string module, std::string reason) {
    if (frozen_) { throw std::logic_error("lora_store: the directory is frozen after ensure_banks"); }
    layer_refusals_.emplace(std::make_pair(layer, std::move(module)), std::move(reason));
}

void LoraStore::ensure_banks() {
    if (slots_ == 0) { throw std::logic_error("lora_store: configure() first"); }
    for (const auto& [where, binding] : directory_) {
        const Key key{binding.key, binding.port};
        if (banks_.find(key) != banks_.end()) { continue; }
        Bank bank;
        bank.a             = upload_zeroed(static_cast<std::size_t>(slots_) * max_rank_ * binding.in);
        bank.b             = upload_zeroed(static_cast<std::size_t>(slots_) * binding.out * max_rank_);
        bank.view.a        = bank.a;
        bank.view.b        = bank.b;
        bank.view.a_stride = static_cast<std::int64_t>(max_rank_) * binding.in;
        bank.view.b_stride = static_cast<std::int64_t>(binding.out) * max_rank_;
        bank.view.rank     = max_rank_;
        bank.view.n        = binding.out;
        bank.view.k        = binding.in;
        banks_.emplace(key, bank);
    }
    frozen_ = true;
}

void LoraStore::set_module_slot(std::int32_t layer, const std::string& module, std::int32_t slot,
                                const std::vector<std::uint16_t>& a,
                                const std::vector<std::uint16_t>& b, std::int32_t rank,
                                std::int32_t in_dim, std::int32_t out_dim, float scale) {
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
    set_slot(binding.key, binding.port, slot, a, b, rank, in_dim, out_dim, scale);
}

void LoraStore::clear_slot(std::int32_t slot) {
    if (slot < 0 || slot >= slots_) { return; }
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
    if (uniform_cell_ == nullptr) { return; }
    // Stream-ordered so it lands before the round that reads it, and captured as a
    // node when the round is being recorded.
    CUDA_CHECK(cudaMemcpyAsync(uniform_cell_, &slot, sizeof(slot), cudaMemcpyHostToDevice, stream));
}

LoraStore& lora_store_for_current_device() { return engine_slot<LoraStore>(); }

namespace {
thread_local LoraRound t_round;
} // namespace

bool lora_active() { return engine_slot<LoraStore>().active(); }
void lora_set_active(bool active) { engine_slot<LoraStore>().set_active(active); }

void lora_set_round(const LoraRound& round) { t_round = round; }
void lora_clear_round() { t_round = LoraRound{}; }
const LoraRound& lora_current_round() { return t_round; }

} // namespace sinfer::ops
