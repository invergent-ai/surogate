#pragma once

// Resident adapters: PEFT tensors uploaded once into stacked banks, keyed by the
// base weight they adapt.
//
// The variant methods that run a projection do not take a layer index -- they
// take the layer's weights -- so a bank is keyed by the base weight's device
// pointer. Each layer's projection has its own, which makes the pointer a
// sufficient identity and leaves every projection signature unchanged.
//
// Every bank is padded to (slots, max_rank), so a shorter adapter is zero-padded
// rather than given a narrower bank. That costs a little memory -- a rank-8
// adapter in a rank-64 bank wastes seven eighths of its slot, and the slot is
// kilobytes -- and buys the thing that matters: one launch geometry regardless of
// which adapters a round happens to touch, which is what a CUDA graph can hold
// and what lets tokens of different adapters share a kernel.

#include "api/ops/lora.h"

#include <cstdint>
#include <map>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace sinfer::ops {

class LoraStore {
public:
    /// Reserves `slots` adapters of at most `max_rank` for every bank.
    void configure(std::int32_t slots, std::int32_t max_rank, std::int32_t max_tokens);

    /// Writes one adapter's A/B into `slot` of the bank for `base_key`, creating
    /// the bank on first use. `a` is [rank, in] and `b` is [out, rank] in host
    /// BF16; both are zero-padded up to the bank's max_rank. `scale` (PEFT's
    /// alpha/r) is folded into A here, once.
    /// `port` distinguishes outputs that share a base weight. q, k and v come out
    /// of one fused projection, so the weight pointer alone cannot say which of
    /// them an adapter belongs to; the pair can.
    void set_slot(const void* base_key, std::int32_t port, std::int32_t slot,
                  const std::vector<std::uint16_t>& a, const std::vector<std::uint16_t>& b,
                  std::int32_t rank, std::int32_t in_dim, std::int32_t out_dim, float scale);

    /// Where an adapter module lands on this model: the projection's bank key and
    /// which of its outputs, plus the shapes an adapter must have to fit it.
    struct ModuleBinding {
        const void* key   = nullptr;
        std::int32_t port = 0;
        std::int32_t in   = 0;
        std::int32_t out  = 0;
    };

    /// Target startup registers where every adaptable module of every layer lives,
    /// and why the non-adaptable ones are refused. The directory is what makes
    /// loading target-agnostic afterwards: an adapter names "layers.7.down_proj"
    /// and the store knows the rest, whether the load happens at startup or from a
    /// runtime endpoint.
    void register_module(std::int32_t layer, std::string module, ModuleBinding binding);
    /// A module refused on every layer (e.g. gate/up fused into SwiGLU).
    void register_refusal(std::string module, std::string reason);
    /// A module refused on one layer (e.g. attention names on a linear-attention layer).
    void register_layer_refusal(std::int32_t layer, std::string module, std::string reason);

    /// Creates every registered bank, zero-filled, then freezes the directory.
    ///
    /// Banks must all exist before the first round is captured, for two reasons
    /// with the same root. A captured graph records only the launches it sees, and
    /// the hooks launch only for projections that have a bank -- so a bank created
    /// later would leave every already-captured graph without its kernels, and a
    /// runtime-loaded adapter silently absent under graphs. And creation inserts
    /// into the map the hot-path find() reads without a lock, which is only safe
    /// while nothing serves. After the freeze, loading an adapter only writes into
    /// memory that already exists.
    void ensure_banks();

    /// Writes one adapter module into `slot`, resolving (layer, module) through
    /// the directory. Refusals and missing bindings throw with the module named.
    void set_module_slot(std::int32_t layer, const std::string& module, std::int32_t slot,
                         const std::vector<std::uint16_t>& a, const std::vector<std::uint16_t>& b,
                         std::int32_t rank, std::int32_t in_dim, std::int32_t out_dim, float scale);

    [[nodiscard]] bool has_bindings() const noexcept { return !directory_.empty(); }

    /// Zeroes a slot across every bank, so a token selecting it adds nothing.
    /// This is what unloading an adapter does: the memory stays, the contribution
    /// goes, and a request already in flight that still names the slot degrades to
    /// the base model rather than reading weights that were freed underneath it.
    void clear_slot(std::int32_t slot);

    /// The bank for a projection, or nullptr when it has none. Hot path.
    [[nodiscard]] const LoraBank* find(const void* base_key, std::int32_t port) const noexcept {
        const auto found = banks_.find(Key{base_key, port});
        return found == banks_.end() ? nullptr : &found->second.view;
    }

    /// Scratch for the intermediate `A · x`, sized once for the widest round the
    /// deployment allows. It belongs to the store rather than the round so its
    /// address never moves: a captured graph records it once and every replay
    /// writes the same buffer, which is what the per-call arena allocation this
    /// replaced could not promise.
    [[nodiscard]] Tensor scratch(std::int32_t tokens) const;

    /// Device cell holding the slot a uniform (prefill) round uses. It is memory
    /// rather than a launch argument so a captured prefill graph re-reads it on
    /// every replay instead of freezing the slot it was captured with.
    [[nodiscard]] const std::int32_t* uniform_cell() const noexcept { return uniform_cell_; }
    void write_uniform_slot(std::int32_t slot, cudaStream_t stream) const;

    [[nodiscard]] bool empty() const noexcept { return banks_.empty(); }
    /// True once any adapter machinery is live for this engine; the projection
    /// hooks read it to skip the lookup in the common case.
    [[nodiscard]] bool active() const noexcept { return active_; }
    void set_active(bool active) noexcept { active_ = active; }
    [[nodiscard]] std::int32_t slots() const noexcept { return slots_; }
    [[nodiscard]] std::int32_t max_rank() const noexcept { return max_rank_; }

    ~LoraStore();
    LoraStore()                            = default;
    LoraStore(const LoraStore&)            = delete;
    LoraStore& operator=(const LoraStore&) = delete;

private:
    struct Bank {
        LoraBank view;
        void* a = nullptr;
        void* b = nullptr;
    };
    struct Key {
        const void* weight = nullptr;
        std::int32_t port  = 0;
        bool operator==(const Key& other) const noexcept {
            return weight == other.weight && port == other.port;
        }
    };
    struct KeyHash {
        std::size_t operator()(const Key& key) const noexcept {
            return std::hash<const void*>{}(key.weight) ^ (static_cast<std::size_t>(key.port) << 1U);
        }
    };
    std::unordered_map<Key, Bank, KeyHash> banks_;
    std::map<std::pair<std::int32_t, std::string>, ModuleBinding> directory_;
    std::map<std::string, std::string> refusals_;
    std::map<std::pair<std::int32_t, std::string>, std::string> layer_refusals_;
    bool frozen_ = false;
    void* scratch_          = nullptr;
    std::int32_t* uniform_cell_ = nullptr;
    std::int32_t scratch_tokens_ = 0;
    std::int32_t slots_    = 0;
    std::int32_t max_rank_ = 0;
    bool active_           = false;
};

/// The engine-bound store (via the thread's ops context; the process default
/// serves single-engine paths and tools). The name is historical -- an engine
/// serves one device, so per-engine and per-device coincide.
[[nodiscard]] LoraStore& lora_store_for_current_device();

/// True when any adapter is resident, so the projection hooks skip the lookup in
/// the common case. Re-read after a load or unload.
[[nodiscard]] bool lora_active();
void lora_set_active(bool active);

/// The round's adapter selection, published by the decode schedule and read by
/// the projection hooks.
///
/// It is a thread-local rather than a parameter because the hooks sit inside
/// variant methods whose signatures the family shares; threading a round context
/// through all of them would change every target's interface to serve one
/// feature. Both tensors are frame-resident, so a captured graph records their
/// addresses once and replays against whatever the round wrote.
struct LoraRound {
    const Tensor* slots = nullptr; ///< I32 [tokens], -1 for base-model tokens
    /// One slot for every column, used when `slots` is null. A prefill chunk is
    /// exactly this: all of its columns belong to the request being prefilled, so
    /// a vector would hold one value repeated and need refilling per chunk.
    /// Set when the round is uniform; the value lives in the store's device cell.
    bool uniform = false;
    Tensor scratch;                ///< BF16, at least max_rank * tokens
    bool valid() const noexcept {
        return scratch.data != nullptr && (slots != nullptr || uniform);
    }
};
void lora_set_round(const LoraRound& round);
void lora_clear_round();
[[nodiscard]] const LoraRound& lora_current_round();

} // namespace sinfer::ops
