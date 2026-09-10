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
#include "api/ops/lora_base.h"
#include "api/shared_weights.h"
#include "core/arena.h"

#include <array>
#include <atomic>
#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>
#include <set>
#include <tuple>

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
        // A checkpoint projection may fan out into several serving tensors.
        // Row i of this bank comes from offset + (i/group)*stride + i%group
        // of the checkpoint's B. Zero group/stride denotes contiguous rows.
        std::int32_t source_out = 0;
        std::int32_t row_offset = 0;
        std::int32_t row_group = 0;
        std::int32_t row_stride = 0;
        [[nodiscard]] std::int32_t checkpoint_out() const noexcept { return source_out ? source_out : out; }
        [[nodiscard]] std::int32_t source_row(std::int32_t i) const noexcept {
            return row_group ? row_offset + (i / row_group) * row_stride + i % row_group
                             : row_offset + i;
        }
    };

    /// Target startup registers where every adaptable module of every layer lives,
    /// and why the non-adaptable ones are refused. The directory is what makes
    /// loading target-agnostic afterwards: an adapter names "layers.7.down_proj"
    /// and the store knows the rest, whether the load happens at startup or from a
    /// runtime endpoint.
    void register_module(std::int32_t layer, std::string module, ModuleBinding binding);
    void register_base_weight(const Weight& weight);
    void register_base(const void* key, std::int32_t port, std::vector<LoraBaseView> base,
                       Tensor bias = {}, bool bias_in_output = false);
    /// A module refused on every layer (e.g. gate/up fused into SwiGLU).
    void register_refusal(std::string module, std::string reason);
    /// A module refused on one layer (e.g. attention names on a linear-attention layer).
    void register_layer_refusal(std::int32_t layer, std::string module, std::string reason);

    /// Creates the banks of every registered module that has none yet, zero-filled,
    /// on the device that is current when it is called.
    ///
    /// A pipeline calls this once per stage, with that stage's device current and
    /// that stage's layers freshly registered, so the banks of a layer land on the
    /// device that holds it. A single-device engine calls it once and nothing
    /// changes for it.
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
    /// A graph-stable device directory for projections selected by a device expert id.
    void register_bank_table(const void* key, const std::vector<ModuleBinding>& bindings);
    [[nodiscard]] const LoraBank* bank_table(const void* key) const noexcept;
    [[nodiscard]] const LoraBank* host_bank_table(const void* key) const noexcept;
    struct AutoProjection { std::int32_t port = 0, offset = 0, rows = 0; };
    void register_auto(const void* key, AutoProjection projection);
    void register_replacement(const void* key, std::int32_t port);
    [[nodiscard]] const void* const* replacement_table(const void* key, std::int32_t port) const;
    [[nodiscard]] const std::vector<AutoProjection>* auto_projections(const void* key) const;
    [[nodiscard]] std::int32_t scratch_tokens() const noexcept { return scratch_tokens_; }

    /// Writes one adapter module into `slot`, resolving (layer, module) through
    /// the directory. Refusals and missing bindings throw with the module named.
    void set_module_slot(std::int32_t layer, const std::string& module, std::int32_t slot,
                         const std::vector<std::uint16_t>& a, const std::vector<std::uint16_t>& b,
                         std::int32_t rank, std::int32_t in_dim, std::int32_t out_dim, float scale,
                         const std::vector<float>& magnitude = {}, const std::vector<float>& bias = {},
                         const std::vector<float>& lora_bias = {}, const std::vector<std::uint16_t>& base_weight = {},
                         std::shared_ptr<DeviceArena> prepared_replacement = {});
    void validate_module(std::int32_t layer, const std::string& module,
                         const std::vector<std::uint16_t>& a, const std::vector<std::uint16_t>& b,
                         std::int32_t rank, std::int32_t in_dim, std::int32_t out_dim, float scale) const;
    template<class Payload>
    void validate_payloads(const std::vector<Payload>& payloads) const {
        std::set<std::tuple<std::int32_t, const void*, std::int32_t>> used;
        for (const auto& p : payloads) {
            if (!covers_layer(p.layer)) { continue; }
            validate_module(p.layer, p.module, p.a, p.b, p.rank, p.in_dim, p.out_dim, p.scale);
            auto staged = stage_replacement(p.layer, p.module, p.base_weight);
            (void)prepare_affine(p.layer, p.module, p.a, p.b, p.rank, p.scale,
                                  p.magnitude, p.bias, p.lora_bias, p.base_weight, staged);
            if (staged) { p.prepared_replacements[this] = std::move(staged); }
            for (const auto& part : module_parts(p.layer, p.module)) {
                if (!used.emplace(p.slot, part.key, part.port).second) {
                    throw std::invalid_argument("adapter modules overlap at '" + p.module + "'");
                }
            }
        }
    }
    template<class Payload>
    void set_payload(std::int32_t slot, const Payload& p) {
        std::shared_ptr<DeviceArena> staged;
        if (const auto it = p.prepared_replacements.find(this); it != p.prepared_replacements.end()) {
            staged = std::static_pointer_cast<DeviceArena>(it->second);
        }
        set_module_slot(p.layer, p.module, slot, p.a, p.b, p.rank, p.in_dim, p.out_dim, p.scale,
                         p.magnitude, p.bias, p.lora_bias, p.base_weight, std::move(staged));
        p.prepared_replacements.erase(this);
    }
    void validate_device_module(const DeviceAdapterModule& module) const;
    void set_device_module(std::int32_t slot, const DeviceAdapterModule& module);

    [[nodiscard]] bool has_bindings() const noexcept { return !directory_.empty(); }

    /// Whether the directory holds any module for this layer.
    ///
    /// Under pipeline parallelism a stage registers only the layers it holds, so
    /// this is how a startup payload is matched to the stage that can apply it:
    /// every stage is handed the whole `--lora-modules` list and each takes its
    /// own layers. A module name that is wrong on a layer this stage does hold
    /// still throws, which is the error worth keeping.
    [[nodiscard]] bool covers_layer(std::int32_t layer) const;

    /// Zeroes a slot across every bank, so a token selecting it adds nothing.
    /// The caller must first drain every request that can select this slot.
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
    /// Read once per round by the schedule that publishes `LoraRound`, never per
    /// projection: the hook takes it from the round, so the store is off the hot
    /// path once the round is published.
    [[nodiscard]] const std::int32_t* uniform_cell() const noexcept { return uniform_cell_; }
    void write_uniform_slot(std::int32_t slot, cudaStream_t stream) const;

    [[nodiscard]] bool empty() const noexcept { return banks_.empty(); }
    /// True once any adapter machinery is live for this engine; the projection
    /// hooks read it to skip the lookup in the common case.
    [[nodiscard]] bool active() const noexcept { return active_.load(std::memory_order_relaxed); }
    void set_active(bool active) noexcept { active_.store(active, std::memory_order_relaxed); }
    [[nodiscard]] std::int32_t slots() const noexcept { return slots_; }
    [[nodiscard]] std::int32_t max_rank() const noexcept { return max_rank_; }
    /// The device this store's memory lives on, or -1 before anything is built.
    [[nodiscard]] int device() const noexcept { return device_; }
    /// The calling thread's current CUDA device.
    [[nodiscard]] static int current_device();

    ~LoraStore();
    LoraStore()                            = default;
    LoraStore(const LoraStore&)            = delete;
    LoraStore& operator=(const LoraStore&) = delete;

private:
    struct Bank {
        LoraBank view;
        void* a  = nullptr;
        void* b  = nullptr;
        bool raw = false; ///< individually cudaMalloc'd (directory-less path); arena otherwise
        std::vector<std::uint16_t> host_a, host_b;
        void* affine = nullptr;
        std::vector<float> host_affine;
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
    struct Base { std::vector<LoraBaseView> parts; Tensor bias; bool bias_in_output = false; };
    std::unordered_map<const void*, Weight> base_weights_;
    std::unordered_map<Key, Base, KeyHash> bases_;
    [[nodiscard]] Base base_for(const ModuleBinding& binding) const;
    [[nodiscard]] std::vector<std::vector<float>> prepare_affine(std::int32_t layer,
        const std::string& module, const std::vector<std::uint16_t>& a,
        const std::vector<std::uint16_t>& b, std::int32_t rank, float scale,
        const std::vector<float>& magnitude, const std::vector<float>& bias,
        const std::vector<float>& lora_bias, const std::vector<std::uint16_t>& base_weight, std::shared_ptr<DeviceArena> prepared = {}) const;
    [[nodiscard]] std::shared_ptr<DeviceArena> stage_replacement(std::int32_t layer, const std::string& module,
        const std::vector<std::uint16_t>& weight) const;
    /// The arenas holding every bank plus the round state, allocated by
    /// ensure_banks. Each is an owning DeviceArena, so under sleep mode it joins
    /// the engine's sleepable estate as an Offload region: a slept model's
    /// adapters leave VRAM with it and come back byte-identical. There is one per
    /// ensure_banks call, and earlier ones are kept rather than replaced, so a
    /// second call cannot free the banks the first one handed out.
    std::vector<std::unique_ptr<DeviceArena>> storage_;

    /// The device every bank, the scratch and the cell of this store live on,
    /// learned from the thread that built them. Writes from another thread -- an
    /// adapter upload from an HTTP handler -- bind it first, because a copy
    /// resolves its destination against the current device, not against the
    /// pointer.
    int device_ = -1;
    void* scratch_              = nullptr;
    std::int32_t* uniform_cell_ = nullptr;
    bool raw_round_state_       = false; ///< scratch/cell cudaMalloc'd (directory-less use)
    using ModuleParts = std::vector<ModuleBinding>;
    [[nodiscard]] const ModuleParts& module_parts(std::int32_t layer, const std::string& module) const;
    std::map<std::pair<std::int32_t, std::string>, ModuleParts> directory_;
    std::map<const void*, std::vector<ModuleBinding>> table_bindings_;
    std::map<const void*, const LoraBank*> tables_;
    std::map<const void*, std::vector<LoraBank>> host_tables_;
    std::unordered_map<const void*, std::vector<AutoProjection>> auto_;
    struct Replacement {
        void** table = nullptr;
        std::vector<std::shared_ptr<DeviceArena>> values;
    };
    std::unordered_map<Key, Replacement, KeyHash> replacements_;
    const void* owner_ = nullptr;
    std::map<std::string, std::string> refusals_;
    std::map<std::pair<std::int32_t, std::string>, std::string> layer_refusals_;
    /// True once ensure_banks has created any bank. It stops `set_slot` making
    /// one lazily afterwards, which would insert into the map the hot-path
    /// `find()` reads without a lock, and would leave already-captured graphs
    /// without the kernels for it.
    bool banks_built_ = false;
    std::int32_t scratch_tokens_ = 0;
    std::int32_t slots_    = 0;
    std::int32_t max_rank_ = 0;
    std::atomic<bool> active_{false};
    void ensure_raw_round_state();
};

/// An engine's adapter stores, one per device.
///
/// A single-device engine has exactly one and nothing about it changes. A
/// pipeline has one per stage's device, and that separation is not a
/// convenience: the stages construct concurrently, one thread per card, and each
/// one registers its modules and builds its banks as it goes. Sharing a store
/// would put those inserts next to the lock-free `find()` another stage's graph
/// capture is already calling.
///
/// Lookup is a relaxed atomic load off a small array, because it happens once per
/// adapted projection. Only creation takes the lock, and only during construction.
class LoraStoreSet {
public:
    /// More than any single host holds, and the index is the CUDA ordinal.
    static constexpr int kMaxDevices = 16;

    /// The store for `device`, created on first use.
    [[nodiscard]] LoraStore& for_device(int device);
    /// The store for `device`, or nullptr if it has none. Never creates.
    [[nodiscard]] LoraStore* peek(int device) const noexcept;
    /// Every device that has a store, in ordinal order.
    [[nodiscard]] std::vector<int> devices() const;

    LoraStoreSet() = default;
    LoraStoreSet(const LoraStoreSet&)            = delete;
    LoraStoreSet& operator=(const LoraStoreSet&) = delete;

private:
    std::array<std::atomic<LoraStore*>, kMaxDevices> stores_{};
    std::vector<std::unique_ptr<LoraStore>> owned_;
    mutable std::mutex mutex_;
};

/// The store for the calling thread's current CUDA device, from the engine bound
/// to this thread (the process default serves single-engine paths and tools).
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
    /// The device cell holding the uniform round's slot, on the device this round
    /// runs on. Carried here rather than looked up per projection so that a
    /// pipeline stage reads its own device's cell without the hook asking which
    /// device it is on.
    const std::int32_t* uniform_cell = nullptr;
    bool valid() const noexcept {
        return scratch.data != nullptr && (slots != nullptr || uniform);
    }
};
void lora_set_round(const LoraRound& round);
void lora_clear_round();
[[nodiscard]] const LoraRound& lora_current_round();

inline constexpr int kLoraEmbeddingPort = 800;
inline constexpr int kLoraAutomaticPort = 810;
inline constexpr int kLoraBiasPort = 900;
void lora_auto_linear(const Weight& base, const Tensor& x, Tensor& out, cudaStream_t stream);
void lora_auto_bias(const Tensor& base_bias, Tensor& out, cudaStream_t stream);
void lora_auto_embedding(const Weight& base, const Tensor& ids, Tensor& out, cudaStream_t stream);

} // namespace sinfer::ops
