#pragma once

// Expert LoRA for Inkling-2-Mini's mixture (api/ops/inkling_moe.h): the load-side binding of the
// NVFP4 (layers 0-19) and MXFP8 (layers 20-23) expert banks into the engine's LoRA slot store, and
// the per-call adapter selection the target hands `inkling_moe`.
//
// Separate from family/impl/lora_bind.h on purpose: `bind_lora_moe` there binds the codec-generic
// `SparseMoeWeights` (and registers base views for DoRA); Inkling's experts are a different op with
// its own bank-table contract. Nothing in the Qwen / GLM / Gemma paths includes this header.
//
// Bank table (the order `inkling_moe`'s kernels read, inkling_gemm_common.cuh `lora_bank`):
//   [0] router: none, [1] bias: none, then per expert e: [2 + 3e] gate, [3 + 3e] up, [4 + 3e] down
//   (ports 100 + 3e + {0, 1, 2} under the key `weights.router`).
// Module names (PEFT, the trainer's export `<prefix>.layers.<L>.mlp.experts.<e>.<proj>`, L = the PHYSICAL
// layer: a looped core layer's three visits share one weight set and so one bank table):
//   mlp.experts.<e>.gate_up_proj  [2304 out, 2304 in] over w13's STORED rows, which interleave gate and up
//                                 (row 2j = gate j, row 2j + 1 = up j): gate = even B rows, up = odd B rows,
//                                 A shared (the kit's recipe: r16 / alpha 32 on gate_up_proj + down_proj)
//   mlp.experts.<e>.gate_proj / up_proj  [1152, 2304] (split adapters, if an exporter writes them)
//   mlp.experts.<e>.down_proj     [2304, 1152]
// Bank rank: `expert_rank` (--max-expert-lora-rank; 0 = --max-lora-rank) sizes every expert bank, so an engine can
// serve attention adapters at a higher rank than the experts' (SkyRL's preset: attention r64, experts r16) without
// padding 128 x 3 expert banks a layer to the attention rank. The expert kernels read each bank's own rank.
// Plain LoRA only: the kernels read A and B (alpha/r folded into A by the store), never a bank's gain /
// bias rows, and no base view is registered, so a DoRA magnitude or a bias is refused at load
// (`prepare_affine` -> "adapter base projection was not registered"). The router takes no adapter.
// Hot swap: every bank lives at a fixed address in the store's arena, so `stage_slot` / `commit_staged`
// (and `set_device_module`, `clear_slot`) rewrite a slot in place between rounds and captured graphs read
// the new bytes on their next replay.

#include "api/ops/inkling_moe.h"
#include "api/ops/lora_store.h"

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace sinfer::family {

inline constexpr std::int32_t kInklingExpertLoraPort = 100;

/// Registers physical layer `layer`'s expert adapters (both formats: the banks are BF16 on the
/// GEMMs' activation sides, so their shapes do not depend on the expert codec). Call inside the
/// engine's construction window, before `ensure_banks` (as `bind_lora_moe`).
inline void bind_lora_inkling_moe(ops::LoraStore& store, int layer, const ops::InklingMoeWeights& weights,
                                  std::int32_t expert_rank = 0) {
    using G       = ops::InklingMoeGeometry;
    using Binding = ops::LoraStore::ModuleBinding;
    if (weights.router == nullptr) {
        throw std::invalid_argument("inkling expert LoRA: the layer's router weight (the bank key) is not bound");
    }
    if (expert_rank < 0 || expert_rank > store.max_rank()) {
        throw std::invalid_argument("inkling expert LoRA: --max-expert-lora-rank must lie in [0, --max-lora-rank]");
    }
    const std::int32_t rank = expert_rank > 0 ? expert_rank : store.max_rank();
    if (rank > G::kMaxAdapterRank) {
        throw std::invalid_argument("inkling expert LoRA supports an expert rank up to " +
                                    std::to_string(G::kMaxAdapterRank) +
                                    " (--max-lora-rank, or --max-expert-lora-rank when it is set)");
    }
    const void* key = weights.router;
    std::vector<Binding> table(2); // [0] router, [1] bias: no adapter
    // The directory strips a single-component "mlp." before it looks refusals up ("mlp.gate" -> "gate").
    for (const char* router : {"gate", "router", "mlp.gate.bias"}) {
        store.register_layer_refusal(layer, router, "Inkling's router takes no adapter in this engine");
    }
    for (std::int32_t expert = 0; expert < G::kExperts; ++expert) {
        const std::int32_t port = kInklingExpertLoraPort + 3 * expert;
        Binding gate{key, port, G::kHidden, G::kIntermediate};
        Binding up{key, port + 1, G::kHidden, G::kIntermediate};
        Binding down{key, port + 2, G::kIntermediate, G::kHidden};
        gate.rank = up.rank = down.rank = expert_rank; // 0 = the store's max_rank (unchanged default)
        const std::string prefix = "mlp.experts." + std::to_string(expert) + ".";
        store.register_module(layer, prefix + "gate_proj", gate);
        store.register_module(layer, prefix + "up_proj", up);
        store.register_module(layer, prefix + "down_proj", down);
        // The fused adapter over w13's interleaved rows: bank row j of gate <- B row 2j, of up <- B row 2j + 1.
        Binding gate_part   = gate;
        gate_part.source_out = G::kGateUpRows;
        gate_part.row_offset = 0;
        gate_part.row_group  = 1;
        gate_part.row_stride = 2;
        Binding up_part     = up;
        up_part.source_out   = G::kGateUpRows;
        up_part.row_offset   = 1;
        up_part.row_group    = 1;
        up_part.row_stride   = 2;
        store.register_module(layer, prefix + "gate_up_proj", gate_part);
        store.register_module(layer, prefix + "gate_up_proj", up_part);
        table.insert(table.end(), {gate, up, down});
    }
    store.register_bank_table(key, table);
}

/// The adapter selection for one `inkling_moe` call of the current round, or nullptr for a base
/// round (no adapter machinery live, a `ScopedLoraBaseRound` open, no published round, or a layer
/// without a table) -- the same test `sparse_moe` makes. `first_column` is the call's first column
/// in the round's slot vector (0 for a whole-round call). `storage` holds the returned value.
inline const ops::InklingMoeAdapters* inkling_moe_round_adapters(const ops::InklingMoeWeights& weights,
                                                                 ops::InklingMoeAdapters& storage,
                                                                 std::int32_t first_column = 0) {
    if (!ops::lora_active()) { return nullptr; }
    const ops::LoraRound& round = ops::lora_current_round();
    if (!round.valid()) { return nullptr; }
    const ops::LoraStore& store = ops::lora_store_for_current_device();
    const ops::LoraBank* table  = store.bank_table(weights.router);
    if (table == nullptr) { return nullptr; }
    // The expert banks' rank (all of a table's banks share it): the store's max_rank unless the engine set
    // --max-expert-lora-rank. It sizes the call's low-rank workspace, exactly as max_rank did before.
    const ops::LoraBank* host = store.host_bank_table(weights.router);
    storage = {};
    storage.table    = table;
    storage.max_rank = host != nullptr ? host[2].rank : store.max_rank();
    if (round.slots != nullptr) {
        if (first_column < 0 || first_column >= round.slots->ne[0]) {
            throw std::invalid_argument("inkling expert LoRA: the call's columns lie outside the round's slots");
        }
        storage.token_slot = static_cast<const std::int32_t*>(round.slots->data) + first_column;
    } else {
        storage.uniform_slot = round.uniform_cell;
    }
    return &storage;
}

} // namespace sinfer::family
