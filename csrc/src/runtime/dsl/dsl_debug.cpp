// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0

#include "runtime/dsl/dsl_debug.h"

#include <algorithm>
#include <cstddef>
#include <limits>
#include <string>
#include <unordered_map>
#include <utility>

#include "runtime/dsl/dsl_model.h"
#include "runtime/dsl/graph_compiler.h"
#include "runtime/executor/graph_executor.h"

namespace dsl {

const char* debug_graph_kind_name(DebugGraphKind k) {
    switch (k) {
        case DebugGraphKind::Forward: return "forward";
        case DebugGraphKind::Backward: return "backward";
    }
    return "unknown";
}

namespace {

// ----------------------------------------------------------------------------
// Helpers
// ----------------------------------------------------------------------------

const CompiledGraph* compiled_graph_for(const DslModel& model, bool is_backward) {
    const GraphExecutor* exec = model.graph_executor();
    if (!exec) {
        return nullptr;
    }
    return is_backward ? exec->compiled_backward() : exec->compiled_forward();
}

constexpr std::uint64_t kUnassigned64 = (std::numeric_limits<std::uint64_t>::max)();

DebugTensorEntry make_tensor_entry(const CompiledGraph& graph, int tid, DebugGraphKind graph_kind) {
    DebugTensorEntry e;
    e.graph = graph_kind;
    e.tid = tid;
    if (tid >= 0 && static_cast<std::size_t>(tid) < graph.tensor_id_to_name.size()) {
        e.name = graph.tensor_id_to_name[static_cast<std::size_t>(tid)];
    }
    const TensorMeta* meta = graph.meta_for_tensor_id(tid);
    if (!meta) {
        e.kind = "Unknown";
        e.region = "Unknown";
        return e;
    }
    e.kind = tensor_kind_name(meta->kind);
    e.region = region_kind_name(meta->region);
    e.block_layer_idx = meta->block_layer_idx;
    if (meta->offset == SIZE_MAX) {
        e.offset = kUnassigned64;
        e.offset_assigned = false;
    } else {
        e.offset = static_cast<std::uint64_t>(meta->offset);
        e.offset_assigned = true;
    }
    e.bytes = static_cast<std::uint64_t>(meta->bytes);
    e.retain_through_forward = meta->retain_through_forward;
    e.base_param_tid = meta->base_param_tid;
    e.base_producer_tid = meta->base_producer_tid;
    e.base_grad_tid = meta->base_grad_tid;
    e.is_blocks = meta->is_blocks();
    e.is_d_blocks = meta->is_d_blocks();
    e.is_cross_layer = meta->is_cross_layer();
    e.is_moe_offsets = meta->is_moe_offsets();
    e.is_moe_gather = meta->is_moe_gather();
    return e;
}

void convert_phase_node(const PhaseNode& src, DebugPhaseNode& dst) {
    dst.kind = phase_kind_name(src.kind);
    dst.label = src.label;
    dst.op_start = static_cast<std::uint64_t>(src.op_start);
    dst.op_end = static_cast<std::uint64_t>(src.op_end);
    dst.block_index = src.block_index;
    dst.children.clear();
    dst.children.reserve(src.children.size());
    for (const auto& child : src.children) {
        dst.children.emplace_back();
        convert_phase_node(child, dst.children.back());
    }
}

void fill_graph_arena(DebugGraphArena& out,
                      const CompiledGraph* graph,
                      const PhaseArenas& arenas,
                      DebugGraphKind kind) {
    out.graph = kind;
    if (!graph) {
        return;
    }
    out.name = graph->name;
    out.num_tensors = static_cast<std::uint64_t>(graph->num_tensors);
    out.num_ops = static_cast<std::uint64_t>(graph->ops.size());
    out.persistent_bytes = static_cast<std::uint64_t>(graph->persistent_bytes);
    out.accumulator_bytes = static_cast<std::uint64_t>(graph->accumulator_bytes);
    out.fwd_stack_peak = static_cast<std::uint64_t>(graph->fwd_stack_peak);
    out.bwd_stack_peak = static_cast<std::uint64_t>(graph->bwd_stack_peak);
    out.save_for_bwd_bytes = static_cast<std::uint64_t>(graph->save_for_bwd_bytes);
    out.save_for_bwd_block_bytes.assign(graph->save_for_bwd_block_bytes.begin(), graph->save_for_bwd_block_bytes.end());
    out.layout_hash = graph->layout_hash;

    // Coverage (validate_arena_coverage requires the arenas to be allocated; if
    // they aren't, we still return zeros — the caller can check arenas_allocated).
    if (arenas.allocated) {
        ArenaCoverage cov = validate_arena_coverage(arenas, *graph);
        out.tids_covered = static_cast<std::uint64_t>(cov.covered);
        out.tids_total = static_cast<std::uint64_t>(cov.total);
        out.tids_size_exceeded = static_cast<std::uint64_t>(cov.size_exceeded);
        OpOperandCoverage op_cov = validate_op_operand_coverage(arenas, *graph);
        out.covered_inputs = static_cast<std::uint64_t>(op_cov.covered_inputs);
        out.covered_outputs = static_cast<std::uint64_t>(op_cov.covered_outputs);
        out.total_inputs = static_cast<std::uint64_t>(op_cov.total_inputs);
        out.total_outputs = static_cast<std::uint64_t>(op_cov.total_outputs);
    }

    // Per-region tid counts + byte totals.
    std::unordered_map<std::string, DebugRegionCount> by_region;
    for (int tid = 0; tid < graph->num_tensors; ++tid) {
        const TensorMeta* meta = graph->meta_for_tensor_id(tid);
        if (!meta) {
            continue;
        }
        std::string region_name = region_kind_name(meta->region);
        auto& rc = by_region[region_name];
        rc.region = region_name;
        rc.tid_count += 1;
        rc.tid_bytes += static_cast<std::uint64_t>(meta->bytes);
    }
    out.regions.reserve(by_region.size());
    for (auto& kv : by_region) {
        out.regions.push_back(std::move(kv.second));
    }
    std::sort(out.regions.begin(), out.regions.end(), [](const DebugRegionCount& a, const DebugRegionCount& b) {
        return a.region < b.region;
    });
}

}  // namespace

// ----------------------------------------------------------------------------
// Public collectors
// ----------------------------------------------------------------------------

std::vector<DebugTensorEntry> collect_tensor_layout(const DslModel& model) {
    std::vector<DebugTensorEntry> out;
    const CompiledGraph* fwd = compiled_graph_for(model, /*is_backward=*/false);
    const CompiledGraph* bwd = compiled_graph_for(model, /*is_backward=*/true);
    const std::size_t reserve = (fwd ? fwd->num_tensors : 0) + (bwd ? bwd->num_tensors : 0);
    out.reserve(reserve);
    if (fwd) {
        for (int tid = 0; tid < fwd->num_tensors; ++tid) {
            out.push_back(make_tensor_entry(*fwd, tid, DebugGraphKind::Forward));
        }
    }
    if (bwd) {
        for (int tid = 0; tid < bwd->num_tensors; ++tid) {
            out.push_back(make_tensor_entry(*bwd, tid, DebugGraphKind::Backward));
        }
    }
    return out;
}

DebugArenaSummary collect_arena_summary(const DslModel& model) {
    DebugArenaSummary s{};
    const GraphExecutor* exec = model.graph_executor();
    if (!exec) {
        return s;
    }
    const PhaseArenas& arenas = exec->phase_arenas();
    s.arenas_allocated = arenas.allocated;
    s.arena_persistent_bytes = static_cast<std::uint64_t>(arenas.persistent_bytes);
    s.arena_persistent_activation_bytes = static_cast<std::uint64_t>(arenas.persistent_activation_bytes);
    s.arena_accumulator_bytes = static_cast<std::uint64_t>(arenas.accumulator_bytes);
    s.arena_fwd_stack_bytes = static_cast<std::uint64_t>(arenas.fwd_stack_bytes);
    s.arena_bwd_stack_bytes = static_cast<std::uint64_t>(arenas.bwd_stack_bytes);
    s.arena_save_for_bwd_bytes = static_cast<std::uint64_t>(arenas.save_for_bwd_bytes);
    s.arena_unified_stack_bytes = static_cast<std::uint64_t>(arenas.unified_stack_bytes);
    s.arena_bwd_cross_layer_bytes = static_cast<std::uint64_t>(arenas.bwd_cross_layer_bytes);
    s.arena_moe_saved_bytes = static_cast<std::uint64_t>(arenas.moe_saved_bytes);
    s.arena_save_for_bwd_block_bases.assign(arenas.save_for_bwd_block_bases.begin(),
                                            arenas.save_for_bwd_block_bases.end());
    fill_graph_arena(s.forward, exec->compiled_forward(), arenas, DebugGraphKind::Forward);
    fill_graph_arena(s.backward, exec->compiled_backward(), arenas, DebugGraphKind::Backward);
    return s;
}

DebugPhaseTree collect_phase_tree(const DslModel& model, bool is_backward) {
    DebugPhaseTree out;
    out.graph = is_backward ? DebugGraphKind::Backward : DebugGraphKind::Forward;
    const CompiledGraph* graph = compiled_graph_for(model, is_backward);
    if (!graph) {
        return out;
    }
    if (graph->phase_tree.has_value()) {
        out.present = true;
        convert_phase_node(*graph->phase_tree, out.root);
    }
    out.instruction_stream.reserve(graph->instruction_stream.size());
    for (const auto& inst : graph->instruction_stream) {
        DebugInstruction di;
        di.kind = inst_kind_name(inst.kind);
        di.phase_kind = phase_kind_name(inst.phase_kind);
        di.block_index = inst.block_index;
        di.op_start = static_cast<std::uint64_t>(inst.op_start);
        di.op_end = static_cast<std::uint64_t>(inst.op_end);
        di.eager = inst.eager;
        out.instruction_stream.push_back(std::move(di));
    }
    return out;
}

namespace {

//! Build `first_write_op[tid]` + `last_use_op[tid]` tables in one pass over
//! ops + CompiledGraph::last_use_index. SIZE_MAX means "never written" / "no
//! last use record" — those tids get treated as having an empty lifetime and
//! thus never participate in aliasing pairs.
struct TidLifetimes {
    std::vector<std::size_t> first_write;  //!< Earliest op with this tid in outputs
    std::vector<std::size_t> last_use;     //!< Latest op that reads/writes this tid
};

TidLifetimes compute_tid_lifetimes(const CompiledGraph& graph) {
    TidLifetimes lt;
    lt.first_write.assign(static_cast<std::size_t>(graph.num_tensors), SIZE_MAX);
    lt.last_use.assign(static_cast<std::size_t>(graph.num_tensors), SIZE_MAX);
    for (std::size_t op_idx = 0; op_idx < graph.ops.size(); ++op_idx) {
        const auto& op = graph.ops[op_idx];
        for (const auto& ref : op.outputs) {
            if (ref.tensor_id < 0 || ref.tensor_id >= graph.num_tensors) {
                continue;
            }
            auto& fw = lt.first_write[static_cast<std::size_t>(ref.tensor_id)];
            if (fw == SIZE_MAX) {
                fw = op_idx;
            }
        }
    }
    for (const auto& kv : graph.last_use_index) {
        int tid = graph.find_tensor_id(kv.first);
        if (tid >= 0 && tid < graph.num_tensors) {
            lt.last_use[static_cast<std::size_t>(tid)] = kv.second;
        }
    }
    return lt;
}

//! Scan one compiled graph for aliasing bugs: tids whose byte ranges overlap
//! AND whose [first_write_op, last_use_op] lifetime intervals overlap. Under
//! correct compilation within-frame coloring deliberately places
//! disjoint-lifetime tids at the same offset; those are NOT bugs and are
//! filtered out here.
void scan_static_aliasing(const CompiledGraph& graph, DebugGraphKind kind, std::vector<DebugAliasingPair>& out) {
    struct TidRange {
        int tid;
        std::uint64_t offset;
        std::uint64_t bytes;
        std::size_t first_write;
        std::size_t last_use;
    };
    const TidLifetimes lt = compute_tid_lifetimes(graph);
    //! Bucket by (region, block_layer_idx). Arena-backed regions only.
    std::unordered_map<std::string, std::vector<TidRange>> buckets;
    buckets.reserve(static_cast<std::size_t>(graph.num_tensors));
    for (int tid = 0; tid < graph.num_tensors; ++tid) {
        const TensorMeta* meta = graph.meta_for_tensor_id(tid);
        if (!meta || meta->region == RegionKind::Unknown) {
            continue;
        }
        if (meta->offset == SIZE_MAX || meta->bytes == 0) {
            continue;
        }
        const std::size_t tidu = static_cast<std::size_t>(tid);
        const std::size_t fw = lt.first_write[tidu];
        const std::size_t lu = lt.last_use[tidu];
        //! Lifetime-less tids (never written / never used) cannot be live,
        //! so they cannot participate in real aliasing.
        if (fw == SIZE_MAX || lu == SIZE_MAX) {
            continue;
        }
        std::string key = region_kind_name(meta->region);
        key.push_back('#');
        key.append(std::to_string(meta->block_layer_idx));
        buckets[key].push_back(
            {tid, static_cast<std::uint64_t>(meta->offset), static_cast<std::uint64_t>(meta->bytes), fw, lu});
    }
    for (auto& kv : buckets) {
        auto& ranges = kv.second;
        if (ranges.size() < 2) {
            continue;
        }
        std::sort(ranges.begin(), ranges.end(), [](const TidRange& a, const TidRange& b) {
            return a.offset < b.offset;
        });
        //! Sweep: for each range, compare against successors whose start < this.end.
        for (std::size_t i = 0; i < ranges.size(); ++i) {
            const std::uint64_t end_i = ranges[i].offset + ranges[i].bytes;
            for (std::size_t j = i + 1; j < ranges.size(); ++j) {
                if (ranges[j].offset >= end_i) {
                    break;
                }
                //! Byte-range overlap — check lifetime overlap before flagging.
                //! Lifetimes [fw, lu] are closed intervals. Overlap iff
                //! max(fw_a, fw_b) <= min(lu_a, lu_b).
                const std::size_t fw_max = std::max(ranges[i].first_write, ranges[j].first_write);
                const std::size_t lu_min = std::min(ranges[i].last_use, ranges[j].last_use);
                if (fw_max > lu_min) {
                    continue;
                }
                const std::uint64_t start = ranges[j].offset;  // == max
                const std::uint64_t end = std::min(end_i, ranges[j].offset + ranges[j].bytes);
                DebugAliasingPair p;
                p.graph = kind;
                const TensorMeta* meta_a = graph.meta_for_tensor_id(ranges[i].tid);
                p.region = meta_a ? region_kind_name(meta_a->region) : "Unknown";
                p.block_layer_idx = meta_a ? meta_a->block_layer_idx : -1;
                p.tid_a = ranges[i].tid;
                p.name_a = std::string(graph.name_for_tensor_id(ranges[i].tid));
                p.offset_a = ranges[i].offset;
                p.bytes_a = ranges[i].bytes;
                p.tid_b = ranges[j].tid;
                p.name_b = std::string(graph.name_for_tensor_id(ranges[j].tid));
                p.offset_b = ranges[j].offset;
                p.bytes_b = ranges[j].bytes;
                p.overlap_bytes = (end > start) ? (end - start) : 0;
                out.push_back(std::move(p));
            }
        }
    }
}

}  // namespace

std::vector<DebugAliasingPair> collect_static_aliasing(const DslModel& model) {
    std::vector<DebugAliasingPair> out;
    if (const CompiledGraph* fwd = compiled_graph_for(model, /*is_backward=*/false)) {
        scan_static_aliasing(*fwd, DebugGraphKind::Forward, out);
    }
    if (const CompiledGraph* bwd = compiled_graph_for(model, /*is_backward=*/true)) {
        scan_static_aliasing(*bwd, DebugGraphKind::Backward, out);
    }
    return out;
}

DebugTensorResolution resolve_tensor(const DslModel& model, const std::string& name, int tid, bool is_backward) {
    DebugTensorResolution out;
    out.graph = is_backward ? DebugGraphKind::Backward : DebugGraphKind::Forward;
    const CompiledGraph* graph = compiled_graph_for(model, is_backward);
    if (!graph) {
        return out;
    }
    int resolved_tid = tid;
    if (!name.empty()) {
        resolved_tid = graph->find_tensor_id(name);
    }
    if (resolved_tid < 0 || resolved_tid >= graph->num_tensors) {
        return out;
    }
    out.found = true;
    out.entry = make_tensor_entry(*graph, resolved_tid, out.graph);
    out.description = graph->describe_tensor_id(resolved_tid);

    //! first_write_op: earliest op that has this tid in outputs.
    for (std::size_t op_idx = 0; op_idx < graph->ops.size(); ++op_idx) {
        const auto& op = graph->ops[op_idx];
        bool hit = false;
        for (const auto& ref : op.outputs) {
            if (ref.tensor_id == resolved_tid) {
                hit = true;
                break;
            }
        }
        if (hit) {
            out.first_write_op = op_idx;
            break;
        }
    }

    //! last_use_op: CompiledGraph::last_use_index is keyed by name.
    auto it = graph->last_use_index.find(out.entry.name);
    if (it != graph->last_use_index.end()) {
        out.last_use_op = static_cast<std::uint64_t>(it->second);
    }

    //! Phase path: walk the phase tree, follow the deepest child whose op
    //! range contains `first_write_op` (or `last_use_op` if no first_write).
    if (graph->phase_tree.has_value()) {
        std::uint64_t probe = out.first_write_op;
        if (probe == static_cast<std::uint64_t>(-1)) {
            probe = out.last_use_op;
        }
        if (probe != static_cast<std::uint64_t>(-1)) {
            const PhaseNode* node = &graph->phase_tree.value();
            while (node != nullptr) {
                out.phase_path.push_back(node->label.empty() ? std::string(phase_kind_name(node->kind)) : node->label);
                const PhaseNode* next = nullptr;
                for (const auto& child : node->children) {
                    if (probe >= child.op_start && probe < child.op_end) {
                        next = &child;
                        break;
                    }
                }
                node = next;
            }
        }
    }
    return out;
}

}  // namespace dsl
