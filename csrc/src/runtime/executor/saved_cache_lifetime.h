#pragma once
#include <algorithm>
#include <string>
#include <vector>
#include <unordered_map>
#include "runtime/dsl/graph_compiler.h"
namespace dsl {
// Conservative layer-level ownership: include every input/output and phase boundary.
// In particular, a Saved ref consumed by a later layer extends its owner's lifetime.
inline std::vector<long long> saved_cache_last_backward_use(const CompiledGraph& graph, int layers) {
    std::vector<long long> last(layers, -1);
    auto mark = [&](int layer, std::size_t i) {
        if (layer >= 0 && layer < layers) last[layer] = std::max(last[layer], static_cast<long long>(i));
    };
    for (int l=0;l<layers;++l)
        if (static_cast<std::size_t>(l)<graph.layer_end_indices.size() && graph.layer_end_indices[l]>0)
            mark(l,graph.layer_end_indices[l]-1);
    for(std::size_t i=0;i<graph.ops.size();++i) {
        const auto& op=graph.ops[i]; mark(op.attrs.layer_idx,i);
        auto ref_use=[&](const TensorRef& ref) {
            mark(ref.layer_idx,i);
            if (auto* meta=graph.meta_for_tensor_id(ref.tensor_id)) mark(meta->block_layer_idx,i);
            std::string name=ref.name;
            for (int j=0;j<2;++j) {
                if(name.rfind("saved.",0)==0) name.erase(0,6);
                if(name.rfind("d_",0)==0) name.erase(0,2);
            }
            if(name.rfind("blocks[",0)==0) {
                auto end=name.find(']',7);
                if(end!=std::string::npos && end>7 && name.find_first_not_of("0123456789",7)>=end)
                    mark(std::stoi(name.substr(7,end-7)),i);
            }
        };
        for(const auto& ref:op.inputs) ref_use(ref);
        for(const auto& ref:op.outputs) ref_use(ref);
    }
    return last;
}
inline bool saved_cache_layer_dead(const std::vector<long long>& last, int layer, std::size_t executed) {
    return layer>=0 && static_cast<std::size_t>(layer)<last.size() && last[layer]>=0 &&
           static_cast<unsigned long long>(last[layer])<=executed;
}
// Exact key contract used by these five production dispatchers, not a substring
// search over cache names. Unknown operations/outputs remain conservatively live.
inline std::unordered_map<std::string,int> backward_dqkv_cache_owners(const CompiledGraph& graph) {
    std::unordered_map<std::string,int> owners;
    for(const auto& op:graph.ops) {
        switch(op.type) {
            case CompiledOpType::FlashAttentionBackward:
            case CompiledOpType::RoPEBackward:
            case CompiledOpType::MRoPEBackward:
            case CompiledOpType::QKVQKNormBackward:
            case CompiledOpType::QKVQKNormRoPEBackward: break;
            default: continue;
        }
        if(op.outputs.empty() || op.op_id.empty()) continue;
        const auto& ref=op.outputs[0]; if(ref.name.empty()) continue;
        int owner=ref.layer_idx;
        if(owner<0) if(auto* meta=graph.meta_for_tensor_id(ref.tensor_id)) owner=meta->block_layer_idx;
        if(owner<0) {
            std::string name=ref.name;
            if(name.rfind("d_",0)==0) name.erase(0,2);
            if(name.rfind("blocks[",0)==0) {
                auto end=name.find(']',7);
                if(end!=std::string::npos && end>7 && name.find_first_not_of("0123456789",7)>=end)
                    owner=std::stoi(name.substr(7,end-7));
            }
        }
        if(owner>=0) {
            const std::string key=op.op_id+"."+ref.name+".d_qkv";
            auto [it,inserted]=owners.emplace(key,owner);
            if(!inserted && it->second!=owner) throw std::runtime_error("Ambiguous backward scratch ownership");
        }
    }
    return owners;
}

}
