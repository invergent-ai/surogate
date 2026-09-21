#pragma once
#include <algorithm>
#include <unordered_set>
#include "runtime/dsl/graph_compiler.h"
#include "runtime/executor/graph_executor_utils.h"
namespace dsl {
// Retain declared foreign-layer dependencies as well as local producers.
// The slot owner alone is insufficient for shared-KV/custom cross-block ops.
inline std::unordered_set<std::string> replay_backward_saved_dependencies(const CompiledGraph* graph,int layer) {
 std::unordered_set<std::string> required;if(!graph)return required;
 for(std::size_t i=0;i<graph->ops.size();++i){
  const auto&op=graph->ops[i];int owner=-1;
  auto scan=[&](const TensorRef&ref){if(!ref.is_gradient){int n=ref.layer_idx;std::string field;if(n<0)parse_block_param(ref.name,n,field);owner=std::max(owner,n);}};
  for(const auto&ref:op.inputs)scan(ref);for(const auto&ref:op.outputs)scan(ref);
  const bool in_range=layer>=0 && std::size_t(layer)<graph->layer_start_indices.size() && std::size_t(layer)<graph->layer_end_indices.size() && i>=graph->layer_start_indices[layer] && i<graph->layer_end_indices[layer];
  if(owner==layer||in_range||op.attrs.layer_idx==layer)
   for(const auto&ref:op.inputs)if(ref.slot==TensorSlot::Saved&&!ref.name.empty())required.insert(ref.name);
 }
 return required;
}
inline bool replay_saved_copy_required(int owner,int replay_layer,bool produced_here,bool backward_dependency,bool complete_layer_metadata) {
 return !complete_layer_metadata || owner<0 || owner==replay_layer || produced_here || backward_dependency;
}
}
