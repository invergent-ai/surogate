// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0

#include "runtime/dsl/fusions/fusion_pass.h"

namespace dsl {

FusionPassRegistry& FusionPassRegistry::instance() {
    static FusionPassRegistry reg;
    return reg;
}

void FusionPassRegistry::register_pass(FusionPassInfo info) {
    auto id = info.id;
    id_to_index_[id] = passes_.size();
    passes_.push_back(std::move(info));
}

const FusionPassInfo* FusionPassRegistry::find(const std::string& id) const {
    auto it = id_to_index_.find(id);
    return (it != id_to_index_.end()) ? &passes_[it->second] : nullptr;
}

} // namespace dsl
