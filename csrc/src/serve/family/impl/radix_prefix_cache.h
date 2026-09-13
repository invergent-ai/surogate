// Copyright 2023-2024 SGLang Team
// Copyright 2026 Invergent SA
// SPDX-License-Identifier: Apache-2.0
//
// C++ adaptation of SGLang's radix_cache.py insertion/splitting and
// mamba_radix_cache.py matching at nodes with complete state snapshots.
// https://github.com/sgl-project/sglang/tree/ddc1df1203036c97ef2398ef4e2d012301c4c17f/python/sglang/srt/mem_cache
// Storage is supplied by the caller. Immutable snapshots may be pinned by request
// plans; LRU eviction only removes unpinned values. Matching never invents a
// recurrent-state checkpoint when splitting a shared token prefix.
#pragma once

#include <algorithm>
#include <cstdint>
#include <memory>
#include <span>
#include <unordered_map>
#include <vector>

namespace sinfer::family::detail {

template<class Value> class RadixPrefixCache {
public:
    using Pointer = std::shared_ptr<const Value>;
    explicit RadixPrefixCache(std::size_t capacity, std::size_t max_entries = 64)
        : capacity_(capacity), max_entries_(max_entries) {}

    bool reserve(std::size_t bytes) {
        if (!bytes || bytes > capacity_) { return false; }
        while (used_ > capacity_ - bytes || values_.size() >= max_entries_) {
            auto oldest = values_.end();
            for (auto it = values_.begin(); it != values_.end(); ++it) {
                if (it->second.value.use_count() == 1 &&
                    (oldest == values_.end() || it->second.access < oldest->second.access)) {
                    oldest = it;
                }
            }
            if (oldest == values_.end()) { return false; }
            used_ -= oldest->second.bytes;
            values_.erase(oldest);
            ++revision_;
            prune(root_);
        }
        return true;
    }

    bool insert(std::span<const std::int32_t> tokens, std::span<const std::uint32_t> boundaries,
                Pointer value, std::size_t bytes) {
        if (!value || !bytes || bytes > capacity_ || boundaries.empty()) { return false; }
        for (auto end : boundaries) {
            if (end == 0 || end > tokens.size()) { return false; }
        }
        if (!reserve(bytes)) { return false; }
        const auto id = ++clock_;
        values_.emplace(id, Entry{std::move(value), bytes, id});
        used_ += bytes;
        try {
            for (auto end : boundaries) { insert_node(tokens.first(end), id); }
        } catch (...) {
            // A failed edge split must not leave an empty or unreachable node.
            // Request plans keep any already leased snapshots alive.
            clear();
            throw;
        }
        ++revision_;
        return true;
    }

    // The caller verifies non-token identity (adapter, positions, and media) and
    // returns the usable frontier. A deeper token match alone is insufficient.
    template<class Match> Pointer match(std::span<const std::int32_t> key, Match&& validate,
                                        bool partial = false) {
        std::uint32_t best = 0;
        std::uint64_t selected = 0, newest = 0;
        const auto visit = [&](const auto& self, const Node& node, std::size_t matched,
                               bool diverged) -> void {
            if (diverged && matched < best) { return; }
            for (auto id : node.values) {
                auto found = values_.find(id);
                if (found == values_.end()) { continue; }
                const auto frontier = validate(*found->second.value);
                if (frontier > matched) { continue; }
                if (frontier > best || (frontier && frontier == best && found->second.access > newest)) {
                    selected = id; best = frontier; newest = found->second.access;
                }
            }
            for (const auto& [unused, child] : node.children) {
                const auto shared = diverged ? 0 : common_prefix(child->key, key.subspan(matched));
                const bool stops_here = diverged || shared != child->key.size();
                if (!stops_here || (partial && matched + shared > 0)) {
                    self(self, *child, matched + shared, stops_here);
                }
            }
        };
        // Hybrid models require a complete stored state boundary. Pure attention
        // may also use KV from a longer image when a query ends or branches inside
        // a compressed edge, as in SGLang's ordinary radix cache.
        visit(visit, root_, 0, false);
        if (!selected) { return {}; }
        auto& entry = values_.at(selected);
        entry.access = ++clock_;
        return entry.value;
    }

    void clear() { values_.clear(); root_ = {}; used_ = 0; ++revision_; }
    [[nodiscard]] bool empty() const noexcept { return values_.empty(); }
    [[nodiscard]] std::size_t bytes() const noexcept { return used_; }
    [[nodiscard]] std::uint64_t revision() const noexcept { return revision_; }

private:
    struct Node {
        std::vector<std::int32_t> key;
        std::unordered_map<std::int32_t, std::unique_ptr<Node>> children;
        std::vector<std::uint64_t> values;
    };
    struct Entry { Pointer value; std::size_t bytes; std::uint64_t access; };
    static std::size_t common_prefix(std::span<const std::int32_t> a,
                                     std::span<const std::int32_t> b) {
        std::size_t count = 0;
        while (count < std::min(a.size(), b.size()) && a[count] == b[count]) { ++count; }
        return count;
    }
    void insert_node(std::span<const std::int32_t> key, std::uint64_t id) {
        Node* node = &root_;
        while (!key.empty()) {
            auto& child = node->children[key.front()];
            if (!child) {
                child = std::make_unique<Node>();
                child->key.assign(key.begin(), key.end());
                node = child.get();
                break;
            }
            const auto prefix = common_prefix(child->key, key);
            if (prefix < child->key.size()) {
                auto split = std::make_unique<Node>();
                split->key.assign(child->key.begin(), child->key.begin() + prefix);
                child->key.erase(child->key.begin(), child->key.begin() + prefix);
                const auto suffix = child->key.front();
                split->children.emplace(suffix, std::move(child));
                child = std::move(split);
            }
            node = child.get();
            key = key.subspan(prefix);
        }
        node->values.push_back(id);
    }
    void prune(Node& node) {
        std::erase_if(node.values, [&](auto id) { return !values_.contains(id); });
        for (auto it = node.children.begin(); it != node.children.end();) {
            prune(*it->second);
            if (it->second->values.empty() && it->second->children.empty()) {
                it = node.children.erase(it);
            } else {
                auto& child = it->second;
                if (child->values.empty() && child->children.size() == 1) {
                    auto& descendant = child->children.begin()->second;
                    // Recompress an edge after eviction so repeated continuations
                    // cannot leave an unbounded chain of empty ancestor nodes.
                    child->key.insert(child->key.end(), descendant->key.begin(), descendant->key.end());
                    auto replacement = std::move(descendant);
                    replacement->key = std::move(child->key);
                    child = std::move(replacement);
                }
                ++it;
            }
        }
    }
    Node root_;
    std::unordered_map<std::uint64_t, Entry> values_;
    std::size_t capacity_, max_entries_, used_ = 0;
    std::uint64_t clock_ = 0, revision_ = 0;
};

} // namespace sinfer::family::detail
