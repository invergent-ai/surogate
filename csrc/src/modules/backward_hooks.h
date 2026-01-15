// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//

#ifndef SUROGATE_SRC_MODULES_BACKWARD_HOOKS_H
#define SUROGATE_SRC_MODULES_BACKWARD_HOOKS_H

#include <functional>
#include <vector>
#include <unordered_map>
#include <cuda_runtime.h>

namespace modules {

/**
 * @brief Hook points during the backward pass
 *
 * These correspond to specific locations in the transformer block
 * where additional computation can be injected (e.g., LoRA gradient computation).
 */
enum class BackwardHookPoint {
    // Attention sub-block
    BeforeQKVBackward,      ///< Before QKV projection backward
    AfterQKVBackward,       ///< After QKV projection backward, gradients computed
    BeforeAttnOutBackward,  ///< Before attention output projection backward
    AfterAttnOutBackward,   ///< After attention output projection backward

    // MLP sub-block
    BeforeMLPDownBackward,  ///< Before MLP down projection backward
    AfterMLPDownBackward,   ///< After MLP down projection backward
    BeforeMLPUpBackward,    ///< Before MLP up projection backward
    AfterMLPUpBackward,     ///< After MLP up projection backward
    MoEExpertGroupManual,    ///< Manual expert group backward (fused)

    // MoE router
    AfterRouterBackward,    ///< After router backward (for router LoRA gradients)

    // Layer-level
    BeforeLayerBackward,    ///< Before any backward computation for this layer
    AfterLayerBackward,     ///< After all backward computation for this layer
};

/**
 * @brief Convert hook point to string for debugging
 */
constexpr const char* hook_point_name(BackwardHookPoint point) {
    switch (point) {
        case BackwardHookPoint::BeforeQKVBackward: return "BeforeQKVBackward";
        case BackwardHookPoint::AfterQKVBackward: return "AfterQKVBackward";
        case BackwardHookPoint::BeforeAttnOutBackward: return "BeforeAttnOutBackward";
        case BackwardHookPoint::AfterAttnOutBackward: return "AfterAttnOutBackward";
        case BackwardHookPoint::BeforeMLPDownBackward: return "BeforeMLPDownBackward";
        case BackwardHookPoint::AfterMLPDownBackward: return "AfterMLPDownBackward";
        case BackwardHookPoint::BeforeMLPUpBackward: return "BeforeMLPUpBackward";
        case BackwardHookPoint::AfterMLPUpBackward: return "AfterMLPUpBackward";
        case BackwardHookPoint::MoEExpertGroupManual: return "MoEExpertGroupManual";
        case BackwardHookPoint::AfterRouterBackward: return "AfterRouterBackward";
        case BackwardHookPoint::BeforeLayerBackward: return "BeforeLayerBackward";
        case BackwardHookPoint::AfterLayerBackward: return "AfterLayerBackward";
        default: return "Unknown";
    }
}

/**
 * @brief Callback signature for backward hooks
 *
 * @param layer_idx The transformer layer index (0-based)
 * @param point Where in the backward pass this hook is called
 * @param accumulate Whether gradients should be accumulated (vs overwritten)
 * @param stream CUDA stream for the computation
 *
 * Hooks are called synchronously during the backward pass. The hook
 * implementation should enqueue GPU work on the provided stream.
 */
using BackwardHook = std::function<void(
    int layer_idx,
    bool accumulate,
    cudaStream_t stream,
    BackwardHookPoint point,
    void* context
)>;

/**
 * @brief Registry for managing backward hooks
 *
 * Allows registration of multiple hooks at different hook points.
 * Hooks are invoked in registration order.
 *
 * Thread safety: Not thread-safe. Hooks should be registered before
 * training starts and not modified during training.
 *
 * Usage:
 * @code
 * BackwardHookRegistry hooks;
 *
 * // Register LoRA gradient hooks
 * hooks.register_hook(BackwardHookPoint::AfterQKVBackward,
 *     [&](int layer, BackwardHookPoint pt, bool accum, cudaStream_t s) {
 *         compute_lora_qkv_grads(layer, accum, s);
 *     });
 *
 * // During backward pass in the model
 * hooks.invoke(layer_idx, BackwardHookPoint::AfterQKVBackward, accumulate, stream, nullptr);
 * @endcode
 */
class BackwardHookRegistry {
public:
    using HookId = std::size_t;

    BackwardHookRegistry() = default;

    /**
     * @brief Register a hook for a specific hook point
     *
     * @param point The hook point to register at
     * @param hook The callback function
     * @return Unique ID for this hook registration (for later removal)
     */
    HookId register_hook(BackwardHookPoint point, BackwardHook hook) {
        HookId id = mNextId++;
        mHooks[point].push_back({id, std::move(hook)});
        return id;
    }

    /**
     * @brief Register a hook for multiple hook points
     *
     * @param points Vector of hook points
     * @param hook The callback function (will be copied for each point)
     * @return Vector of hook IDs (one per point)
     */
    std::vector<HookId> register_hook(const std::vector<BackwardHookPoint>& points, BackwardHook hook) {
        std::vector<HookId> ids;
        ids.reserve(points.size());
        for (auto point : points) {
            ids.push_back(register_hook(point, hook));
        }
        return ids;
    }

    /**
     * @brief Unregister a hook by ID
     *
     * @param id The hook ID returned from register_hook
     * @return true if hook was found and removed, false otherwise
     */
    bool unregister_hook(HookId id) {
        for (auto& [point, hooks] : mHooks) {
            auto it = std::find_if(hooks.begin(), hooks.end(),
                [id](const HookEntry& e) { return e.id == id; });
            if (it != hooks.end()) {
                hooks.erase(it);
                return true;
            }
        }
        return false;
    }

    /**
     * @brief Invoke all hooks registered at a hook point
     *
     * Hooks are called in registration order.
     *
     * @param layer_idx Current layer index
     * @param point The hook point being invoked
     * @param accumulate Whether to accumulate gradients
     * @param stream CUDA stream for GPU work
     * @param context Optional caller-provided context (passed to hook)
     */
    void invoke(int layer_idx, BackwardHookPoint point, bool accumulate, cudaStream_t stream, void* context = nullptr) const {
        auto it = mHooks.find(point);
        if (it != mHooks.end()) {
            for (const auto& entry : it->second) {
                entry.hook(layer_idx, accumulate, stream, point, context);
            }
        }
    }

    /**
     * @brief Check if any hooks are registered at a point
     */
    bool has_hooks(BackwardHookPoint point) const {
        auto it = mHooks.find(point);
        return it != mHooks.end() && !it->second.empty();
    }

    /**
     * @brief Check if any hooks are registered
     */
    bool has_any_hooks() const {
        for (const auto& [point, hooks] : mHooks) {
            if (!hooks.empty()) return true;
        }
        return false;
    }

    /**
     * @brief Remove all registered hooks
     */
    void clear() {
        mHooks.clear();
    }

    /**
     * @brief Get count of hooks at a point
     */
    std::size_t hook_count(BackwardHookPoint point) const {
        auto it = mHooks.find(point);
        return it != mHooks.end() ? it->second.size() : 0;
    }

private:
    struct HookEntry {
        HookId id;
        BackwardHook hook;
    };

    std::unordered_map<BackwardHookPoint, std::vector<HookEntry>> mHooks;
    HookId mNextId = 0;
};

/**
 * @brief RAII guard for temporarily registering hooks
 *
 * Automatically unregisters hooks when the guard goes out of scope.
 *
 * Usage:
 * @code
 * {
 *     BackwardHookGuard guard(registry);
 *     guard.add(BackwardHookPoint::AfterQKVBackward, my_hook);
 *     // ... hooks are active during this scope
 * } // hooks automatically removed here
 * @endcode
 */
class BackwardHookGuard {
public:
    explicit BackwardHookGuard(BackwardHookRegistry& registry)
        : mRegistry(registry) {}

    ~BackwardHookGuard() {
        for (auto id : mHookIds) {
            mRegistry.unregister_hook(id);
        }
    }

    // Non-copyable
    BackwardHookGuard(const BackwardHookGuard&) = delete;
    BackwardHookGuard& operator=(const BackwardHookGuard&) = delete;

    // Movable
    BackwardHookGuard(BackwardHookGuard&& other) noexcept
        : mRegistry(other.mRegistry), mHookIds(std::move(other.mHookIds)) {
        other.mHookIds.clear();
    }

    BackwardHookGuard& operator=(BackwardHookGuard&&) = delete;

    /**
     * @brief Add a hook that will be removed when guard is destroyed
     */
    void add(BackwardHookPoint point, BackwardHook hook) {
        mHookIds.push_back(mRegistry.register_hook(point, std::move(hook)));
    }

private:
    BackwardHookRegistry& mRegistry;
    std::vector<BackwardHookRegistry::HookId> mHookIds;
};

} // namespace modules

#endif // SUROGATE_SRC_MODULES_BACKWARD_HOOKS_H
