#pragma once

// The pimpl a target package needs and no target chooses.
//
// Every target hands the engine a LoadPlan (what to materialize) and a
// LoadedModel (what was materialized), both opaque so the engine's translation
// unit never sees a target's weight structs. The pattern is forced by that
// contract rather than by anything about a model, and it came out identical in
// all nine targets: the same two class declarations, the same Impl definitions,
// the same constructor that moves one into the other. 77 lines apiece, and the
// only thing that varied was the namespace they sat in.
//
// So the pimpl is written once, here. The target-specific types the macros name
// -- WeightsProfile, ArtifactLoadPlan, BindingPlan, LoadedModelData -- resolve
// in the target's own namespace, which is what let the nine copies be identical
// in the first place.
//
// A macro rather than a template or a CRTP base: these are member definitions of
// distinct classes in distinct namespaces, which no template can generate, and a
// base class would put a vtable or an extra indirection into the load path to
// save text the compiler never reads twice anyway.

#include <memory>

// clang-format off

/// Declares LoadPlan and LoadedModel. Expand inside the target's `detail`
/// namespace, in its api package header. OWNER is the package that may reach
/// through them, namespace-qualified from `detail`'s parent -- `llama::Package`.
///
/// The move rules differ between the two on purpose: a LoadPlan is moved into
/// `construct_loaded_model` and consumed there, while a LoadedModel owns device
/// memory the engine holds by reference for its lifetime and must not move.
#define SINFER_TARGET_LOAD_TYPES(OWNER) \
    class LoadPlan { \
    public: \
        LoadPlan(LoadPlan&&) noexcept; \
        LoadPlan& operator=(LoadPlan&&) noexcept; \
        ~LoadPlan(); \
 \
        LoadPlan(const LoadPlan&)            = delete; \
        LoadPlan& operator=(const LoadPlan&) = delete; \
 \
        [[nodiscard]] const artifact::MaterializationPlan& materialization() const; \
 \
    private: \
        class Impl; \
        explicit LoadPlan(std::unique_ptr<Impl> impl) noexcept; \
        std::unique_ptr<Impl> impl_; \
 \
        friend struct OWNER; \
    }; \
 \
    class LoadedModel { \
    public: \
        ~LoadedModel(); \
 \
        LoadedModel(const LoadedModel&)            = delete; \
        LoadedModel& operator=(const LoadedModel&) = delete; \
        LoadedModel(LoadedModel&&)                 = delete; \
        LoadedModel& operator=(LoadedModel&&)      = delete; \
 \
    private: \
        class Impl; \
        explicit LoadedModel(std::unique_ptr<Impl> impl) noexcept; \
        std::unique_ptr<Impl> impl_; \
 \
        friend struct OWNER; \
    }

/// Defines LoadedModel::Impl. Expand inside the target's `detail` namespace in
/// its bindings header, after BindingPlan and LoadedModelData are complete --
/// LoadedModelData is held by value, so the engine cannot see this and the
/// target's weight structs stay out of its translation unit.
#define SINFER_TARGET_LOADED_MODEL_IMPL() \
    class LoadedModel::Impl { \
    public: \
        Impl(WeightsProfile weights_profile_in, BindingPlan plan, \
             artifact::MaterializedArtifact materialized) \
            : weights_profile(weights_profile_in), data(std::move(plan), std::move(materialized)) {} \
 \
        WeightsProfile weights_profile; \
        LoadedModelData data; \
    }

/// Defines LoadPlan::Impl and the out-of-line special members both classes
/// declared. Expand inside the target's `detail` namespace in its package.cpp.
/// Needs <stdexcept> and <utility> in the including file.
#define SINFER_TARGET_LOAD_PIMPL() \
    class LoadPlan::Impl { \
    public: \
        Impl(WeightsProfile weights_profile_in, ArtifactLoadPlan target_plan) \
            : weights_profile(weights_profile_in), plan(std::move(target_plan)) {} \
 \
        WeightsProfile weights_profile; \
        ArtifactLoadPlan plan; \
    }; \
 \
    LoadPlan::LoadPlan(std::unique_ptr<Impl> impl) noexcept : impl_(std::move(impl)) {} \
 \
    LoadPlan::LoadPlan(LoadPlan&&) noexcept            = default; \
    LoadPlan& LoadPlan::operator=(LoadPlan&&) noexcept = default; \
    LoadPlan::~LoadPlan()                              = default; \
 \
    const artifact::MaterializationPlan& LoadPlan::materialization() const { \
        if (impl_ == nullptr) { throw std::logic_error("target load plan is empty"); } \
        return impl_->plan.materialization; \
    } \
 \
    LoadedModel::LoadedModel(std::unique_ptr<Impl> impl) noexcept : impl_(std::move(impl)) {} \
 \
    LoadedModel::~LoadedModel() = default

/// Defines Package::construct_loaded_model: moves the plan's bindings and the
/// materialized artifact into a LoadedModel, and empties the plan so a second
/// call cannot hand out a second owner of the same weights. Expand inside the
/// target's own namespace (not `detail`), in its package.cpp.
#define SINFER_TARGET_CONSTRUCT_LOADED_MODEL() \
    std::unique_ptr<Package::LoadedModel> Package::construct_loaded_model( \
        LoadPlan&& plan, artifact::MaterializedArtifact&& materialized) { \
        if (plan.impl_ == nullptr) { throw std::invalid_argument("target load plan is empty"); } \
        auto impl = std::make_unique<LoadedModel::Impl>(plan.impl_->weights_profile, \
                                                        std::move(plan.impl_->plan.bindings), \
                                                        std::move(materialized)); \
        plan.impl_.reset(); \
        return std::unique_ptr<LoadedModel>(new LoadedModel(std::move(impl))); \
    } \
    static_assert(true, "consume the trailing semicolon")

// clang-format on
