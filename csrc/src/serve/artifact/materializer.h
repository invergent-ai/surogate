#pragma once

#include "artifact/binder.h"
#include "core/tensor.h"
#include "core/arena.h"
#include "core/device.h"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <span>
#include <string_view>
#include <vector>

namespace sinfer::artifact {

struct LoadProgress {
    std::function<void(std::string_view, std::uint64_t, std::uint64_t)> callback;
};

/// How much device memory the load may hold at once for objects it rearranges rather than
/// copies. Their source bytes land in scratch and a kernel writes the real allocation, so the
/// scratch is transient -- but anything created before the load (the expert slot pool) has to
/// leave room for it, and on a 200 GB checkpoint the sum of every such object is ~9 GiB of pool
/// nobody gets. The load therefore works in waves of at most this much, plus whatever single
/// object exceeds it, and `projected_load_staging_bytes` projects the same figure.
inline constexpr std::uint64_t kLoadStagingCapBytes = 1ULL << 30;

struct MaterializationStats {
    std::uint64_t file_bytes              = 0;
    std::uint64_t h2d_bytes               = 0;
    std::uint64_t device_capacity_bytes   = 0;
    std::uint64_t retained_resource_bytes = 0;
    std::uint64_t peak_staging_bytes      = 0;
    std::size_t tensor_count              = 0;
    std::size_t resource_count            = 0;
    double upload_seconds                 = 0.0;
};

class MaterializedArtifact {
public:
    MaterializedArtifact()                                           = default;
    ~MaterializedArtifact()                                          = default;
    MaterializedArtifact(MaterializedArtifact&&) noexcept            = default;
    MaterializedArtifact& operator=(MaterializedArtifact&&) noexcept = default;
    MaterializedArtifact(const MaterializedArtifact&)                = delete;
    MaterializedArtifact& operator=(const MaterializedArtifact&)     = delete;

    void* device_data(ObjectHandle handle) const;
    /// Give a `HostBank`-placed object the pointer its kernels will read: the device-mapped
    /// alias of its pinned host bytes. Everything downstream is unchanged -- an object's
    /// pointer is an object's pointer -- which is what makes host residency a placement rather
    /// than a second way to build every weight in the engine.
    void attach_host_object(ObjectHandle handle, const void* device_pointer);
    /// The typed row runs of a ggml-blocks tensor whose rows are not all one format, pointing
    /// into its device bytes; empty for a homogeneous object.
    std::span<const WeightSegment> segments(ObjectHandle handle) const;
    /// The device-resident column group map of a ggml-blocks tensor read in the file's column
    /// order; empty when its columns are in order.
    std::span<const std::int32_t> input_group_map(ObjectHandle handle) const;
    std::span<const std::byte> resource_bytes(ObjectHandle handle) const;
    std::vector<std::byte> take_resource_bytes(ObjectHandle handle);

    const MaterializationStats& stats() const noexcept { return stats_; }

    DeviceArena& device_arena();

private:
    friend MaterializedArtifact materialize(const Reader&, const MaterializationPlan&,
                                            DeviceContext&, LoadProgress*);

    struct ObjectStorage {
        void* device = nullptr;
        std::vector<std::byte> resource;
        std::vector<WeightSegment> segments;
        const std::int32_t* input_group_map = nullptr;
        std::size_t input_groups            = 0;
    };

    std::unique_ptr<DeviceArena> device_arena_;
    std::unique_ptr<DeviceArena> input_maps_; // the column group maps the ops read at every launch
    std::vector<ObjectStorage> objects_;
    MaterializationStats stats_;
};

MaterializedArtifact materialize(const Reader& reader, const MaterializationPlan& plan,
                                 DeviceContext& device, LoadProgress* progress = nullptr);

} // namespace sinfer::artifact
