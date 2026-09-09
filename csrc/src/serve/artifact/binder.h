#pragma once

#include "artifact/reader.h"

#include <cstddef>
#include <cstdint>
#include <span>
#include <optional>
#include <string_view>
#include <vector>

namespace sinfer::artifact {

enum class TensorPlacement : std::uint8_t {
    Device,
    ValidateOnly,
    /// Pinned, device-mapped host memory: the object's bytes stay in RAM and the kernels read
    /// them over PCIe through a mapped alias. What lets a checkpoint larger than the cards run
    /// at all. The bytes are still the artifact's -- nothing is dequantised or rewritten -- so
    /// a target chooses this per object exactly as it chooses `Device`, and everything
    /// downstream (`materialized_weight`, `materialized_tensor`) is unchanged: the pointer an
    /// object resolves to simply addresses host memory.
    HostBank,
};

struct ObjectHandle {
    std::size_t index = 0;
};

struct DeviceMaterialization {
    ObjectHandle object;
    std::uint64_t offset    = 0;
    std::uint64_t bytes     = 0;
    std::uint64_t alignment = 0;
};

struct HostMaterialization {
    ObjectHandle object;
};

/// An object bound `HostBank`: pinned in host memory after materialization and attached to the
/// artifact by its mapped device pointer.
struct BankMaterialization {
    ObjectHandle object;
};

struct MaterializationPlan {
    std::size_t object_count            = 0;
    std::uint64_t device_capacity_bytes = 0;
    std::vector<DeviceMaterialization> device_objects;
    std::vector<HostMaterialization> host_objects;
    std::vector<BankMaterialization> bank_objects;
};

class Binder {
public:
    explicit Binder(const Reader& reader);

    /// Applies to decoder-layer tensors regardless of which target binds them.
    /// nullopt keeps every layer on the GPU; zero offloads every decoder layer.
    void set_offload(std::optional<std::uint32_t> gpu_layers,
                     std::uint32_t host_moe_layers = 0, std::uint32_t stage_first = 0);
    [[nodiscard]] bool offloads(std::string_view name) const;

    ObjectHandle require_tensor(std::string_view name, NumericFormat format, StorageLayout layout,
                                std::span<const std::uint64_t> shape);
    /// The tensor named, in whatever format the artifact stores it: only the
    /// shape is asserted. Structure is the target's to require; format is the
    /// checkpoint's to declare, and the caller reads it from `descriptor()`.
    ObjectHandle require_tensor_shaped(std::string_view name,
                                       std::span<const std::uint64_t> shape);
    // surogate vendor patch (PATCHES.md #15): presence probe for optional
    // object families (e.g. targets whose artifacts may omit the MTP block).
    [[nodiscard]] bool has(std::string_view name) const noexcept;
    /// The artifact being bound. A target reads its declared `geometry` from here: the
    /// dimensions belong to the checkpoint, and the binder is where the checkpoint and the
    /// target's contract meet.
    [[nodiscard]] const Reader& reader() const noexcept { return reader_; }
    ObjectHandle require_resource(std::string_view name, ResourceEncoding encoding);

    const ObjectDescriptor& descriptor(ObjectHandle handle) const;
    PayloadSpan payload(ObjectHandle handle) const;
    /// The runs an object is assembled from, and the bytes of one of them.
    std::span<const PayloadRun> runs(ObjectHandle handle) const;
    std::span<const std::byte> run_span(const PayloadRun& run) const;
    void materialize_on_device(ObjectHandle handle);
    void retain_on_host(ObjectHandle handle);
    /// Plan this tensor into the pinned host bank instead of device memory. It is validated
    /// like any other object and costs the device nothing.
    void bank_on_host(ObjectHandle handle);
    void validate_only(ObjectHandle handle);
    /// Account for every object this caller did not bind, without placing any of them.
    ///
    /// `finish` insists that a target consume the whole artifact, which is what catches a
    /// target that silently ignores an object a checkpoint ships. A caller that wants one
    /// part of a checkpoint -- the vision tower on its own, with no text model -- has to say
    /// so rather than be caught by that rule, and saying so is this call. Targets never make
    /// it; the invariant they are held to is unchanged.
    void discard_unconsumed();
    MaterializationPlan finish();

private:
    ObjectHandle find_unconsumed(std::string_view name);

    const Reader& reader_;
    std::vector<bool> consumed_;
    std::vector<bool> planned_;
    MaterializationPlan materialization_;
    std::optional<std::uint32_t> gpu_layers_;
    std::vector<std::uint32_t> host_moe_layers_;
};

} // namespace sinfer::artifact
