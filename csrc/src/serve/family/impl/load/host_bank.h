#pragma once

#include "api/types.h"

// Pinned, device-mapped host memory for the objects that never become device resident: a
// mixture's routed experts, and whatever else a target decides not to keep on the card.
// Kernels read them zero-copy over PCIe through the mapped device pointer; a target that also
// wants a device slot cache and host-side expert compute builds those in front of this bank.
//
// This is family machinery, not one target's: a mixture is a mixture, and where its experts'
// bytes live is a question about memory rather than about the model. A target opts in by
// binding those objects with `host_tensor`/`host_linear` instead of the device binders, and by
// handing the kernels `host_ggml_weight`/`host_w8_weight` over the bank's mapped pointers.

#include "artifact/binder.h"
#include "artifact/materializer.h"
#include "artifact/reader.h"
#include "artifact/typed_binding.h"
#include "core/tensor.h"

#include <cstddef>
#include <memory>
#include <cstdint>
#include <initializer_list>
#include <span>
#include <string>
#include <vector>

namespace sinfer::family {

struct HostObjectPlan {
    artifact::ObjectHandle handle;
    /// The artifact mapping, valid while the reader lives. An object read in place from a GGUF
    /// is assembled from several runs -- a fused expert parent is one per source tensor per row
    /// block -- so the bank concatenates `parts` into pinned memory. `payload` is the single
    /// span an object stored in the artifact itself has, and is empty when `parts` is used.
    std::span<const std::byte> payload;
    std::string name;
    std::vector<std::span<const std::byte>> parts;
    // Non-zero: requantise this W8 row-split object to Q4G32AM while it is copied into pinned
    // memory (`q4_rows x q4_k` weights; `q4_w8_scale_offset` locates the source scales plane).
    std::int64_t q4_rows           = 0;
    std::int32_t q4_k              = 0;
    std::size_t q4_w8_scale_offset = 0;
    // Non-zero: the object is a GGUF's own blocks (`decode_type`, `parts` in row order), and the
    // bank decodes them into W8 row-split planes while it copies -- `decode_rows x decode_k`
    // weights, 32 int8 codes and one FP16 scale per group, amax/127, the same requantisation
    // the device gather does. The bank then holds exactly what a converted artifact would have
    // handed it: the gather is a copy, the host expert path reads planes, and the file is
    // still the only copy of the experts on disk.
    std::int64_t decode_rows = 0;
    std::int32_t decode_k    = 0;
    QType decode_type        = QType::W8G32_F16S; // meaningless unless decode_rows != 0
};

struct HostBankPlan {
    std::vector<HostObjectPlan> objects;
    /// Reported against while the bank is filled. Building it reads every routed expert out of
    /// the mapped checkpoint and into pinned memory -- tens of seconds for a model of this size
    /// -- and without this the engine says nothing for all of it.
    LoadProgress progress;
    [[nodiscard]] std::size_t total_bytes() const noexcept;
};

/// One pinned allocation per host object; `device_pointer` is the mapped alias.
struct HostObject {
    void* host             = nullptr;
    const void* device     = nullptr;
    std::size_t bytes      = 0;
    /// True when `host` is an mmap'd region pinned with cudaHostRegister (the fast path:
    /// threaded first-touch then register, ~10 GB/s against cudaHostAlloc's 1.8); false when it
    /// came from the cudaHostAlloc fallback. Decides the release path.
    bool registered        = false;
    std::string name;
};

class HostBank {
public:
    HostBank() = default;
    explicit HostBank(const HostBankPlan& plan);
    ~HostBank();

    HostBank(const HostBank&)            = delete;
    HostBank& operator=(const HostBank&) = delete;
    HostBank(HostBank&&)                 = delete;
    HostBank& operator=(HostBank&&)      = delete;

    [[nodiscard]] const HostObject& object(artifact::ObjectHandle handle) const;
    [[nodiscard]] std::size_t total_bytes() const noexcept { return total_bytes_; }

    /// The process-wide bank for this plan: pipeline stages of one model in one process share
    /// the pinned experts instead of pinning them once per device (keyed by the objects' names
    /// and sizes, so the same artifact loaded for another stage reuses the live bank).
    [[nodiscard]] static std::shared_ptr<HostBank> shared(const HostBankPlan& plan);

    /// Point every banked object at its pinned bytes. After this the artifact resolves those
    /// objects to a mapped host pointer and every `materialized_weight`/`materialized_tensor`
    /// in the engine reads them without knowing where they live.
    void attach(artifact::MaterializedArtifact& backing) const;

private:
    std::vector<std::pair<std::size_t, HostObject>> objects_;
    std::size_t total_bytes_ = 0;
};

// -------------------------------------------------------------------------------------------
// Binding an object into the bank instead of onto the device
// -------------------------------------------------------------------------------------------

/// Everything a binder placed with `artifact::TensorPlacement::HostBank`, ready to pin. A
/// target says only *which* objects go to the host -- at the same site where it already says
/// Device or ValidateOnly -- and this collects them. Call after `Binder::finish()`.
[[nodiscard]] HostBankPlan collect_host_bank(artifact::Binder& binder,
                                             const artifact::MaterializationPlan& plan,
                                             LoadProgress progress = {});

/// How the bank will find an object's bytes. One run is a span; several are the stretches of a
/// GGUF a fused parent is assembled from, and the bank concatenates them into pinned memory.
[[nodiscard]] HostObjectPlan host_plan(artifact::Binder& binder, artifact::ObjectHandle handle,
                                       const std::string& name);

/// Validate the object against the artifact and record its mapping for the bank. The object is
/// never uploaded: `artifact::TensorPlacement::ValidateOnly` is what keeps its bytes off the device.
artifact::ObjectHandle host_tensor(artifact::Binder& binder, HostBankPlan& bank,
                                   const std::string& name, artifact::NumericFormat format,
                                   std::initializer_list<std::uint64_t> shape);

/// As `host_tensor`, but the format is read from the artifact rather than asserted -- a
/// GGUF-native artifact stores a mixture as the file's own blocks and never rewrites them.
artifact::LinearBinding host_linear(artifact::Binder& binder, HostBankPlan& bank,
                                    const std::string& name, std::int32_t rows,
                                    std::int32_t columns);

// -------------------------------------------------------------------------------------------
// Reading the bank from a kernel
// -------------------------------------------------------------------------------------------

/// A weight held as the file's own GGML blocks, addressed through the bank's mapped device
/// pointer. A block carries its own scale, so there is no separate plane and no padding.
[[nodiscard]] Weight host_ggml_weight(const HostObject& object, artifact::NumericFormat format,
                                      std::int32_t rows, std::int32_t columns);

/// A row-split W8 weight whose planes live in the bank, in the layout the device materializer
/// would have produced.
[[nodiscard]] Weight host_w8_weight(const HostObject& object, std::int32_t rows,
                                    std::int32_t columns);

} // namespace sinfer::family
