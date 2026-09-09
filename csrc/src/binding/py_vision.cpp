// The serving engine's vision tower, callable from Python.
//
// The trainer needs image features and used to get them by running a second copy of this
// encoder out of transformers. Both read the same checkpoint, so they were two answers to
// one question, and only one of them is the one serving gives. This exposes the tower the
// engine runs, bound from the same artifact objects.
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>

#include "core/arena.h"
#include "family/vision_standalone.h"

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

namespace nb = nanobind;
namespace {

using sinfer::family::PromptModality;
using sinfer::family::StandaloneVisionTower;
using sinfer::family::VisionGrid;

PromptModality modality_from(std::string_view name) {
    if (name == "image") { return PromptModality::Image; }
    if (name == "video") { return PromptModality::Video; }
    throw std::invalid_argument("vision modality must be \"image\" or \"video\"");
}

class VisionEncoder {
public:
    VisionEncoder(const std::string& artifact, int device)
        : device_(device), tower_(artifact, device) {}

    [[nodiscard]] nb::dict geometry() const {
        const auto& g = tower_.geometry();
        nb::dict out;
        out["layers"]             = g.layers;
        out["hidden"]             = g.hidden;
        out["output_hidden"]      = g.output_hidden;
        out["heads"]              = g.heads;
        out["patch_dim"]          = g.patch_dim;
        out["merge"]              = g.merge;
        out["deepstack_layers"]   = g.deepstack_layers;
        out["intermediate"]       = g.intermediate;
        return out;
    }

    /// `patches` is BF16, `patch_count * patch_dim`, patch-major -- the image processor's
    /// output for one item, viewed as uint16. Returns a CUDA tensor
    /// [1 + deepstack_layers, merged_tokens, output_hidden] whose first plane is the
    /// projection and whose remaining planes are the deepstack mergers, in tower order.
    nb::ndarray<> encode(nb::ndarray<> patches, std::int32_t temporal, std::int32_t height,
                         std::int32_t width, const std::string& modality) {
        if (patches.device_type() != nb::device::cpu::value) {
            throw std::invalid_argument("vision patches must be a host array");
        }
        if (patches.dtype().bits != 16) {
            throw std::invalid_argument("vision patches must be a 16-bit (BF16) array");
        }
        const VisionGrid grid{temporal, height, width};
        const auto& g      = tower_.geometry();
        const auto merged  = tower_.merged_tokens(grid);
        const auto planes  = static_cast<std::size_t>(1 + g.deepstack_layers);
        const auto bytes   = tower_.output_bytes(grid);

        auto storage = std::make_shared<sinfer::DeviceBuffer>(bytes);
        sinfer::Tensor output(storage->p, sinfer::DType::BF16,
                              {g.output_hidden, static_cast<std::int32_t>(merged),
                               static_cast<std::int32_t>(planes)});
        const std::span<const std::uint16_t> values(
            static_cast<const std::uint16_t*>(patches.data()), patches.size());
        tower_.encode(values, grid, modality_from(modality), output);

        // The buffer outlives this call for exactly as long as the array does: the capsule
        // holds the only remaining reference and frees it when Python drops the view.
        auto* keep = new std::shared_ptr<sinfer::DeviceBuffer>(std::move(storage));
        nb::capsule owner(keep, [](void* p) noexcept {
            delete static_cast<std::shared_ptr<sinfer::DeviceBuffer>*>(p);
        });
        const std::size_t shape[3] = {planes, merged, static_cast<std::size_t>(g.output_hidden)};
        return nb::ndarray<>(output.data, 3, shape, owner, nullptr,
                             nb::dlpack::dtype{static_cast<std::uint8_t>(nb::dlpack::dtype_code::Bfloat), 16, 1},
                             nb::device::cuda::value, device_);
    }

private:
    int device_;
    StandaloneVisionTower tower_;
};

} // namespace

void bind_vision_encoder(nb::module_& m) {
    nb::class_<VisionEncoder>(m, "VisionEncoder")
        .def(nb::init<const std::string&, int>(), nb::arg("artifact"), nb::arg("device") = 0,
             "Load a `.sinfer` artifact's vision tower onto a CUDA device.\n\n"
             "Only the tower's objects are bound and materialized; the text model is not read.")
        .def_prop_ro("geometry", &VisionEncoder::geometry,
                     "The tower's dimensions, as the artifact declares them.")
        .def("encode", &VisionEncoder::encode, nb::arg("patches"), nb::arg("temporal"),
             nb::arg("height"), nb::arg("width"), nb::arg("modality") = "image",
             nb::call_guard<nb::gil_scoped_release>(),
             "Encode one image or video item into merged visual tokens.\n\n"
             "Parameters:\n"
             "- patches: host BF16 array, patch_count * patch_dim, patch-major.\n"
             "- temporal, height, width: the item's patch grid; height and width must be\n"
             "  multiples of the tower's merge factor.\n"
             "- modality: \"image\" or \"video\".\n\n"
             "Returns a CUDA BF16 array [1 + deepstack_layers, merged_tokens, output_hidden];\n"
             "plane 0 is the projection, the rest are the deepstack mergers in tower order.");
}
