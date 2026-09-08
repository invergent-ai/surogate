#include "artifact/reader.h"
#include <api/family/text_geometry.h>
#include <api/family/vision_geometry.h>

#include <map>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <cerrno>
#include <cstring>
#include <functional>
#include <limits>
#include <span>
#include <string_view>
#include <filesystem>
#include <system_error>
#include <type_traits>
#include <unordered_map>
#include <utility>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace sinfer::artifact {
namespace {

using Json = nlohmann::json;

// The format identifier. Spelled a byte at a time, so a text sweep over the tree
// does not reach it -- when the project was renamed this was the one constant a
// rename script could not touch, and missing it would have left the Python writer
// emitting SINFER while this reader demanded NINFER.
constexpr std::array<std::byte, 8> kMagic = {
    std::byte{'S'}, std::byte{'I'}, std::byte{'N'}, std::byte{'F'},
    std::byte{'E'}, std::byte{'R'}, std::byte{0},   std::byte{2},
};

// What artifacts written before the rename carry. The layout is identical -- only
// the first byte differs -- so these are recognised in order to say so, rather
// than reported as "not an artifact".
constexpr std::array<std::byte, 8> kPreRenameMagic = {
    std::byte{'N'}, std::byte{'I'}, std::byte{'N'}, std::byte{'F'},
    std::byte{'E'}, std::byte{'R'}, std::byte{0},   std::byte{2},
};
constexpr std::array<std::byte, 8> kV1Magic = {
    std::byte{'N'}, std::byte{'I'}, std::byte{'N'}, std::byte{'F'},
    std::byte{'E'}, std::byte{'R'}, std::byte{0},   std::byte{1},
};
constexpr std::uint64_t kPrefixBytes      = 16;
constexpr std::uint64_t kPayloadAlignment = 4096;

std::uint64_t checked_add(std::uint64_t a, std::uint64_t b, std::string_view label) {
    if (b > std::numeric_limits<std::uint64_t>::max() - a) {
        throw ArtifactError(std::string(label) + " overflows u64");
    }
    return a + b;
}

std::uint64_t align_up(std::uint64_t value, std::uint64_t alignment, std::string_view label) {
    const auto biased = checked_add(value, alignment - 1, label);
    return biased / alignment * alignment;
}

std::uint64_t read_u64_le(const std::byte* data) noexcept {
    std::uint64_t value = 0;
    for (unsigned i = 0; i < 8; ++i) {
        value |= std::uint64_t(std::to_integer<unsigned char>(data[i])) << (i * 8);
    }
    return value;
}

template <std::size_t N>
void require_members(const Json& value, const std::array<const char*, N>& members,
                     std::string_view label, std::size_t optional = 0) {
    if (!value.is_object() || value.size() != N + optional) {
        throw ArtifactError(std::string(label) + " has missing or extra members");
    }
    for (const char* member : members) {
        if (!value.contains(member)) {
            throw ArtifactError(std::string(label) + " has missing or extra members");
        }
    }
}

const std::string& require_string(const Json& value, std::string_view label) {
    if (!value.is_string()) {
        throw ArtifactError(std::string(label) + " must be a nonempty string");
    }
    const auto& result = value.get_ref<const std::string&>();
    if (result.empty()) { throw ArtifactError(std::string(label) + " must be a nonempty string"); }
    return result;
}

std::uint64_t require_unsigned(const Json& value, std::string_view label, bool positive) {
    if (!value.is_number_unsigned()) {
        throw ArtifactError(std::string(label) + " must be an integer");
    }
    const auto result = value.get<std::uint64_t>();
    if (positive && result == 0) { throw ArtifactError(std::string(label) + " must be positive"); }
    return result;
}

NumericFormat parse_format(std::string_view name) {
    if (name == "BF16") { return NumericFormat::BF16; }
    if (name == "FP32") { return NumericFormat::FP32; }
    if (name == "I32") { return NumericFormat::I32; }
    if (name == "Q4G64_F16S") { return NumericFormat::Q4G64_F16S; }
    if (name == "Q5G64_F16S") { return NumericFormat::Q5G64_F16S; }
    if (name == "Q6G64_F16S") { return NumericFormat::Q6G64_F16S; }
    if (name == "W8G32_F16S") { return NumericFormat::W8G32_F16S; }
    if (name == "NVFP4") { return NumericFormat::NVFP4; }
    if (name == "FP8_E4M3FN_ROW_BF16S") { return NumericFormat::FP8_E4M3FN_ROW_BF16S; }
    if (name == "FP8_E4M3FN_BLK128_F32S") { return NumericFormat::FP8_E4M3FN_BLK128_F32S; }
    if (name == "FP8_E4M3FN_ROW_F32S") { return NumericFormat::FP8_E4M3FN_ROW_F32S; }
    if (name == "Q2_K") { return NumericFormat::Q2_K; }
    if (name == "Q3_K") { return NumericFormat::Q3_K; }
    if (name == "Q4_K") { return NumericFormat::Q4_K; }
    if (name == "Q5_K") { return NumericFormat::Q5_K; }
    if (name == "Q6_K") { return NumericFormat::Q6_K; }
    if (name == "Q8_0") { return NumericFormat::Q8_0; }
    if (name == "Q4_1") { return NumericFormat::Q4_1; }
    if (name == "Q5_1") { return NumericFormat::Q5_1; }
    if (name == "IQ4_NL") { return NumericFormat::IQ4_NL; }
    if (name == "Q4_0") { return NumericFormat::Q4_0; }
    if (name == "Q5_0") { return NumericFormat::Q5_0; }
    if (name == "IQ2_XXS") { return NumericFormat::IQ2_XXS; }
    if (name == "IQ2_XS") { return NumericFormat::IQ2_XS; }
    if (name == "IQ2_S") { return NumericFormat::IQ2_S; }
    if (name == "IQ3_XXS") { return NumericFormat::IQ3_XXS; }
    if (name == "IQ3_S") { return NumericFormat::IQ3_S; }
    if (name == "IQ1_S") { return NumericFormat::IQ1_S; }
    if (name == "IQ1_M") { return NumericFormat::IQ1_M; }
    if (name == "IQ4_XS") { return NumericFormat::IQ4_XS; }
    if (name == "TQ1_0") { return NumericFormat::TQ1_0; }
    if (name == "TQ2_0") { return NumericFormat::TQ2_0; }
    if (name == "MXFP4") { return NumericFormat::MXFP4; }
    if (name == "NVFP4_GGML") { return NumericFormat::NVFP4_GGML; }
    if (name == "Q1_0") { return NumericFormat::Q1_0; }
    if (name == "Q2_0") { return NumericFormat::Q2_0; }
    if (name == "F16") { return NumericFormat::F16; }
    throw ArtifactError("unknown tensor format: " + std::string(name));
}

StorageLayout parse_layout(std::string_view name) {
    if (name == "contiguous-le-v1") { return StorageLayout::ContiguousLeV1; }
    if (name == "row-split-k128-v1") { return StorageLayout::RowSplitK128V1; }
    if (name == "blockscale-k16-m128x4-v1") { return StorageLayout::BlockScaleK16M128x4V1; }
    if (name == "row-scale-v1") { return StorageLayout::RowScaleV1; }
    if (name == "ggml-blocks-v1") { return StorageLayout::GgmlBlocksV1; }
    if (name == "block-scale-128-fp8-v1") { return StorageLayout::BlockScale128Fp8V1; }
    if (name == "row-scale-f32-v1") { return StorageLayout::RowScaleF32V1; }
    throw ArtifactError("unknown tensor layout: " + std::string(name));
}

ResourceEncoding parse_encoding(std::string_view name) {
    if (name == "raw-bytes-v1") { return ResourceEncoding::RawBytesV1; }
    throw ArtifactError("unknown resource encoding: " + std::string(name));
}

TensorDescriptor parse_tensor(const Json& value) {
    static constexpr std::array members = {
        "name", "kind", "shape", "format", "layout", "offset", "bytes",
    };
    require_members(value, members, "tensor entry",
                    (value.contains("runs") ? 1 : 0) + (value.contains("transform") ? 1 : 0) +
                        (value.contains("group_map") ? 1 : 0) +
                        (value.contains("segments") ? 1 : 0));

    const auto name        = require_string(value.at("name"), "tensor name");
    const auto format      = parse_format(require_string(value.at("format"), "tensor format"));
    const auto layout      = parse_layout(require_string(value.at("layout"), "tensor layout"));
    const auto offset      = require_unsigned(value.at("offset"), "tensor offset", false);
    const auto stored_size = require_unsigned(value.at("bytes"), "tensor bytes", true);
    auto transform         = PayloadTransform::None;
    std::vector<std::int32_t> group_map;
    if (value.contains("group_map")) {
        const auto& raw = value.at("group_map");
        if (!raw.is_array() || raw.empty()) {
            throw ArtifactError("group_map must be a nonempty array");
        }
        group_map.reserve(raw.size());
        for (const auto& entry : raw) {
            group_map.push_back(
                static_cast<std::int32_t>(require_unsigned(entry, "group_map entry", false)));
        }
    }
    if (value.contains("transform")) {
        const auto& name = require_string(value.at("transform"), "tensor transform");
        if (name == "q8_0-to-w8g32") {
            transform = PayloadTransform::Q8ToW8RowSplit;
        } else {
            throw ArtifactError("unknown payload transform: " + name);
        }
    }

    const auto& raw_shape = value.at("shape");
    if (!raw_shape.is_array()) { throw ArtifactError("tensor shape must be an array"); }
    std::vector<std::uint64_t> shape;
    shape.reserve(raw_shape.size());
    for (const auto& dim : raw_shape) {
        shape.push_back(require_unsigned(dim, "shape dimension", true));
    }

    std::vector<TensorSegment> segments;
    if (value.contains("segments")) {
        const auto& raw = value.at("segments");
        if (!raw.is_array() || raw.size() < 2) {
            throw ArtifactError("tensor " + name + ": segments must list at least two runs");
        }
        if (layout != StorageLayout::GgmlBlocksV1 || shape.size() != 2) {
            throw ArtifactError("tensor " + name + ": only a rank-two ggml-blocks-v1 object has segments");
        }
        std::uint64_t rows = 0;
        for (const auto& entry : raw) {
            TensorSegment segment{parse_format(require_string(entry.at("format"), "segment format")),
                                  require_unsigned(entry.at("rows"), "segment rows", true)};
            rows = checked_add(rows, segment.rows, "segment rows");
            segments.push_back(segment);
        }
        if (rows != shape.at(0)) {
            throw ArtifactError("tensor " + name + ": segments cover " + std::to_string(rows) +
                                " rows of " + std::to_string(shape.at(0)));
        }
        if (segments.front().format != format) {
            throw ArtifactError("tensor " + name + ": format must name the first segment's");
        }
    }
    std::uint64_t expected_size = 0;
    if (segments.empty()) {
        expected_size = tensor_encoded_size(layout, format, shape);
    } else {
        for (const TensorSegment& segment : segments) {
            const std::array<std::uint64_t, 2> segment_shape = {segment.rows, shape.at(1)};
            expected_size = checked_add(
                expected_size, tensor_encoded_size(layout, segment.format, segment_shape),
                "segment bytes");
        }
    }
    if (stored_size != expected_size) {
        throw ArtifactError("tensor " + name + " stores " + std::to_string(stored_size) +
                            " bytes; layout requires " + std::to_string(expected_size));
    }
    if (!group_map.empty()) {
        // Without a transform the map is not applied to the bytes: a ggml-blocks object read
        // as the file holds it, whose columns the runtime rearranges on the activation side.
        if (transform == PayloadTransform::None && layout != StorageLayout::GgmlBlocksV1) {
            throw ArtifactError("tensor " + name + " has a group_map but no transform to apply it");
        }
        const auto groups = shape.at(1) / 32;
        if (group_map.size() != groups) {
            throw ArtifactError("tensor " + name + " has " + std::to_string(group_map.size()) +
                                " group_map entries for " + std::to_string(groups) + " groups");
        }
        for (const std::int32_t entry : group_map) {
            if (entry < 0 || static_cast<std::uint64_t>(entry) >= groups) {
                throw ArtifactError("tensor " + name + " has a group_map entry out of range");
            }
        }
    }
    return {name,   std::move(shape), format,    layout,
            offset, stored_size,      transform, std::move(group_map), std::move(segments)};
}

ResourceDescriptor parse_resource(const Json& value) {
    static constexpr std::array members = {
        "name", "kind", "encoding", "offset", "bytes",
    };
    require_members(value, members, "resource entry", value.contains("runs") ? 1 : 0);
    return {
        require_string(value.at("name"), "resource name"),
        parse_encoding(require_string(value.at("encoding"), "resource encoding")),
        require_unsigned(value.at("offset"), "resource offset", false),
        require_unsigned(value.at("bytes"), "resource bytes", true),
    };
}

ObjectDescriptor parse_object(const Json& value) {
    if (!value.is_object()) { throw ArtifactError("each object entry must be a JSON object"); }
    const auto it = value.find("kind");
    if (it == value.end() || !it->is_string()) {
        throw ArtifactError("object kind must be 'tensor' or 'resource'");
    }
    const auto& kind = it->get_ref<const std::string&>();
    if (kind == "tensor") { return parse_tensor(value); }
    if (kind == "resource") { return parse_resource(value); }
    throw ArtifactError("object kind must be 'tensor' or 'resource'");
}

struct TransparentStringHash {
    using is_transparent = void;

    std::size_t operator()(std::string_view value) const noexcept {
        return std::hash<std::string_view>{}(value);
    }

    std::size_t operator()(const std::string& value) const noexcept {
        return (*this)(std::string_view(value));
    }
};

class MappedFile {
public:
    explicit MappedFile(const std::filesystem::path& path) {
        const int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC | O_DIRECT);
        if (fd < 0) {
            throw std::system_error(errno, std::generic_category(), "open " + path.string());
        }

        struct stat status {};

        if (::fstat(fd, &status) != 0) {
            const int error = errno;
            ::close(fd);
            throw std::system_error(error, std::generic_category(), "fstat " + path.string());
        }
        if (status.st_size < 0 ||
            static_cast<std::uintmax_t>(status.st_size) > std::numeric_limits<std::size_t>::max()) {
            ::close(fd);
            throw ArtifactError("artifact size does not fit the process address space");
        }

        const auto size = static_cast<std::size_t>(status.st_size);
        void* mapping   = nullptr;
        if (size != 0) {
            mapping = ::mmap(nullptr, size, PROT_READ, MAP_PRIVATE, fd, 0);
            if (mapping == MAP_FAILED) {
                const int error = errno;
                ::close(fd);
                throw std::system_error(error, std::generic_category(), "mmap " + path.string());
            }
        }
        fd_   = fd;
        data_ = static_cast<const std::byte*>(mapping);
        size_ = size;
    }

    ~MappedFile() {
        if (data_ != nullptr) { ::munmap(const_cast<std::byte*>(data_), size_); }
        if (fd_ >= 0) { ::close(fd_); }
    }

    MappedFile(const MappedFile&)            = delete;
    MappedFile& operator=(const MappedFile&) = delete;

    const std::byte* data() const noexcept { return data_; }

    std::size_t size() const noexcept { return size_; }

    std::size_t read_direct(std::uint64_t absolute_offset, std::span<std::byte> destination) const {
        constexpr std::size_t alignment = Reader::direct_io_alignment;
        if (absolute_offset % alignment != 0 || destination.size() % alignment != 0 ||
            reinterpret_cast<std::uintptr_t>(destination.data()) % alignment != 0) {
            throw ArtifactError("direct artifact read is not 4096-byte aligned");
        }
        if (absolute_offset > static_cast<std::uint64_t>(std::numeric_limits<off_t>::max()) ||
            destination.size() > static_cast<std::size_t>(std::numeric_limits<ssize_t>::max())) {
            throw ArtifactError("direct artifact read exceeds platform I/O limits");
        }

        ssize_t bytes = -1;
        do {
            bytes = ::pread(fd_, destination.data(), destination.size(),
                            static_cast<off_t>(absolute_offset));
        } while (bytes < 0 && errno == EINTR);
        if (bytes < 0) {
            throw std::system_error(errno, std::generic_category(), "direct artifact read");
        }
        return static_cast<std::size_t>(bytes);
    }

private:
    int fd_                = -1;
    const std::byte* data_ = nullptr;
    std::size_t size_      = 0;
};

} // namespace

std::string_view object_name(const ObjectDescriptor& object) noexcept {
    return std::visit([](const auto& descriptor) -> std::string_view { return descriptor.name; },
                      object);
}

std::uint64_t object_offset(const ObjectDescriptor& object) noexcept {
    return std::visit([](const auto& descriptor) { return descriptor.offset; }, object);
}

std::uint64_t object_bytes(const ObjectDescriptor& object) noexcept {
    return std::visit([](const auto& descriptor) { return descriptor.bytes; }, object);
}

struct Reader::Impl {
    explicit Impl(const std::filesystem::path& path) : file(path) {
        if (file.size() < kPrefixBytes) {
            throw ArtifactError("artifact is shorter than the v2 prefix");
        }
        if (std::equal(kV1Magic.begin(), kV1Magic.end(), file.data())) {
            throw ArtifactError("SInfer artifact v1 is no longer supported; "
                                "rebuild the serving artifact from its source checkpoint");
        }
        if (std::equal(kPreRenameMagic.begin(), kPreRenameMagic.end(), file.data())) {
            throw ArtifactError(
                "this artifact was written before the sinfer rename; its layout is "
                "identical and migrating it is a one-byte edit, not a reconversion: "
                "python3 -m surogate.serve.artifact.rename_magic <artifact>");
        }
        if (!std::equal(kMagic.begin(), kMagic.end(), file.data())) {
            throw ArtifactError("artifact magic is not SInfer v2");
        }

        const auto json_bytes = read_u64_le(file.data() + 8);
        if (json_bytes == 0) { throw ArtifactError("json_bytes must be positive"); }
        const auto metadata_end = checked_add(kPrefixBytes, json_bytes, "JSON range");
        payload_start           = align_up(metadata_end, kPayloadAlignment, "payload offset");
        if (metadata_end > file.size() || payload_start > file.size()) {
            throw ArtifactError("declared JSON or payload start extends beyond the file");
        }

        Json directory;
        try {
            const auto* begin = reinterpret_cast<const char*>(file.data() + kPrefixBytes);
            directory         = Json::parse(begin, begin + json_bytes);
        } catch (const Json::exception& error) {
            throw ArtifactError(std::string("invalid JSON directory: ") + error.what());
        }

        // "external" is one of the optional root members: an artifact that stores every object
        // itself does not carry it, and one that reads a file in place does.
        static constexpr std::array root_members = {"identity", "objects"};
        require_members(directory, root_members, "directory root",
                        (directory.contains("external") ? 1 : 0) +
                            (directory.contains("geometry") ? 1 : 0) +
                            (directory.contains("vision_geometry") ? 1 : 0) +
                            (directory.contains("layer_types") ? 1 : 0) +
                            (directory.contains("dflash_geometry") ? 1 : 0) +
                            (directory.contains("dflash_target_layers") ? 1 : 0));
        // "geometry" is another optional root member: the model's dimensions as a flat
        // object of numbers, keyed by the names the family's TextGeometry knows. A target lays
        // it over its compiled config, so an artifact without it loads as one size and an
        // artifact with it can be any size of its family.
        const auto read_geometry = [&](const char* member, std::map<std::string, double>& into) {
            if (!directory.contains(member)) { return; }
            const auto& raw = directory.at(member);
            if (!raw.is_object()) {
                throw ArtifactError(std::string(member) + " must be an object");
            }
            for (const auto& [key, value] : raw.items()) {
                if (!value.is_number()) {
                    throw ArtifactError(std::string(member) + "." + key + " must be a number");
                }
                into.emplace(key, value.get<double>());
            }
        };
        read_geometry("geometry", geometry);
        // The tower a checkpoint ships is its own size, independent of the text stack's, so it
        // is declared separately rather than folded into "geometry" where two members would
        // collide on names like "layers" and "hidden".
        read_geometry("vision_geometry", vision_geometry);
        read_geometry("dflash_geometry", dflash_geometry);
        try {
            family::TextGeometry text;
            text.override_from(geometry);
            family::VisionGeometry vision;
            vision.override_from(vision_geometry);
            if (directory.contains("dflash_target_layers")) {
                const auto& raw = directory.at("dflash_target_layers");
                if (!raw.is_array()) { throw ArtifactError("dflash_target_layers must be an array"); }
                for (const auto& layer : raw) {
                    if (!layer.is_number_integer() || layer.get<double>() < 0 || layer.get<double>() > 255) {
                        throw ArtifactError("dflash_target_layers entries must be integers in 0..255");
                    }
                    dflash_target_layers.push_back(layer.get<std::int32_t>());
                }
            }
            if (directory.contains("dflash_geometry") || directory.contains("dflash_target_layers")) {
                (void)family::DFlashGeometry::resolved(dflash_geometry, dflash_target_layers,
                                                       text.hidden, text.layers, text.output_rows);
            }
            if (directory.contains("layer_types")) {
                const auto& raw = directory.at("layer_types");
                if (!raw.is_array()) { throw ArtifactError("layer_types must be an array"); }
                for (const auto& kind : raw) {
                    layer_types.push_back(require_string(kind, "layer_types entry"));
                }
                text.apply_layer_types(layer_types);
            }
        } catch (const std::invalid_argument& error) {
            throw ArtifactError(error.what());
        }
        // An artifact may serve some of its objects straight out of another file rather than
        // copying them in. The table is absent from every artifact that does not, and those load
        // exactly as before.
        if (directory.contains("external")) {
            const auto& raw_external = directory.at("external");
            if (!raw_external.is_array()) { throw ArtifactError("external must be an array"); }
            for (const auto& entry : raw_external) {
                ExternalFile out;
                out.path  = require_string(entry.at("path"), "external path");
                out.bytes = require_unsigned(entry.at("bytes"), "external bytes", true);
                external.push_back(std::move(out));
            }
        }
        const auto& raw_identity                     = directory.at("identity");
        static constexpr std::array identity_members = {"model_id", "weights_id"};
        require_members(raw_identity, identity_members, "artifact identity",
                        raw_identity.contains("architecture") ? 1 : 0);
        identity.model_id   = require_string(raw_identity.at("model_id"), "model_id");
        identity.weights_id = require_string(raw_identity.at("weights_id"), "weights_id");
        if (raw_identity.contains("architecture")) {
            identity.architecture = require_string(raw_identity.at("architecture"), "architecture");
        }

        const auto& raw_objects = directory.at("objects");
        if (!raw_objects.is_array() || raw_objects.empty()) {
            throw ArtifactError("objects must be a nonempty array");
        }
        entries.reserve(raw_objects.size());
        index.reserve(raw_objects.size());

        const auto payload_bytes = static_cast<std::uint64_t>(file.size()) - payload_start;
        std::uint64_t cursor     = 0;
        for (const auto& raw_object : raw_objects) {
            auto object          = parse_object(raw_object);
            const auto name      = object_name(object);
            const auto offset    = object_offset(object);
            const auto bytes     = object_bytes(object);
            const auto alignment = std::visit(
                [](const auto& descriptor) {
                    using Descriptor = std::decay_t<decltype(descriptor)>;
                    if constexpr (std::is_same_v<Descriptor, TensorDescriptor>) {
                        return tensor_alignment(descriptor.layout);
                    } else {
                        return resource_alignment(descriptor.encoding);
                    }
                },
                object);

            const bool external_object = raw_object.contains("runs");
            if (!external_object && offset < cursor) {
                throw ArtifactError("object " + std::string(name) + " overlaps or is out of order");
            }
            if (!external_object && offset % alignment != 0) {
                throw ArtifactError("object " + std::string(name) + " is not " +
                                    std::to_string(alignment) + "-byte aligned");
            }
            std::vector<PayloadRun> object_runs;
            if (raw_object.contains("runs")) {
                // Served from an external file. `offset` and `bytes` still describe the object as
                // the binder sees it -- one logical, contiguous tensor -- but its bytes are
                // gathered from the runs rather than read at that offset here.
                const auto& raw_runs = raw_object.at("runs");
                if (!raw_runs.is_array() || raw_runs.empty()) {
                    throw ArtifactError("object " + std::string(name) + " has an empty run list");
                }
                std::uint64_t covered = 0;
                for (const auto& raw_run : raw_runs) {
                    PayloadRun run;
                    run.source = static_cast<std::uint32_t>(require_unsigned(raw_run.at("source"), "run source", true));
                    run.offset = require_unsigned(raw_run.at("offset"), "run offset", false);
                    run.bytes  = require_unsigned(raw_run.at("bytes"), "run bytes", true);
                    if (run.source == 0 || run.source > external.size()) {
                        throw ArtifactError("object " + std::string(name) +
                                            " names an external source that is not declared");
                    }
                    const auto run_end = checked_add(run.offset, run.bytes, "run range");
                    if (run_end > external[run.source - 1].bytes) {
                        throw ArtifactError("object " + std::string(name) +
                                            " runs past the end of " +
                                            external[run.source - 1].path);
                    }
                    covered = checked_add(covered, run.bytes, "object run coverage");
                    object_runs.push_back(run);
                }
                // A transformed object's runs carry its *source* bytes, which is a different
                // count from the stored form they become.
                const auto* tensor = std::get_if<TensorDescriptor>(&object);
                const bool transformed =
                    tensor != nullptr && tensor->transform != PayloadTransform::None;
                if (!transformed && covered != bytes) {
                    throw ArtifactError("object " + std::string(name) + " declares " +
                                        std::to_string(bytes) + " bytes but its runs cover " +
                                        std::to_string(covered));
                }
                if (transformed && object_runs.empty()) {
                    throw ArtifactError("object " + std::string(name) + " has no source runs");
                }
            } else {
                const auto end = checked_add(offset, bytes, "object payload range");
                if (end > payload_bytes) {
                    throw ArtifactError("object " + std::string(name) + " extends beyond the file");
                }
                object_runs.push_back(PayloadRun{0, checked_add(payload_start, offset,
                                                                "absolute payload offset"),
                                                 bytes});
                cursor = end;
            }
            const auto object_index = entries.size();
            auto [_, inserted]      = index.emplace(std::string(name), object_index);
            if (!inserted) { throw ArtifactError("duplicate object name: " + std::string(name)); }
            entries.push_back(std::move(object));
            runs.push_back(std::move(object_runs));
        }

        // Mapped last, so a malformed directory fails before any of them is opened. A relative
        // path is resolved against the artifact's own directory, which keeps a model and the
        // GGUF it reads movable together.
        external_maps.reserve(external.size());
        for (const ExternalFile& entry : external) {
            std::filesystem::path resolved = entry.path;
            if (resolved.is_relative()) { resolved = path.parent_path() / resolved; }
            std::error_code ec;
            const auto actual = std::filesystem::file_size(resolved, ec);
            if (ec) {
                throw ArtifactError("this artifact serves weights from " + resolved.string() +
                                    ", which cannot be read: " + ec.message());
            }
            if (actual != entry.bytes) {
                throw ArtifactError(resolved.string() + " is " + std::to_string(actual) +
                                    " bytes but this artifact was written against " +
                                    std::to_string(entry.bytes) +
                                    "; the file it serves weights from has changed");
            }
            external_maps.push_back(std::make_unique<MappedFile>(resolved));
        }
    }

    MappedFile& file_for(std::uint32_t source) {
        if (source == 0) { return file; }
        if (source > external_maps.size()) { throw ArtifactError("unknown external source"); }
        return *external_maps[source - 1];
    }
    const MappedFile& file_for(std::uint32_t source) const {
        return const_cast<Impl*>(this)->file_for(source);
    }

    MappedFile file;
    ArtifactIdentity identity;
    std::vector<ObjectDescriptor> entries;
    std::vector<std::vector<PayloadRun>> runs;
    std::vector<ExternalFile> external;
    std::map<std::string, double> geometry;
    std::map<std::string, double> vision_geometry;
    std::vector<std::string> layer_types;
    std::map<std::string, double> dflash_geometry;
    std::vector<std::int32_t> dflash_target_layers;
    std::vector<std::unique_ptr<MappedFile>> external_maps; // MappedFile owns an fd and a mapping
    std::unordered_map<std::string, std::size_t, TransparentStringHash, std::equal_to<>> index;
    std::uint64_t payload_start = 0;
};

Reader::Reader(const std::filesystem::path& path) : impl_(std::make_unique<Impl>(path)) {}

Reader::~Reader()                            = default;
Reader::Reader(Reader&&) noexcept            = default;
Reader& Reader::operator=(Reader&&) noexcept = default;

const ArtifactIdentity& Reader::identity() const noexcept { return impl_->identity; }

const std::vector<ObjectDescriptor>& Reader::objects() const noexcept { return impl_->entries; }

const ObjectDescriptor* Reader::find(std::string_view name) const noexcept {
    const auto it = impl_->index.find(name);
    return it == impl_->index.end() ? nullptr : &impl_->entries[it->second];
}

std::uint64_t Reader::file_bytes() const noexcept { return impl_->file.size(); }

std::uint64_t Reader::payload_offset() const noexcept { return impl_->payload_start; }

PayloadSpan Reader::payload(const ObjectDescriptor& object) const {
    const auto index = static_cast<std::size_t>(&object - impl_->entries.data());
    if (index >= impl_->runs.size() || impl_->runs[index].size() != 1) {
        throw ArtifactError("object " + std::string(object_name(object)) +
                            " is assembled from several runs and has no single payload span");
    }
    const PayloadRun& run  = impl_->runs[index].front();
    const MappedFile& file = impl_->file_for(run.source);
    const auto end         = checked_add(run.offset, run.bytes, "absolute payload range");
    if (end > file.size()) { throw ArtifactError("object payload extends beyond the file"); }
    return {
        run.offset,
        std::span<const std::byte>(file.data() + run.offset, static_cast<std::size_t>(run.bytes)),
        run.source,
    };
}

std::span<const std::byte> Reader::run_span(const PayloadRun& run) const {
    const MappedFile& file = impl_->file_for(run.source);
    const auto end         = checked_add(run.offset, run.bytes, "absolute payload range");
    if (end > file.size()) { throw ArtifactError("run extends beyond its file"); }
    return {file.data() + run.offset, static_cast<std::size_t>(run.bytes)};
}

std::span<const PayloadRun> Reader::runs(const ObjectDescriptor& object) const {
    const auto index = static_cast<std::size_t>(&object - impl_->entries.data());
    if (index >= impl_->runs.size()) {
        throw ArtifactError("object does not belong to this reader");
    }
    return impl_->runs[index];
}

const std::vector<ExternalFile>& Reader::external_files() const noexcept { return impl_->external; }

PayloadSpan Reader::payload(std::string_view name) const {
    const auto* object = find(name);
    if (object == nullptr) { throw ArtifactError("unknown artifact object: " + std::string(name)); }
    return payload(*object);
}

const std::map<std::string, double>& Reader::geometry() const noexcept { return impl_->geometry; }

const std::vector<std::string>& Reader::layer_types() const noexcept { return impl_->layer_types; }
const std::map<std::string, double>& Reader::dflash_geometry() const noexcept { return impl_->dflash_geometry; }
const std::vector<std::int32_t>& Reader::dflash_target_layers() const noexcept { return impl_->dflash_target_layers; }

const std::map<std::string, double>& Reader::vision_geometry() const noexcept {
    return impl_->vision_geometry;
}

std::size_t Reader::read_direct(std::uint32_t source, std::uint64_t absolute_offset,
                                std::span<std::byte> destination) const {
    return impl_->file_for(source).read_direct(absolute_offset, destination);
}

} // namespace sinfer::artifact
