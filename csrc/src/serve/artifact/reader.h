#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <map>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

namespace sinfer::artifact {

class ArtifactError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

enum class NumericFormat {
    BF16,
    FP32,
    I32,
    Q4G64_F16S,
    Q5G64_F16S,
    Q6G64_F16S,
    W8G32_F16S,
    NVFP4,
    FP8_E4M3FN_ROW_BF16S,
    Q2_K,
    Q3_K,
    Q4_K,
    Q5_K,
    Q6_K,
    Q8_0,
    Q4_1,
    Q5_1,
};

enum class StorageLayout {
    ContiguousLeV1,
    RowSplitK128V1,
    BlockScaleK16M128x4V1,
    RowScaleV1,
    GgmlBlocksV1,
};

enum class ResourceEncoding {
    RawBytesV1,
};

std::string_view format_name(NumericFormat format) noexcept;
std::string_view layout_name(StorageLayout layout) noexcept;
std::string_view encoding_name(ResourceEncoding encoding) noexcept;

std::uint64_t tensor_alignment(StorageLayout layout) noexcept;
std::uint64_t resource_alignment(ResourceEncoding encoding) noexcept;
/// Bytes of one 256-value superblock for a GGML K-quant format.
std::uint64_t ggml_block_bytes(NumericFormat format);
/// Values a stored block holds: 256 for a K-quant superblock, 32 for the plain block types.
std::uint64_t ggml_block_values(NumericFormat format);
std::uint64_t tensor_encoded_size(StorageLayout layout, NumericFormat format,
                                  std::span<const std::uint64_t> shape);

struct RowSplitGeometry {
    std::uint64_t rows                 = 0;
    std::uint64_t columns              = 0;
    std::uint64_t padded_columns       = 0;
    std::uint64_t group_size           = 0;
    std::uint64_t groups_per_row       = 0;
    std::uint64_t low_bytes_per_group  = 0;
    std::uint64_t high_bytes_per_group = 0;
    std::uint64_t low_plane_bytes      = 0;
    std::uint64_t high_plane_offset    = 0;
    std::uint64_t high_plane_bytes     = 0;
    std::uint64_t scale_plane_offset   = 0;
    std::uint64_t scale_plane_bytes    = 0;
    std::uint64_t encoded_bytes        = 0;
};

RowSplitGeometry row_split_geometry(NumericFormat format, std::span<const std::uint64_t> shape);

struct BlockScaleGeometry {
    std::uint64_t rows                  = 0;
    std::uint64_t columns               = 0;
    std::uint64_t groups_per_row        = 0;
    std::uint64_t k_tiles               = 0;
    std::uint64_t code_plane_bytes      = 0;
    std::uint64_t scale_plane_offset    = 0;
    std::uint64_t scale_plane_bytes     = 0;
    std::uint64_t weight_divisor_offset = 0;
    std::uint64_t encoded_bytes         = 0;
};

BlockScaleGeometry block_scale_geometry(NumericFormat format, std::span<const std::uint64_t> shape);

struct RowScaleGeometry {
    std::uint64_t rows               = 0;
    std::uint64_t columns            = 0;
    std::uint64_t code_plane_bytes   = 0;
    std::uint64_t scale_plane_offset = 0;
    std::uint64_t scale_plane_bytes  = 0;
    std::uint64_t encoded_bytes      = 0;
};

RowScaleGeometry row_scale_geometry(NumericFormat format, std::span<const std::uint64_t> shape);

/// How an object's runs become its stored form. `None` is a copy, which is what every object
/// written into the artifact's own payload is. `Q8ToW8RowSplit` rearranges Q8_0 blocks into the
/// row-split planes -- the same numbers either way, so the file's bytes can serve a weight whose
/// kernels want planes without either a copy or a loss.
enum class PayloadTransform : std::uint8_t {
    None,
    Q8ToW8RowSplit,
};

struct TensorDescriptor {
    std::string name;
    std::vector<std::uint64_t> shape;
    NumericFormat format;
    StorageLayout layout;
    std::uint64_t offset;
    std::uint64_t bytes;
    PayloadTransform transform = PayloadTransform::None;
    /// `k / 32` entries naming the source block each destination block takes, when the transform
    /// also carries a column permutation. Empty when the columns are in order.
    std::vector<std::int32_t> group_map;
};

struct ResourceDescriptor {
    std::string name;
    ResourceEncoding encoding;
    std::uint64_t offset;
    std::uint64_t bytes;
};

using ObjectDescriptor = std::variant<TensorDescriptor, ResourceDescriptor>;

std::string_view object_name(const ObjectDescriptor& object) noexcept;
std::uint64_t object_offset(const ObjectDescriptor& object) noexcept;
std::uint64_t object_bytes(const ObjectDescriptor& object) noexcept;

struct PayloadSpan {
    std::uint64_t absolute_offset;
    std::span<const std::byte> data;
    /// Which file the offset is in: zero is this artifact, one and up index `external_files()`.
    std::uint32_t source = 0;
};

/// One contiguous run of an object's bytes. An object written into the artifact's own payload is
/// a single run at source zero -- every artifact before external sources existed is exactly that.
/// An object served straight from a GGUF is one run per contiguous stretch of that file, which is
/// more than one whenever the recipe interleaves two of its tensors: a fused routed gate/up takes
/// each expert's gate rows and then its up rows, and those live in two different GGUF tensors.
struct PayloadRun {
    std::uint32_t source = 0;
    std::uint64_t offset = 0; // absolute within that file
    std::uint64_t bytes  = 0;
};

/// A file the artifact serves bytes from without copying them into itself.
struct ExternalFile {
    std::string path;
    std::uint64_t bytes = 0;
};

struct ArtifactIdentity {
    std::string model_id;
    std::string weights_id;

    bool operator==(const ArtifactIdentity&) const = default;
};

class Reader {
public:
    static constexpr std::size_t direct_io_alignment = 4096;

    explicit Reader(const std::filesystem::path& path);
    ~Reader();

    Reader(Reader&&) noexcept;
    Reader& operator=(Reader&&) noexcept;
    Reader(const Reader&)            = delete;
    Reader& operator=(const Reader&) = delete;

    const ArtifactIdentity& identity() const noexcept;
    const std::vector<ObjectDescriptor>& objects() const noexcept;
    const ObjectDescriptor* find(std::string_view name) const noexcept;

    std::uint64_t file_bytes() const noexcept;
    std::uint64_t payload_offset() const noexcept;
    /// Valid for any object that is one contiguous run; an object spread over several runs of an
    /// external file has no single span and must be read through `runs()`.
    PayloadSpan payload(const ObjectDescriptor& object) const;
    PayloadSpan payload(std::string_view name) const;
    /// The runs an object's bytes are assembled from, in order.
    std::span<const PayloadRun> runs(const ObjectDescriptor& object) const;
    const std::vector<ExternalFile>& external_files() const noexcept;
    /// The artifact's declared dimensions, keyed as `family::TextGeometry` names them; empty
    /// for an artifact written without a `geometry` member.
    const std::map<std::string, double>& geometry() const noexcept;
    /// The declared dimensions of the artifact's vision tower, keyed as `family::VisionGeometry`
    /// names them; empty for a text-only artifact and for one written before the member existed.
    const std::map<std::string, double>& vision_geometry() const noexcept;
    std::size_t read_direct(std::uint32_t source, std::uint64_t absolute_offset,
                            std::span<std::byte> destination) const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace sinfer::artifact
