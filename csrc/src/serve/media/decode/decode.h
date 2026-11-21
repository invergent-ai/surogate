#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace sinfer::media::decode {

enum class ErrorKind {
    BudgetExceeded,
};

class Error final : public std::runtime_error {
public:
    Error(ErrorKind kind, std::string message)
        : std::runtime_error(std::move(message)), kind_(kind) {}

    [[nodiscard]] ErrorKind kind() const noexcept { return kind_; }

private:
    ErrorKind kind_;
};

struct Policy {
    std::size_t max_bytes                  = 256ULL << 20;
    std::uint64_t max_decoded_pixels       = 64ULL * 1024ULL * 1024ULL;
    std::uint64_t max_decoded_video_pixels = 128ULL * 1024ULL * 1024ULL;
    int max_video_source_frames            = 100'000;
    double max_video_duration_seconds      = 600.0;
    std::function<void()> checkpoint;
};

struct Image {
    int width  = 0;
    int height = 0;
    std::vector<std::uint8_t> rgb;
};

struct Video {
    int width        = 0;
    int height       = 0;
    int total_frames = 0;
    double fps       = 0.0;
    double duration  = 0.0;
    std::vector<int> indices;
    std::vector<Image> frames;
};

/// Whether this binary can decode images and video at all: false when it was built without FFmpeg
/// (SINFER_ENABLE_FFMPEG=OFF, decode_stub.cpp), where every decode throws.
[[nodiscard]] bool available() noexcept;

Image decode_image(std::span<const std::uint8_t> bytes, const Policy& policy);

/// An image as Pillow holds it after `Image.open(...).load()` and before any mode conversion, reduced to 8-bit RGBA
/// the way Pillow's own `convert("RGBA")` does (ENGINE-INKLING lane V-b; used only by the Inkling-2 processor, which
/// reproduces vLLM's `vllm/models/inkling2/processing.py::decode_image`). Differences from `decode_image`: NO EXIF /
/// display-matrix orientation (Pillow does not transpose on open); the first frame; a JPEG lacking its EOI marker gets
/// `FF D9` appended first (vLLM does that before Pillow sees the bytes); alpha is KEPT (255 where the source has none);
/// 16-bit samples follow Pillow's modes: RGB / RGBA / LA keep the high byte (rawmodes "RGB;16B" ...), a 16-bit gray
/// PNG is Pillow mode "I;16", whose conversion clips at 255.
struct PilImage {
    int width  = 0;
    int height = 0;
    std::vector<std::uint8_t> rgba;  // [height, width, 4]
    /// Pillow's `has_transparency_data`: the source carries alpha (an alpha channel, a tRNS chunk, a GIF transparent
    /// index). When false every alpha byte is 255.
    bool has_alpha = false;
    /// True when the pixels are Pillow's by construction (lossless formats whose samples FFmpeg hands over unchanged:
    /// PNG, GIF, BMP, lossless WebP, ...). False for formats whose RGB Pillow computes in its own decoder library
    /// (JPEG: libjpeg-turbo's IDCT, fancy upsampling and YCbCr tables; lossy WebP: libwebp's upsampler), where FFmpeg's
    /// pixels are FFmpeg's (converted to RGB with swscale, nearest chroma) and may differ by a few levels.
    bool pil_exact = true;
    /// FFmpeg's name of the decoded frame's pixel format (diagnostics).
    std::string pixel_format;
};

struct PilOptions {
    /// vLLM's Inkling processor (processing.py decode_image) appends FF D9 to a JPEG lacking its EOI marker before
    /// Pillow opens it. vLLM's server path (multimodal/media/image.py ImageMediaIO.load_bytes) does not, and Pillow
    /// then refuses the image: "image file is truncated (N bytes not processed)".
    bool append_jpeg_eoi = true;
    /// ImageOps.exif_transpose (the server path's normalize_image): the EXIF / XMP orientation applied to the pixels.
    bool exif_transpose = false;
};

PilImage decode_image_pil(std::span<const std::uint8_t> bytes, const Policy& policy, const PilOptions& options = {});

/// The image's size as Pillow reports it from the header (`Image.open(...).size`, before any transpose) and the
/// orientation exif_transpose would apply (1 = none; Pillow's sources: EXIF in JPEG APP1 / PNG eXIf / WebP EXIF /
/// TIFF IFD0, else XMP tiff:Orientation). No pixel is decoded.
struct PilHeader {
    int width       = 0;
    int height      = 0;
    int orientation = 1;
};

PilHeader probe_image_pil(std::span<const std::uint8_t> bytes, const PilOptions& options = {});

Video decode_video(std::span<const std::uint8_t> bytes, const Policy& policy, double target_fps,
                   int min_frames, int max_frames, bool temporal_pairs = false);

} // namespace sinfer::media::decode
