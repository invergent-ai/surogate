#pragma once

// Inkling-2 image preprocessing (ENGINE-INKLING lane V-b): the bytes of one image -> the decoder-token grid and the
// vision tower's input, bit-identical to vLLM (image vllm/vllm-openai:inkling-2-mini-x86_64-7cb5dae).
//
// Reference, read in rl/engine-inkling-2026-11-21/ref/src/vllm/models/inkling2/:
//   processing.py   decode_image (Pillow on the original bytes, JPEG EOI fix, NO EXIF, alpha composited on an 8 px
//                   checkerboard 204 / 153 as (rgb*a + bg*(255-a) + 127) // 255), Inkling2ImageProcessor (no resize:
//                   the image is padded bottom / right with -1 to ceil(H/40) x ceil(W/40) cells, one decoder token per
//                   40 x 40 cell), the prompt update <|content_image|> -> <|content_image|> + N x <|unused_200054|>.
//   nvidia/vision.py, nvidia/ops/vision.py   gather_patches: each 40 x 40 cell is 2 x 2 native 20 x 20 patches, one
//                   row of 1,200 features (py, px, channel) each, value 2 * (bf16((v/255 - mean)/clip_std)*std + mean) - 1
//                   in fp32 without FMA, stored as bf16.
// The payload handed to the engine (PreparedMediaPayload) is those rows in MERGE order -- cell by cell in raster order,
// inside a cell (dy, dx) = (0,0), (0,1), (1,0), (1,1) -- i.e. 4 consecutive rows per decoder token, the layout the
// family's vision path uses. vLLM's tower reads them in native raster order; the encoder (lane V-c) owns that mapping.
// Test: sinfer_inkling_vision_test over rl/engine-inkling-2026-11-21/vectors/vision (lane V-a's vectors from vLLM).

#include "media/decode/decode.h"

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace sinfer::family::frontend_internal::inkling {

inline constexpr int kImageCell        = 40;  // PATCH_SIZE: one decoder token per 40 x 40 pixel cell
inline constexpr int kNativePatch      = 20;  // the ViT's patch (native_patch_size)
inline constexpr int kMergeSize        = 2;   // adapter_merge_size: 2 x 2 native patches per decoder token
inline constexpr int kPatchFeatures    = kNativePatch * kNativePatch * 3;  // 1,200: (py, px, channel)
inline constexpr int kPatchesPerToken  = kMergeSize * kMergeSize;
/// vLLM's per-image limit at --max-num-batched-tokens 4096 (the encoder cache size; agent-EN's decision for ours).
inline constexpr std::uint32_t kDefaultMaxImageTokens = 4'096;
/// vLLM's per-prompt image count when --limit-mm-per-prompt does not name images (config/multimodal.py).
inline constexpr std::uint32_t kDefaultImagesPerPrompt = 999;

/// Which of vLLM's two decodes the pixels follow (lane V-a measured on a running vLLM, 18:0x: a chat request's image
/// never reaches processing.py's decode_image -- the chat parser hands the Inkling processor the image vLLM's generic
/// ImageMediaIO already converted).
enum class DecodePath : std::uint8_t {
    /// What vLLM's SERVER feeds the encoder for a chat request (multimodal/media/image.py load_bytes): Pillow,
    /// VLLM_MAX_IMAGE_PIXELS check, ImageOps.exif_transpose, NO EOI fix (a JPEG lacking it is refused as Pillow
    /// refuses it), transparency composited on WHITE (rgba_to_rgb: Image.paste with the alpha as mask), RGB.
    /// Errors: "Failed to load image: {Pillow's message}".
    Served,
    /// processing.py decode_image, the Inkling processor's own: no EXIF, the JPEG EOI fix, alpha on the 8 px
    /// checkerboard. vLLM reaches it only for raw bytes / arrays handed to the processor directly (not chat).
    Processor,
};

/// vLLM's served pixel limit (envs.VLLM_MAX_IMAGE_PIXELS default) and Pillow's decompression-bomb error threshold
/// (2 x Image.MAX_IMAGE_PIXELS): the same number.
inline constexpr std::uint64_t kMaxImagePixels = 178'956'970;

/// vLLM's decoded image: HWC uint8 RGB after the path's alpha composite.
struct DecodedImage {
    int width  = 0;
    int height = 0;
    std::vector<std::uint8_t> rgb;  // [height, width, 3]
    bool has_alpha = false;         // composited on the checkerboard
    bool pil_exact = true;          // see media::decode::PilImage::pil_exact
    std::string pixel_format;       // FFmpeg's decoded frame format (diagnostics)
};

struct ImageGrid {
    int rows = 0;  // ceil(height / 40)
    int cols = 0;  // ceil(width / 40)
    [[nodiscard]] std::uint64_t tokens() const noexcept {
        return static_cast<std::uint64_t>(rows) * static_cast<std::uint64_t>(cols);
    }
};

/// The image's size in decoder-token cells before any pixel is decoded, with vLLM's refusals that precede decoding
/// (the served pixel limit / Pillow's bomb check). Throws RequestError-compatible std::invalid_argument texts.
struct ImageProbe {
    int width  = 0;  // after the path's transpose
    int height = 0;
};
[[nodiscard]] ImageProbe probe_image(std::span<const std::uint8_t> bytes, DecodePath path);
[[nodiscard]] DecodedImage decode_image(std::span<const std::uint8_t> bytes, const media::decode::Policy& policy,
                                        DecodePath path = DecodePath::Served);
/// The checkerboard composite of an 8-bit RGBA image (processing.py _composite_on_checkerboard).
[[nodiscard]] DecodedImage composite(const media::decode::PilImage& image);
/// The white composite of vLLM's server path (multimodal/image.py rgba_to_rgb: Image.new("RGB", size, white).paste(
/// image, mask=alpha) = Pillow Paste.c BLEND per channel: DIV255(255 * (255 - a) + c * a)).
[[nodiscard]] DecodedImage composite_white(const media::decode::PilImage& image);
[[nodiscard]] ImageGrid image_grid(int height, int width);

/// The normalised bf16 value gather_patches writes for raw value `value` (-1 = padding, 0..255) of `channel`.
[[nodiscard]] std::uint16_t normalized_bf16(int value, int channel) noexcept;

/// Writes the payload: [4 * tokens, 1,200] bf16, merge order (see above). `out` must hold exactly that many values.
void write_patches_merge(const DecodedImage& image, std::span<std::uint16_t> out);
/// vLLM's `vision_pixels` for one image: [tokens, 40, 40, 3] bf16 raw values (0..255, padding -1). Tests only.
[[nodiscard]] std::vector<std::uint16_t> raw_pixels(const DecodedImage& image);

/// vLLM's refusals, word for word.
/// v1/engine/input_processor.py: an image item larger than the encoder cache.
[[nodiscard]] std::string image_too_large_message(std::uint64_t tokens, std::uint64_t cap);
/// The per-prompt count refusal is inkling_render.h's image_count_message (the renderer's parse enforces it).

} // namespace sinfer::family::frontend_internal::inkling
