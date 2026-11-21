// ENGINE-INKLING lane V-b: a JPEG decoded the way Pillow 13 decodes it (the reference for Inkling-2's vision processor,
// vllm/models/inkling2/processing.py decode_image, and vLLM's server path multimodal/media/image.py), with the same
// library: libjpeg-turbo 3.2.0 (csrc/cmake/serve_pil_decoders.cmake). Pillow (JpegImagePlugin.py + JpegDecode.c): the
// mode follows the number of components -- 1 "L", 3 "RGB", 4 "CMYK" with rawmode "CMYK;I" (Adobe's inverted CMYK,
// always assumed) -- and the decoder runs with libjpeg's defaults (JDCT_ISLOW, fancy upsampling) and out_color_space
// GRAYSCALE / RGB / CMYK. Pillow's convert to RGB(A): L -> R = G = B; CMYK;I -> inverted, then cmyk2rgb
// (out = clip(nk - muldiv255(c, nk)), nk = 255 - k). No alpha in any JPEG mode.
//
// Pillow feeds the decoder through a SUSPENDING source (JpegDecode.c): when the data ends before the last scanline
// (a JPEG cut short, or one whose last MCU row ends without a marker, as one lacking its EOI), the decoder suspends,
// ImageFile.load finds no more bytes and raises "image file is truncated (N bytes not processed)", N = the bytes the
// decoder had not consumed. The same source here gives the same N. A suspended jpeg_finish_decompress after the last
// scanline is not an error (JpegDecode.c case 4). A libjpeg fatal error is Pillow's IMAGING_CODEC_BROKEN:
// "broken data stream when reading image file".

#include "media/decode/decode.h"

#include <csetjmp>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <jpeglib.h>

namespace sinfer::media::decode::detail {
namespace {

struct ErrorManager {
    jpeg_error_mgr pub{};
    std::jmp_buf jump{};
    char message[JMSG_LENGTH_MAX]{};
    bool pillow_message = false;  // message is Pillow's own refusal (its JPEG plugin), not a libjpeg fatal error
};

extern "C" void sinfer_pil_jpeg_error_exit(j_common_ptr cinfo) {
    auto* error = reinterpret_cast<ErrorManager*>(cinfo->err);
    (*cinfo->err->format_message)(cinfo, error->message);
    std::longjmp(error->jump, 1);
}

// Pillow silences libjpeg's messages (warnings do not stop the decode).
extern "C" void sinfer_pil_jpeg_output_message(j_common_ptr) {}

// JpegDecode.c's source manager: the whole input is one buffer; asking for more suspends.
extern "C" void sinfer_pil_jpeg_init_source(j_decompress_ptr) {}
extern "C" boolean sinfer_pil_jpeg_fill_input_buffer(j_decompress_ptr) { return FALSE; }
extern "C" void sinfer_pil_jpeg_skip_input_data(j_decompress_ptr cinfo, long count) {
    jpeg_source_mgr* src = cinfo->src;
    if (count <= 0) { return; }
    if (static_cast<std::size_t>(count) > src->bytes_in_buffer) {
        src->next_input_byte += src->bytes_in_buffer;
        src->bytes_in_buffer = 0;
    } else {
        src->next_input_byte += count;
        src->bytes_in_buffer -= static_cast<std::size_t>(count);
    }
}
extern "C" void sinfer_pil_jpeg_term_source(j_decompress_ptr) {}

enum class Outcome { Ok, Failed, TooLarge, Truncated };

struct Raw {
    std::vector<std::uint8_t> pixels;
    int width = 0, height = 0, components = 0;
    std::size_t unconsumed = 0;  // Truncated: the bytes Pillow reports as not processed
};

// Only trivially destructible locals between setjmp and the libjpeg calls: a longjmp skips no destructor.
Outcome decode_raw(const std::uint8_t* data, std::size_t size, std::uint64_t max_pixels, bool header_only,
                   ErrorManager& error, Raw& raw) {
    jpeg_decompress_struct cinfo{};
    jpeg_source_mgr source{};
    cinfo.err                = jpeg_std_error(&error.pub);
    error.pub.error_exit     = sinfer_pil_jpeg_error_exit;
    error.pub.output_message = sinfer_pil_jpeg_output_message;
    if (setjmp(error.jump) != 0) {
        jpeg_destroy_decompress(&cinfo);
        return Outcome::Failed;
    }
    jpeg_create_decompress(&cinfo);
    source.init_source       = sinfer_pil_jpeg_init_source;
    source.fill_input_buffer = sinfer_pil_jpeg_fill_input_buffer;
    source.skip_input_data   = sinfer_pil_jpeg_skip_input_data;
    source.resync_to_restart = jpeg_resync_to_restart;
    source.term_source       = sinfer_pil_jpeg_term_source;
    source.next_input_byte   = data;
    source.bytes_in_buffer   = size;
    cinfo.src                = &source;
    const auto truncated = [&] {
        raw.unconsumed = cinfo.src->bytes_in_buffer;
        jpeg_destroy_decompress(&cinfo);
        return Outcome::Truncated;
    };
    int header = JPEG_HEADER_TABLES_ONLY;
    while (header == JPEG_HEADER_TABLES_ONLY) { header = jpeg_read_header(&cinfo, FALSE); }
    if (header == JPEG_SUSPENDED) { return truncated(); }
    raw.width  = static_cast<int>(cinfo.image_width);
    raw.height = static_cast<int>(cinfo.image_height);
    if (header_only) {
        jpeg_destroy_decompress(&cinfo);
        return Outcome::Ok;
    }
    if (cinfo.data_precision != 8) {
        // Pillow: "cannot handle {bits}-bit layers".
        std::snprintf(error.message, sizeof(error.message), "cannot handle %d-bit layers", cinfo.data_precision);
        error.pillow_message = true;
        jpeg_destroy_decompress(&cinfo);
        return Outcome::Failed;
    }
    switch (cinfo.num_components) {
    case 1: cinfo.out_color_space = JCS_GRAYSCALE; break;
    case 3: cinfo.out_color_space = JCS_RGB; break;
    case 4: cinfo.out_color_space = JCS_CMYK; break;
    default:
        std::snprintf(error.message, sizeof(error.message), "cannot handle %d-layer images", cinfo.num_components);
        error.pillow_message = true;
        jpeg_destroy_decompress(&cinfo);
        return Outcome::Failed;
    }
    if (static_cast<std::uint64_t>(cinfo.image_width) * cinfo.image_height > max_pixels) {
        jpeg_destroy_decompress(&cinfo);
        return Outcome::TooLarge;
    }
    if (!jpeg_start_decompress(&cinfo)) { return truncated(); }
    raw.width      = static_cast<int>(cinfo.output_width);
    raw.height     = static_cast<int>(cinfo.output_height);
    raw.components = cinfo.output_components;
    try {
        raw.pixels.resize(static_cast<std::size_t>(raw.width) * static_cast<std::size_t>(raw.height) *
                          static_cast<std::size_t>(raw.components));
    } catch (...) {
        jpeg_destroy_decompress(&cinfo);
        throw;
    }
    while (cinfo.output_scanline < cinfo.output_height) {
        JSAMPROW row = raw.pixels.data() + static_cast<std::size_t>(cinfo.output_scanline) *
                                               static_cast<std::size_t>(raw.width) *
                                               static_cast<std::size_t>(raw.components);
        if (jpeg_read_scanlines(&cinfo, &row, 1) != 1) { return truncated(); }
    }
    (void)jpeg_finish_decompress(&cinfo);  // suspending here (no EOI) is not an error once every row is read
    jpeg_destroy_decompress(&cinfo);
    return Outcome::Ok;
}

[[noreturn]] void refuse(Outcome outcome, const ErrorManager& error, const Raw& raw) {
    if (outcome == Outcome::TooLarge) { throw Error(ErrorKind::BudgetExceeded, "decoded media pixels exceed processor limit"); }
    if (outcome == Outcome::Truncated) {
        throw std::invalid_argument("image file is truncated (" + std::to_string(raw.unconsumed) + " bytes not processed)");
    }
    // Pillow's plugin refuses bit depth / layer count itself; a libjpeg fatal error is its IMAGING_CODEC_BROKEN.
    throw std::invalid_argument(error.pillow_message ? std::string(error.message)
                                                     : std::string("broken data stream when reading image file"));
}

} // namespace

std::pair<int, int> probe_jpeg_pil(std::span<const std::uint8_t> bytes) {
    ErrorManager error;
    Raw raw;
    const Outcome outcome = decode_raw(bytes.data(), bytes.size(), 0, /*header_only=*/true, error, raw);
    if (outcome != Outcome::Ok) { refuse(outcome, error, raw); }
    return {raw.width, raw.height};
}

PilImage decode_jpeg_pil(std::span<const std::uint8_t> bytes, std::uint64_t max_pixels) {
    ErrorManager error;
    Raw raw;
    const Outcome outcome = decode_raw(bytes.data(), bytes.size(), max_pixels, /*header_only=*/false, error, raw);
    if (outcome != Outcome::Ok) { refuse(outcome, error, raw); }
    const int width = raw.width, height = raw.height, components = raw.components;
    PilImage out;
    out.width        = width;
    out.height       = height;
    out.pixel_format = components == 1 ? "jpeg-L" : components == 3 ? "jpeg-RGB" : "jpeg-CMYK";
    out.pil_exact    = true;
    const std::size_t pixels = static_cast<std::size_t>(width) * static_cast<std::size_t>(height);
    out.rgba.resize(pixels * 4);
    for (std::size_t i = 0; i < pixels; ++i) {
        const std::uint8_t* s = raw.pixels.data() + i * static_cast<std::size_t>(components);
        std::uint8_t* d       = out.rgba.data() + i * 4;
        if (components == 1) {
            d[0] = d[1] = d[2] = s[0];
        } else if (components == 3) {
            d[0] = s[0];
            d[1] = s[1];
            d[2] = s[2];
        } else {
            // Unpack "CMYK;I" (each byte inverted), then Convert.c cmyk2rgb.
            const int c = 255 - s[0], m = 255 - s[1], y = 255 - s[2], k = 255 - s[3];
            const int nk = 255 - k;
            const auto muldiv255 = [](int a, int b) {
                const int tmp = a * b + 128;
                return ((tmp >> 8) + tmp) >> 8;
            };
            const auto clip8 = [](int v) { return static_cast<std::uint8_t>(v < 0 ? 0 : v > 255 ? 255 : v); };
            d[0] = clip8(nk - muldiv255(c, nk));
            d[1] = clip8(nk - muldiv255(m, nk));
            d[2] = clip8(nk - muldiv255(y, nk));
        }
        d[3] = 255;
    }
    return out;
}

} // namespace sinfer::media::decode::detail
