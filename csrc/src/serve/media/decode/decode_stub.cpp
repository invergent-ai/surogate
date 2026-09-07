// SPDX-License-Identifier: Apache-2.0
// surogate vendor patch (see csrc/src/serve/PATCHES.md): FFmpeg-free stub for
// hosts without libav* development packages. Built instead of decode.cpp when
// SINFER_ENABLE_FFMPEG=OFF. Text serving is unaffected; any image/video input
// fails with a clear error instead of the build failing at configure time.

#include "media/decode/decode.h"

namespace sinfer::media::decode {

namespace {
[[noreturn]] void unavailable() {
    throw std::runtime_error(
        "media decoding is unavailable: this binary was built without FFmpeg "
        "(SINFER_ENABLE_FFMPEG=OFF). Rebuild with the FFmpeg development "
        "libraries installed to accept image or video input.");
}
}  // namespace

Image decode_image(std::span<const std::uint8_t> /*bytes*/, const Policy& /*policy*/) {
    unavailable();
}

Video decode_video(std::span<const std::uint8_t> /*bytes*/, const Policy& /*policy*/,
                   double /*target_fps*/, int /*min_frames*/, int /*max_frames*/) {
    unavailable();
}

}  // namespace sinfer::media::decode
