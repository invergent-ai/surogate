# Inkling-2 vision: Pillow's own decoder libraries, so that JPEG and WebP decode to Pillow's pixels exactly
# (media::decode::decode_image_pil, used only by the Inkling-2 vision processor). PNG / GIF / BMP need none: FFmpeg hands
# their samples over unchanged and decode.cpp applies Pillow's mode rules.
#
# vLLM's Inkling-2 processor decodes images with Pillow; Pillow 13.0.0 in the reference image bundles libjpeg-turbo
# 3.2.0 and decodes with the library defaults (JDCT_ISLOW, fancy upsampling, libjpeg's YCbCr tables). FFmpeg's MJPEG
# decoder + swscale differ by up to 60 levels at chroma edges (measured on the reference decoder). This builds
# the same library version as a static, position-independent archive with hidden symbols (so it cannot interpose on a
# Pillow loaded in the same process), C code only (no NASM on the build host). Verified: the 6 JPEG case images (4:2:0, 4:4:4, EXIF, no EOI, Adobe CMYK)
# decode bit for bit to Pillow's pixels (Pillow's copy runs libjpeg-turbo's SIMD paths).
include(ExternalProject)
set(SINFER_LIBJPEG_TURBO_VERSION 3.2.0)
set(_sinfer_jt_prefix ${CMAKE_BINARY_DIR}/_deps/libjpeg_turbo)
set(_sinfer_jt_install ${_sinfer_jt_prefix}/install)
ExternalProject_Add(sinfer_libjpeg_turbo_build
  URL https://github.com/libjpeg-turbo/libjpeg-turbo/releases/download/${SINFER_LIBJPEG_TURBO_VERSION}/libjpeg-turbo-${SINFER_LIBJPEG_TURBO_VERSION}.tar.gz
  URL_HASH SHA256=6f30092cef9fb839779646608f4ee14ae3cbac989c47fa05e841b0841f09878e
  PREFIX ${_sinfer_jt_prefix}
  CMAKE_ARGS
    -DCMAKE_BUILD_TYPE=Release
    -DCMAKE_INSTALL_PREFIX=${_sinfer_jt_install}
    -DCMAKE_INSTALL_LIBDIR=lib
    -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
    -DCMAKE_POSITION_INDEPENDENT_CODE=ON
    -DCMAKE_C_FLAGS=-fvisibility=hidden
    -DENABLE_SHARED=OFF
    -DENABLE_STATIC=ON
    -DWITH_TURBOJPEG=OFF
    -DWITH_TOOLS=OFF
    -DWITH_TESTS=OFF
    -DWITH_SIMD=OFF
  BUILD_BYPRODUCTS ${_sinfer_jt_install}/lib/libjpeg.a
  # Built only when something linking sinfer_media_decode is (the engine with media decoding), never by `all` alone.
  EXCLUDE_FROM_ALL TRUE
  DOWNLOAD_EXTRACT_TIMESTAMP ON)
file(MAKE_DIRECTORY ${_sinfer_jt_install}/include)
add_library(sinfer_libjpeg_turbo STATIC IMPORTED GLOBAL)
set_target_properties(sinfer_libjpeg_turbo PROPERTIES
  IMPORTED_LOCATION ${_sinfer_jt_install}/lib/libjpeg.a
  INTERFACE_INCLUDE_DIRECTORIES ${_sinfer_jt_install}/include)
add_dependencies(sinfer_libjpeg_turbo sinfer_libjpeg_turbo_build)

# libwebp 1.6.0 = Pillow 13's: Pillow decodes every WebP through WebPAnimDecoder (libwebpdemux), colour mode RGBA, the
# first frame on a transparent canvas; lossy WebP's YUV -> RGB is libwebp's fancy upsampler and tables, which FFmpeg's
# own VP8 decoder + swscale do not reproduce (up to 57 levels off on the case images). Static, PIC; the API symbols are
# built without libwebp's default-visibility attribute (WEBP_EXTERN) so the archive exports nothing from libsinfer.so.
set(SINFER_LIBWEBP_VERSION 1.6.0)
set(_sinfer_webp_prefix ${CMAKE_BINARY_DIR}/_deps/libwebp)
set(_sinfer_webp_install ${_sinfer_webp_prefix}/install)
ExternalProject_Add(sinfer_libwebp_build
  # The project's release tarball (a GitHub tag archive's bytes are not guaranteed stable).
  URL https://storage.googleapis.com/downloads.webmproject.org/releases/webp/libwebp-${SINFER_LIBWEBP_VERSION}.tar.gz
  URL_HASH SHA256=e4ab7009bf0629fd11982d4c2aa83964cf244cffba7347ecd39019a9e38c4564
  PREFIX ${_sinfer_webp_prefix}
  CMAKE_ARGS
    -DCMAKE_BUILD_TYPE=Release
    -DCMAKE_INSTALL_PREFIX=${_sinfer_webp_install}
    -DCMAKE_INSTALL_LIBDIR=lib
    -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
    -DCMAKE_POSITION_INDEPENDENT_CODE=ON
    "-DCMAKE_C_FLAGS=-fvisibility=hidden -DWEBP_EXTERN=extern"
    -DBUILD_SHARED_LIBS=OFF
    # ON for libwebp's mutex around its one-time DSP setup (decodes run on several media-cache workers); the decoder's
    # own worker threads stay off (use_threads = 0 in pil_webp.cpp).
    -DWEBP_USE_THREAD=ON
    -DWEBP_BUILD_ANIM_UTILS=OFF -DWEBP_BUILD_CWEBP=OFF -DWEBP_BUILD_DWEBP=OFF -DWEBP_BUILD_GIF2WEBP=OFF
    -DWEBP_BUILD_IMG2WEBP=OFF -DWEBP_BUILD_VWEBP=OFF -DWEBP_BUILD_WEBPINFO=OFF -DWEBP_BUILD_LIBWEBPMUX=OFF
    -DWEBP_BUILD_WEBPMUX=OFF -DWEBP_BUILD_EXTRAS=OFF
  BUILD_BYPRODUCTS ${_sinfer_webp_install}/lib/libwebpdemux.a ${_sinfer_webp_install}/lib/libwebp.a
                   ${_sinfer_webp_install}/lib/libsharpyuv.a
  EXCLUDE_FROM_ALL TRUE
  DOWNLOAD_EXTRACT_TIMESTAMP ON)
file(MAKE_DIRECTORY ${_sinfer_webp_install}/include)
add_library(sinfer_libwebp STATIC IMPORTED GLOBAL)
set_target_properties(sinfer_libwebp PROPERTIES
  IMPORTED_LOCATION ${_sinfer_webp_install}/lib/libwebp.a
  INTERFACE_INCLUDE_DIRECTORIES ${_sinfer_webp_install}/include
  INTERFACE_COMPILE_DEFINITIONS WEBP_EXTERN=extern
  INTERFACE_LINK_LIBRARIES "${_sinfer_webp_install}/lib/libsharpyuv.a;Threads::Threads")
add_library(sinfer_libwebpdemux STATIC IMPORTED GLOBAL)
set_target_properties(sinfer_libwebpdemux PROPERTIES
  IMPORTED_LOCATION ${_sinfer_webp_install}/lib/libwebpdemux.a
  INTERFACE_LINK_LIBRARIES sinfer_libwebp)
add_dependencies(sinfer_libwebp sinfer_libwebp_build)
add_dependencies(sinfer_libwebpdemux sinfer_libwebp_build)
