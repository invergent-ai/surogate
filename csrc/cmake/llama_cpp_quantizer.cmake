# The quantiser `surogate quantize` drives, fetched and built from source.
#
# The arithmetic that turns a checkpoint into a K-quant is llama.cpp's, deliberately: every
# published GGUF was made with these encoders, and one of our own would have to match
# `make_qkx2_quants` and the per-tensor mixture in `llama-quant.cpp` closely enough not to be
# quietly worse than the file a user could have downloaded instead.
#
# It used to be a path -- the command looked for a clone at `study/llama.cpp-master` that git
# did not track and no installed copy of the package would have, so it worked on one machine
# and nowhere else. Fetching it pins the revision by content hash, keeps 460 files of someone
# else's code out of this repository, and leaves upstream's build unpatched. That last point is
# not cosmetic: a hand-pruned copy of this tree failed four different ways -- a missing
# `.h.in`, two undefined CMake helpers, and a `build-info.cpp` that will not compile without a
# git checkout to read a revision out of -- every one of them caused by the pruning rather than
# by llama.cpp.
#
# CPU only, and not as an economy: `llama-quantize` reads a GGUF and writes a GGUF. Upstream
# ships prebuilt CPU archives of it for the same reason.

include(FetchContent)

#: Pinned by commit. Update this and `LLAMA_CPP_VERSION` together, then re-run the gate in
#: `surogate/cli/quantize.py`'s docstring: a quantiser that changes its arithmetic changes
#: every artifact made after it, and the perplexity of a known export is what would notice.
set(LLAMA_CPP_COMMIT  "163a40796f0ebaae246325f8d2e15028b413fa9d" CACHE STRING
    "llama.cpp revision the quantiser is built from")
set(LLAMA_CPP_VERSION "b10797-10-g163a40796" CACHE STRING
    "llama.cpp version string, for `llama-quantize --version`")

# Every accelerator backend off, and the parts of llama.cpp we are not asking for.
foreach(_off GGML_CUDA GGML_METAL GGML_VULKAN GGML_SYCL GGML_BLAS GGML_OPENCL GGML_RPC
             GGML_HEXAGON GGML_WEBGPU GGML_OPENVINO GGML_CANN LLAMA_CURL
             LLAMA_BUILD_TESTS LLAMA_BUILD_EXAMPLES LLAMA_BUILD_SERVER LLAMA_BUILD_APP)
    set(${_off} OFF CACHE BOOL "" FORCE)
endforeach()
# Upstream gates `add_subdirectory(tools)` on both of these, and defaults them to "am I the
# top-level project", which under FetchContent we are not. So asking for the tool means asking
# for both -- this is upstream's own gate, not a workaround.
set(LLAMA_BUILD_TOOLS  ON CACHE BOOL "" FORCE)
set(LLAMA_BUILD_COMMON ON CACHE BOOL "" FORCE)
set(BUILD_SHARED_LIBS  OFF CACHE BOOL "" FORCE)

# A tarball rather than a clone: ~25 MB against a repository with its whole history, and the
# content is pinned by the commit in the URL. The cost is that an archive carries no `.git`,
# so upstream cannot read its own revision -- hence the two variables above, which is also
# what `llama-quantize --version` then reports.
set(LLAMA_BUILD_NUMBER 10797)
set(LLAMA_BUILD_COMMIT "163a4079")

FetchContent_Declare(llama_cpp
    URL https://github.com/ggml-org/llama.cpp/archive/${LLAMA_CPP_COMMIT}.tar.gz
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
FetchContent_MakeAvailable(llama_cpp)

# Where an installed package looks. `surogate quantize` needs three things at run time: the
# binary, the Hugging-Face-to-GGUF converter, and the `gguf` library that converter puts on
# `sys.path` ahead of anything installed -- the pip release lags it, and the two going out of
# step is not hypothetical, it broke on the first run with an architecture the pinned converter
# knew and the installed library did not.
set(SUROGATE_LLAMA_CPP_INSTALL_DIR "surogate/serve/_llama_cpp")

install(PROGRAMS $<TARGET_FILE:llama-quantize>
        DESTINATION ${SUROGATE_LLAMA_CPP_INSTALL_DIR}/bin
        COMPONENT quantizer)
install(FILES ${llama_cpp_SOURCE_DIR}/convert_hf_to_gguf.py
              ${llama_cpp_SOURCE_DIR}/LICENSE
        DESTINATION ${SUROGATE_LLAMA_CPP_INSTALL_DIR}
        COMPONENT quantizer)
install(DIRECTORY ${llama_cpp_SOURCE_DIR}/conversion
        DESTINATION ${SUROGATE_LLAMA_CPP_INSTALL_DIR}
        COMPONENT quantizer
        FILES_MATCHING PATTERN "*.py")
install(DIRECTORY ${llama_cpp_SOURCE_DIR}/gguf-py/gguf
        DESTINATION ${SUROGATE_LLAMA_CPP_INSTALL_DIR}/gguf-py
        COMPONENT quantizer
        FILES_MATCHING PATTERN "*.py")
