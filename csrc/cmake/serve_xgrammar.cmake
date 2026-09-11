# Build the pinned C++ grammar library with this project's compiler and ABI.
# The Python wheel's static library may use an incompatible GCC LTO version.
include(FetchContent)
FetchContent_Declare(surogate_xgrammar
  GIT_REPOSITORY https://github.com/mlc-ai/xgrammar.git
  GIT_TAG e6144163e6e87cf8b034030881dbad1c3ae12e1a # v0.2.3
  GIT_SUBMODULES 3rdparty/dlpack
  SOURCE_SUBDIR surogate-no-upstream-build)
FetchContent_MakeAvailable(surogate_xgrammar)
file(GLOB_RECURSE SUROGATE_XGRAMMAR_SOURCES CONFIGURE_DEPENDS
  "${surogate_xgrammar_SOURCE_DIR}/cpp/*.cc")
list(FILTER SUROGATE_XGRAMMAR_SOURCES EXCLUDE REGEX "/tvm_ffi/")
add_library(surogate_xgrammar STATIC EXCLUDE_FROM_ALL ${SUROGATE_XGRAMMAR_SOURCES})
set_target_properties(surogate_xgrammar PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(surogate_xgrammar SYSTEM PUBLIC
  "${surogate_xgrammar_SOURCE_DIR}/include"
  "${surogate_xgrammar_SOURCE_DIR}/3rdparty/dlpack/include"
  PRIVATE "${surogate_xgrammar_SOURCE_DIR}/3rdparty/picojson")
target_compile_definitions(surogate_xgrammar PRIVATE
  XGRAMMAR_ENABLE_CPPTRACE=0 XGRAMMAR_ENABLE_INTERNAL_CHECK=0)
target_link_libraries(surogate_xgrammar PUBLIC Threads::Threads)

install(FILES "${surogate_xgrammar_SOURCE_DIR}/LICENSE"
  DESTINATION surogate/licenses/xgrammar COMPONENT serve)
install(FILES "${surogate_xgrammar_SOURCE_DIR}/3rdparty/dlpack/LICENSE"
  DESTINATION surogate/licenses/dlpack COMPONENT serve)
file(READ "${surogate_xgrammar_SOURCE_DIR}/3rdparty/picojson/picojson.h" picojson_header)
string(FIND "${picojson_header}" "*/" picojson_notice_end)
math(EXPR picojson_notice_length "${picojson_notice_end} + 2")
string(SUBSTRING "${picojson_header}" 0 ${picojson_notice_length} picojson_notice)
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/picojson-LICENSE" "${picojson_notice}\n")
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/picojson-LICENSE"
  DESTINATION surogate/licenses/picojson COMPONENT serve)
