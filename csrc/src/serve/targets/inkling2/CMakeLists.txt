# Inkling-2-Mini (`inkling2_loop`): the looped target (weight bindings in impl/load).
target_sources(sinfer_engine PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/impl/package.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/impl/model.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/impl/forward.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/impl/load/bindings.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/impl/variant.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/impl/dspark.cpp)

target_include_directories(sinfer_engine PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/impl
  ${SERVE_ROOT})

# NAIVE device kernels (replaced by lanes c / d): compiled with the ops, -fmad=false as inkling_mix.
target_sources(sinfer_ops PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/impl/naive/naive_attention.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/impl/naive/naive_moe.cu)
set_source_files_properties(
  ${CMAKE_CURRENT_SOURCE_DIR}/impl/naive/naive_attention.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/impl/naive/naive_moe.cu
  TARGET_DIRECTORY sinfer_ops
  PROPERTIES COMPILE_OPTIONS "$<$<COMPILE_LANGUAGE:CUDA>:-fmad=false>")

# The Inkling attention ops (an archive of their own): the target calls them.
target_link_libraries(sinfer_engine PRIVATE sinfer_inkling_attention)
