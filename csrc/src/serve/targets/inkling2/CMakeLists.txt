# Inkling-2-Mini (`inkling2_loop`): the looped target (weight bindings in impl/load).
target_sources(sinfer_engine PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/impl/package.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/impl/model.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/impl/forward.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/impl/load/bindings.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/impl/variant.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/impl/dspark.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/impl/decision/adapter_file.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/impl/decision/heads.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/impl/decision/registry.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/impl/decision/runner.cpp)

target_include_directories(sinfer_engine PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/impl
  ${SERVE_ROOT})

# NAIVE device kernels (replaced by the attention / MoE ops): compiled with the ops, -fmad=false as inkling_mix.
target_sources(sinfer_ops PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/impl/naive/naive_attention.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/impl/naive/naive_moe.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/impl/decision/kernels.cu)
set_source_files_properties(
  ${CMAKE_CURRENT_SOURCE_DIR}/impl/naive/naive_attention.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/impl/naive/naive_moe.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/impl/decision/kernels.cu
  TARGET_DIRECTORY sinfer_ops
  PROPERTIES COMPILE_OPTIONS "$<$<COMPILE_LANGUAGE:CUDA>:-fmad=false>")

# The Inkling attention ops (an archive of their own): the target calls them.
target_link_libraries(sinfer_engine PRIVATE sinfer_inkling_attention)
