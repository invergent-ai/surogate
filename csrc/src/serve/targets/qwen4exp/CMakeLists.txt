target_sources(sinfer_engine PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/impl/package.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/impl/load/bindings.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/impl/load/host_bank.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/impl/variant.cpp)

target_include_directories(sinfer_engine PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/impl
  ${SERVE_ROOT})
