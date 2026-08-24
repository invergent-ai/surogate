add_executable(ninfer
  # surogate vendor patch (csrc/src/serve/PATCHES.md): product binary names.
  cli/main.cpp
  cli/options.cpp)
target_include_directories(ninfer PRIVATE
  ${PROJECT_SOURCE_DIR}/include
  ${PROJECT_SOURCE_DIR}/src
  ${PROJECT_SOURCE_DIR}/third_party)
target_link_libraries(ninfer PRIVATE
  ninfer_engine
  ninfer_product_load_progress
  ninfer_product_prompt_input)

add_executable(ninfer-serve serve/main.cpp)
target_include_directories(ninfer-serve PRIVATE
  ${PROJECT_SOURCE_DIR}/include
  ${PROJECT_SOURCE_DIR}/src
  ${PROJECT_SOURCE_DIR}/third_party
  ${PROJECT_SOURCE_DIR}/third_party/cpp-httplib)
target_link_libraries(ninfer-serve PRIVATE ninfer_serve ninfer_product_load_progress)

# surogate vendor patch (csrc/src/serve/PATCHES.md): the product binaries carry
# surogate's name; NInfer attribution lives in NOTICE, not in process names.
set_target_properties(ninfer PROPERTIES OUTPUT_NAME surogate-engine-cli)
set_target_properties(ninfer-serve PROPERTIES OUTPUT_NAME surogate-engine)
