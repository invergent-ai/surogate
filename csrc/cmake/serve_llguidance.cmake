include(FetchContent)
find_program(SUROGATE_CARGO_EXECUTABLE cargo REQUIRED)
FetchContent_Declare(surogate_llguidance
  GIT_REPOSITORY https://github.com/guidance-ai/llguidance.git
  GIT_TAG dbaf504d498b6aeede06ae57adc6f7c2c4848c59 # v1.8.0
  SOURCE_SUBDIR surogate-no-upstream-build)
FetchContent_MakeAvailable(surogate_llguidance)
set(llguidance_build "${CMAKE_CURRENT_BINARY_DIR}/llguidance")
set(llguidance_archive "${llguidance_build}/release/libllguidance.a")
add_custom_command(OUTPUT "${llguidance_archive}"
  COMMAND "${SUROGATE_CARGO_EXECUTABLE}" build --locked --release -p llguidance
          --manifest-path "${surogate_llguidance_SOURCE_DIR}/Cargo.toml"
          --target-dir "${llguidance_build}"
  DEPENDS "${surogate_llguidance_SOURCE_DIR}/Cargo.lock"
  WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
  COMMENT "Building pinned llguidance JSON Schema backend" VERBATIM)
add_custom_target(surogate_llguidance_build DEPENDS "${llguidance_archive}")
add_library(surogate_llguidance STATIC IMPORTED GLOBAL)
set_target_properties(surogate_llguidance PROPERTIES
  IMPORTED_LOCATION "${llguidance_archive}"
  INTERFACE_INCLUDE_DIRECTORIES "${surogate_llguidance_SOURCE_DIR}/parser")
add_dependencies(surogate_llguidance surogate_llguidance_build)
target_link_libraries(surogate_llguidance INTERFACE Threads::Threads ${CMAKE_DL_LIBS} m)
install(FILES "${surogate_llguidance_SOURCE_DIR}/LICENSE"
  DESTINATION surogate/licenses/llguidance COMPONENT serve)
