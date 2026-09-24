# Native CPU TTS uses the pinned GGML model runtime; no Torch, Python or CUDA.
cmake_minimum_required(VERSION 3.25)
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    project(SurogateTts LANGUAGES C CXX)
    set(CMAKE_CXX_STANDARD 20)
    include(FetchContent)
    FetchContent_Declare(json URL https://github.com/nlohmann/json/releases/download/v3.12.0/json.tar.xz)
    FetchContent_MakeAvailable(json)
    set(SINFER_JSON_INCLUDE "${json_SOURCE_DIR}/single_include")
    get_filename_component(SUROGATE_VENDOR_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../third_party" ABSOLUTE)
    enable_testing()
endif()
if(SUROGATE_SERVE_TESTS)
    enable_testing()
endif()
find_package(Threads REQUIRED)
find_package(ICU 70 REQUIRED COMPONENTS uc i18n)
add_library(sinfer_tts_frontend STATIC EXCLUDE_FROM_ALL frontend.cpp)
target_link_libraries(sinfer_tts_frontend PUBLIC ICU::uc ICU::i18n)
add_executable(surogate-tts EXCLUDE_FROM_ALL server.cpp runtime.cpp)
target_include_directories(surogate-tts PRIVATE ${SINFER_JSON_INCLUDE} ${SUROGATE_VENDOR_DIR}/cpp-httplib)
target_link_libraries(surogate-tts PRIVATE sinfer_tts_frontend Threads::Threads)
set_target_properties(surogate-tts PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
add_executable(test_tts_frontend EXCLUDE_FROM_ALL frontend_check.cpp)
target_include_directories(test_tts_frontend PRIVATE ${SINFER_JSON_INCLUDE})
target_link_libraries(test_tts_frontend PRIVATE sinfer_tts_frontend)
set_target_properties(test_tts_frontend PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
if(SUROGATE_SERVE_TESTS)
    # The Python benchmark test feeds frozen requests to this native driver.
    set_property(GLOBAL APPEND PROPERTY SINFER_SERVE_TEST_TARGETS test_tts_frontend)
endif()
if(SUROGATE_SERVE_TESTS OR CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    # The billed character count (X-Usage-Characters) and the 4096-character limit agree. Built
    # by the standalone TTS build (make serve-tts-build) as well as by the serve test aggregate.
    add_executable(test_tts_input_characters EXCLUDE_FROM_ALL ../../testing/serve/test_tts_input_characters.cpp)
    target_include_directories(test_tts_input_characters PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    target_link_libraries(test_tts_input_characters PRIVATE sinfer_tts_frontend)
    target_compile_options(test_tts_input_characters PRIVATE -UNDEBUG)
    set_target_properties(test_tts_input_characters PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
    if(SUROGATE_SERVE_TESTS)
        set_property(GLOBAL APPEND PROPERTY SINFER_SERVE_TEST_TARGETS test_tts_input_characters)
    endif()
    add_test(NAME tts_input_characters COMMAND test_tts_input_characters)
endif()
# Resolve the model runtime inside the worker after verified assets are present.
# The adapter's C++ symbols are intentionally supplied by that pinned library.
add_library(sinfer_tts_bridge SHARED EXCLUDE_FROM_ALL native_bridge.cpp)
set_target_properties(sinfer_tts_bridge PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
add_executable(surogate-tts-worker EXCLUDE_FROM_ALL native_worker.cpp)
target_link_libraries(surogate-tts-worker PRIVATE ${CMAKE_DL_LIBS})
target_include_directories(surogate-tts-worker PRIVATE ${SINFER_JSON_INCLUDE})
set_target_properties(surogate-tts-worker PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
add_dependencies(surogate-tts-worker sinfer_tts_bridge)
add_dependencies(surogate-tts surogate-tts-worker)

option(SUROGATE_TTS_CPU_KERNELS "Build validated AVX-512 TTS matrix kernels" ON)
if(SUROGATE_TTS_CPU_KERNELS AND CMAKE_SYSTEM_NAME STREQUAL "Linux" AND CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64)$")
    # Isolate this pinned model backend from the GGML used by the quantizer.
    include(ExternalProject)
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/patches/ggml-cpu.patch")
    file(SHA256 "${CMAKE_CURRENT_SOURCE_DIR}/patches/ggml-cpu.patch" patch_hash)
    string(SUBSTRING "${patch_hash}" 0 12 patch_key)
    set(kernel_library "${CMAKE_BINARY_DIR}/tts-kernels/libggml-cpu.so.0")
    ExternalProject_Add(tts_cpu_kernels
        PREFIX "${CMAKE_BINARY_DIR}/tts-kernels-build-${patch_key}"
        URL https://codeload.github.com/ggml-org/ggml/tar.gz/c03b4e2bcece5134827881af90242086daf75be5
        URL_HASH SHA256=7f1fe40bdd9d75bc675dab06c0436308a69556b71125bd1c9c668b26a641d94c
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE
        PATCH_COMMAND "${CMAKE_COMMAND}"
            -DSOURCE_DIR=<SOURCE_DIR>
            -DPATCH_FILE=${CMAKE_CURRENT_SOURCE_DIR}/patches/ggml-cpu.patch
            -P ${CMAKE_CURRENT_SOURCE_DIR}/patches/apply.cmake
        CMAKE_ARGS
            -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON
            -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER} -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
            -DGGML_NATIVE=OFF -DGGML_AVX=ON -DGGML_AVX2=ON -DGGML_AVX512=ON
            -DGGML_F16C=ON -DGGML_FMA=ON -DGGML_LLAMAFILE=OFF
            -DGGML_CUDA=OFF -DGGML_BUILD_TESTS=OFF -DGGML_BUILD_EXAMPLES=OFF
        BUILD_COMMAND "${CMAKE_COMMAND}" --build <BINARY_DIR> --target ggml-cpu --parallel 4
        INSTALL_COMMAND "${CMAKE_COMMAND}" -E make_directory "${CMAKE_BINARY_DIR}/tts-kernels"
            COMMAND "${CMAKE_COMMAND}" -E copy_if_different
                <BINARY_DIR>/src/libggml-cpu.so.0.12.0 "${kernel_library}"
        BUILD_BYPRODUCTS "${kernel_library}" <BINARY_DIR>/src/libggml-base.so.0.12.0)
    add_dependencies(surogate-tts-worker tts_cpu_kernels)
    if(SUROGATE_SERVE_TESTS OR CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
        ExternalProject_Get_Property(tts_cpu_kernels SOURCE_DIR BINARY_DIR)
        add_executable(test_tts_cpu_kernels EXCLUDE_FROM_ALL ../../testing/serve/test_tts_cpu_kernels.cpp)
        target_include_directories(test_tts_cpu_kernels PRIVATE "${SOURCE_DIR}/include")
        target_link_libraries(test_tts_cpu_kernels PRIVATE "${kernel_library}" "${BINARY_DIR}/src/libggml-base.so.0.12.0")
        add_dependencies(test_tts_cpu_kernels tts_cpu_kernels)
        set_target_properties(test_tts_cpu_kernels PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
        set_property(GLOBAL APPEND PROPERTY SINFER_SERVE_TEST_TARGETS test_tts_cpu_kernels)
        add_test(NAME tts_cpu_kernels COMMAND test_tts_cpu_kernels)
        set_tests_properties(tts_cpu_kernels PROPERTIES SKIP_RETURN_CODE 77)
    endif()
    if(SKBUILD)
        install(FILES "${kernel_library}" DESTINATION surogate/serve/_bin/tts-kernels COMPONENT serve)
        install(FILES vendor/GGML-LICENSE DESTINATION surogate/serve/_bin/tts-kernels COMPONENT serve)
    else()
        install(FILES "${kernel_library}" DESTINATION bin/tts-kernels COMPONENT serve)
        install(FILES vendor/GGML-LICENSE DESTINATION bin/tts-kernels COMPONENT serve)
    endif()
endif()
