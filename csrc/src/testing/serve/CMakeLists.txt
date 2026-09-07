find_package(Python3 REQUIRED COMPONENTS Interpreter)

# This target deliberately receives no src/, CUDA, artifact, kernel, or target
# include root. It proves that the installed product headers stand alone.
add_executable(sinfer_public_api_test test_public_api.cpp)
target_include_directories(sinfer_public_api_test PRIVATE ${SERVE_ROOT})
add_test(NAME sinfer_public_api_test COMMAND sinfer_public_api_test)

function(sinfer_add_test name)
  cmake_parse_arguments(arg "NEEDS_SOURCE_DIR" "" "SOURCES;LIBRARIES" ${ARGN})
  add_executable(${name} ${arg_SOURCES})
  target_include_directories(${name} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${SERVE_ROOT}
    ${SINFER_JSON_INCLUDE}
    ${SUROGATE_VENDOR_DIR}
    ${SUROGATE_VENDOR_DIR}/cpp-httplib
    ${SUROGATE_VENDOR_DIR}/utf8proc
    # Serve headers reach cuda_runtime.h on their own now -- a package header
    # pulls in the KV pool, which pulls in the demand-mapped region. A test that
    # links sinfer_engine rather than sinfer_core does not inherit the toolkit's
    # include path, so give every test target its own.
    ${CUDAToolkit_INCLUDE_DIRS})
  if(arg_LIBRARIES)
    target_link_libraries(${name} PRIVATE ${arg_LIBRARIES})
  else()
    target_link_libraries(${name} PRIVATE sinfer_core)
  endif()
  if(arg_NEEDS_SOURCE_DIR)
    target_compile_definitions(${name} PRIVATE
      SINFER_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}"
      SINFER_PYTHON_EXECUTABLE="${Python3_EXECUTABLE}")
  endif()
  add_test(NAME ${name} COMMAND ${name})
  # Every test joins one aggregate target. Without it nothing builds these: they are
  # excluded from `all`, so `cmake --build` skips them and ctest then reports "Not Run"
  # for a binary that was never compiled -- which is how six of them came to be broken
  # against renamed symbols without anyone noticing.
  set_property(GLOBAL APPEND PROPERTY SINFER_SERVE_TEST_TARGETS ${name})
  # 77 is this suite's skip convention: a test whose fixture is absent (real
  # weights, tokenizer resources) says so and exits 77. Without this property
  # ctest reads that as a failure, which is how the frontend test came to be
  # reported failing while a serial loop that treated 77 as a skip called it
  # green -- and the test only actually runs under ctest, which supplies the
  # source directory it needs.
  set_tests_properties(${name} PROPERTIES SKIP_RETURN_CODE 77)
endfunction()

function(sinfer_add_op_test name)
  sinfer_add_test(${name} ${ARGN})
  set_tests_properties(${name} PROPERTIES SKIP_RETURN_CODE 77)
  if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(${name} PRIVATE
      $<$<COMPILE_LANGUAGE:CXX>:-fno-fast-math>
      $<$<COMPILE_LANGUAGE:CXX>:-ffp-contract=off>)
  endif()
endfunction()

function(sinfer_add_linear_test name source)
  sinfer_add_op_test(${name}
    SOURCES ${source}
            ops/linear/linear_test_common.cpp
    LIBRARIES sinfer_ops)
endfunction()

function(sinfer_add_fused_linear_test name source common)
  sinfer_add_op_test(${name}
    SOURCES ${source} ${common}
    LIBRARIES sinfer_ops)
endfunction()

# Artifact framing and model-independent runtime contracts.
sinfer_add_test(sinfer_artifact_reader_test
  SOURCES test_sinfer_artifact_reader.cpp
  LIBRARIES sinfer_artifact)
sinfer_add_test(sinfer_artifact_materialization_test
  SOURCES test_artifact_materialization.cpp
  LIBRARIES sinfer_artifact)
# surogate vendor patch (csrc/src/serve/PATCHES.md): the media-decode test
# exercises real FFmpeg decoding and cannot pass against the stub.
if(SINFER_ENABLE_FFMPEG)
  sinfer_add_test(sinfer_media_decode_test
    SOURCES test_media_decode.cpp
    LIBRARIES sinfer_media_decode)
endif()
sinfer_add_test(sinfer_device_test       SOURCES test_device.cpp)
sinfer_add_test(sinfer_decode_graph_test SOURCES test_decode_graph.cpp)
sinfer_add_test(sinfer_tensor_test       SOURCES test_tensor.cpp)
sinfer_add_test(sinfer_arena_test        SOURCES test_arena.cpp)
sinfer_add_test(sinfer_request_memory_test SOURCES test_request_memory.cpp)
sinfer_add_test(sinfer_admission_policy_test SOURCES test_admission_policy.cpp)
sinfer_add_test(sinfer_kv_capacity_test SOURCES test_kv_capacity.cpp)
sinfer_add_test(sinfer_cuda_visibility_test
  SOURCES test_cuda_visibility.cpp
  LIBRARIES sinfer_product_cuda_visibility)
sinfer_add_test(sinfer_sampling_defaults_test
  SOURCES test_sampling_defaults.cpp
  LIBRARIES sinfer_engine)
sinfer_add_test(sinfer_kv_cache_test
  SOURCES test_kv_cache.cpp
  LIBRARIES sinfer_core)
sinfer_add_test(sinfer_elastic_kv_region_test
  SOURCES test_elastic_kv_region.cpp
  LIBRARIES sinfer_core)
sinfer_add_test(sinfer_prefill_graph_reach_test
  SOURCES test_prefill_graph_reach.cpp
  LIBRARIES sinfer_core)
sinfer_add_test(sinfer_state_store_test
  SOURCES test_state_store.cpp
  LIBRARIES sinfer_core)
sinfer_add_test(sinfer_gdn_replay_records_test
  SOURCES test_gdn_replay_records.cpp
  LIBRARIES sinfer_core)
set_tests_properties(
  sinfer_artifact_materialization_test
  sinfer_device_test
  sinfer_decode_graph_test
  sinfer_arena_test
  sinfer_request_memory_test
  sinfer_kv_cache_test
  sinfer_state_store_test
  PROPERTIES SKIP_RETURN_CODE 77)
# Family frontend and one opt-in real-artifact prefix/MTP integration test.
sinfer_add_test(sinfer_family_frontend_test
  SOURCES family/test_frontend.cpp
  NEEDS_SOURCE_DIR
  LIBRARIES sinfer_engine sinfer_core)
target_include_directories(sinfer_family_frontend_test PRIVATE
  ${SERVE_ROOT}/family/export
  ${SERVE_ROOT}/family/impl)
# The tokenizer_config.json chat-template contract: three JSON strings and no
# tokenizer, so it runs where sinfer_family_frontend_test skips.
sinfer_add_test(sinfer_family_pretokenizer_digits_test
  SOURCES family/test_pretokenizer_digits.cpp
  LIBRARIES sinfer_engine sinfer_core)
target_include_directories(sinfer_family_pretokenizer_digits_test PRIVATE
  ${SERVE_ROOT}/family/export
  ${SERVE_ROOT}/family/impl)

sinfer_add_test(sinfer_family_tokenizer_config_test
  SOURCES family/test_tokenizer_config.cpp
  LIBRARIES sinfer_engine sinfer_core)
target_include_directories(sinfer_family_tokenizer_config_test PRIVATE
  ${SERVE_ROOT}/family/export
  ${SERVE_ROOT}/family/impl)
sinfer_add_test(sinfer_family_runtime_mechanisms_test
  SOURCES family/test_runtime_mechanisms.cpp
  LIBRARIES sinfer_engine sinfer_core)
sinfer_add_test(sinfer_qwen3_5_prefix_real_test
  SOURCES targets/qwen3_5/test_engine_prefix_real.cpp
  LIBRARIES sinfer_engine)
set_tests_properties(sinfer_qwen3_5_prefix_real_test PROPERTIES SKIP_RETURN_CODE 77)
# surogate: 4B prefix/rewrite-checkpoint exercise, run in both checkpoint
# modes — the "defer" registration is the gate for defaulting
# SUROGATE_SERVE_DEFER_REWRITE_CHECKPOINT on (PATCHES.md #24).
# surogate: byte-exact host verification of the derived FP8/NVFP4 planes
# (PATCHES.md #20/#21/#25) incl the cutlass atom SF layout.
sinfer_add_test(sinfer_lora_store_test
  SOURCES ops/test_lora_store.cpp
  LIBRARIES sinfer_ops)
set_tests_properties(sinfer_lora_store_test PROPERTIES SKIP_RETURN_CODE 77)
sinfer_add_test(sinfer_w8_quant_planes_test
  SOURCES ops/test_w8_quant_planes.cpp
  LIBRARIES sinfer_ops)
set_tests_properties(sinfer_w8_quant_planes_test PROPERTIES SKIP_RETURN_CODE 77)
# Pure host: the route selector takes n, k and T and returns a launcher, so this needs no
# device and catches an alignment regression before a checkpoint does.
sinfer_add_test(sinfer_w8_route_selection_test
  SOURCES ops/test_w8_route_selection.cpp
  LIBRARIES sinfer_ops)
sinfer_add_test(sinfer_qwen3_5_4b_prefix_real_test
  SOURCES targets/qwen3_5/test_engine_prefix_real_4b.cpp
  LIBRARIES sinfer_engine)
set_tests_properties(sinfer_qwen3_5_4b_prefix_real_test PROPERTIES SKIP_RETURN_CODE 77)
add_test(NAME sinfer_qwen3_5_4b_prefix_defer_test
  COMMAND sinfer_qwen3_5_4b_prefix_real_test defer)
set_tests_properties(sinfer_qwen3_5_4b_prefix_defer_test PROPERTIES SKIP_RETURN_CODE 77)
sinfer_add_test(sinfer_qwen3_5_load_plan_test
  SOURCES targets/qwen3_5/test_load_plan.cpp
  NEEDS_SOURCE_DIR
  LIBRARIES sinfer_engine)
target_include_directories(sinfer_qwen3_5_load_plan_test PRIVATE
  ${SERVE_ROOT}/family/export
  ${SERVE_ROOT}/targets/qwen3_5/export)
target_link_libraries(sinfer_qwen3_5_load_plan_test PRIVATE CUDA::cudart)
set_tests_properties(sinfer_qwen3_5_load_plan_test PROPERTIES SKIP_RETURN_CODE 77)
sinfer_add_test(sinfer_qwen3_5_moe_real_test
  SOURCES targets/qwen3_5_moe/test_engine_real.cpp
  LIBRARIES sinfer_engine)
set_tests_properties(sinfer_qwen3_5_moe_real_test PROPERTIES SKIP_RETURN_CODE 77)
sinfer_add_test(sinfer_qwen3_5_moe_dflash_real_test
  SOURCES targets/qwen3_5_moe/test_engine_dflash_real.cpp
  LIBRARIES sinfer_engine)
set_tests_properties(sinfer_qwen3_5_moe_dflash_real_test PROPERTIES SKIP_RETURN_CODE 77)
sinfer_add_test(sinfer_qwen3_5_moe_dflash_load_plan_test
  SOURCES targets/qwen3_5_moe/test_dflash_load_plan.cpp
  NEEDS_SOURCE_DIR
  LIBRARIES sinfer_engine)
target_include_directories(sinfer_qwen3_5_moe_dflash_load_plan_test PRIVATE
  ${SERVE_ROOT}/family/export
  ${SERVE_ROOT}/targets/qwen3_5_moe/export)
target_link_libraries(sinfer_qwen3_5_moe_dflash_load_plan_test PRIVATE CUDA::cudart)
set_tests_properties(sinfer_qwen3_5_moe_dflash_load_plan_test
  PROPERTIES SKIP_RETURN_CODE 77)

# Product protocol contracts.
sinfer_add_test(sinfer_openai_schema_test
  SOURCES test_openai_schema.cpp
  LIBRARIES sinfer_serve)
sinfer_add_test(sinfer_responses_schema_test
  SOURCES test_responses_schema.cpp
  LIBRARIES sinfer_serve)
sinfer_add_test(sinfer_response_store_test
  SOURCES test_response_store.cpp
  LIBRARIES sinfer_serve)
sinfer_add_test(sinfer_anthropic_schema_test
  SOURCES test_anthropic_schema.cpp
  LIBRARIES sinfer_serve)
sinfer_add_test(sinfer_tool_call_parser_test
  SOURCES test_tool_call_parser.cpp
  LIBRARIES sinfer_serve)
sinfer_add_test(sinfer_serve_options_test
  SOURCES test_serve_options.cpp
  LIBRARIES sinfer_serve)
sinfer_add_test(sinfer_output_parsers_test
  SOURCES test_output_parsers.cpp
  LIBRARIES sinfer_serve)
sinfer_add_test(sinfer_request_log_test
  SOURCES test_request_log.cpp
  LIBRARIES sinfer_serve)
sinfer_add_test(sinfer_http_error_handler_test
  SOURCES test_http_error_handler.cpp
  LIBRARIES sinfer_serve)
target_include_directories(sinfer_http_error_handler_test PRIVATE
  ${SUROGATE_VENDOR_DIR}/cpp-httplib)
sinfer_add_test(sinfer_bench_support_test
  SOURCES test_sinfer_bench_support.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/bench/targets/qwen3_5/sinfer_bench_support.cpp
  NEEDS_SOURCE_DIR
  LIBRARIES sinfer_engine)
target_include_directories(sinfer_bench_support_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/bench/targets/qwen3_5)

# The encoder CLI: one sequence in, one vector out. Not a ctest — it needs an
# artifact — but built beside the tests so it stays compiling.
add_executable(sinfer_embed_cli encoder/embed_cli.cpp)
target_include_directories(sinfer_embed_cli PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${SERVE_ROOT})
target_link_libraries(sinfer_embed_cli PRIVATE sinfer_encoder)

# The host kernels: no GPU and no artifact, so this one is a real ctest.
sinfer_add_op_test(sinfer_cpu_ops_test
  SOURCES encoder/test_cpu_ops.cpp
  LIBRARIES sinfer_encoder)

add_executable(sinfer_cpu_embed_cli encoder/cpu_embed_cli.cpp)
target_include_directories(sinfer_cpu_embed_cli PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${SERVE_ROOT})
target_link_libraries(sinfer_cpu_embed_cli PRIVATE sinfer_encoder)

# The /v1/embeddings server moved to the product build as `surogate-embed`
# (csrc/CMakeLists.txt): `surogate serve --embed` execs it, so it is not a test
# fixture. Building it here too would compile the same source twice.

# Shared Op numerical tests. Target-composed schedule tests are registered separately below.
set(sinfer_op_tests
  add_bias
  gelu
  gelu_mul
  silu_mul
  head_linear
  mean_pool
  encoder_attention
  residual_add
  sigmoid_mul
  rmsnorm
  gated_rmsnorm
  expert_slot_cache
  lora
  qsa_indexer
  cpu_expert_compute
  l2norm
  gated_delta_net
  causal_conv1d_silu
  layer_norm
  logit_softcap
  embedding
  argmax
  gdn_gating
  gdn_gating_proj
  rope
  vision_pos_embed
  vision_attention
  sampling
  scalar
  cast
  scatter
  short_conv
  position)
foreach(op IN LISTS sinfer_op_tests)
  sinfer_add_op_test(sinfer_${op}_test
    SOURCES ops/test_${op}.cpp
    LIBRARIES sinfer_ops)
endforeach()

# Stateful, fused, and exact-shape Ops remain central even when the current implementation domain
# is one checkpoint/device.
sinfer_add_op_test(sinfer_gqa_attention_test
  SOURCES ops/test_gqa_attention.cpp
  LIBRARIES sinfer_ops)
sinfer_add_op_test(sinfer_bidirectional_gqa_attention_test
  SOURCES ops/test_bidirectional_gqa_attention.cpp
  LIBRARIES sinfer_ops)
sinfer_add_op_test(sinfer_swa_test
  SOURCES ops/test_swa.cpp
  LIBRARIES sinfer_ops)
sinfer_add_op_test(sinfer_kv_cache_append_prefix_test
  SOURCES ops/test_kv_cache_append_prefix.cpp
  LIBRARIES sinfer_ops)
sinfer_add_op_test(sinfer_prepare_masked_block_test
  SOURCES ops/test_prepare_masked_block.cpp
  LIBRARIES sinfer_ops)
sinfer_add_op_test(sinfer_kimi_delta_net_test
  SOURCES ops/test_kimi_delta_net.cpp
  LIBRARIES sinfer_ops)
sinfer_add_op_test(sinfer_manifold_hyper_connection_test
  SOURCES ops/test_manifold_hyper_connection.cpp
  LIBRARIES sinfer_ops)
sinfer_add_op_test(sinfer_sparse_moe_test
  SOURCES ops/test_sparse_moe.cpp
  LIBRARIES sinfer_ops)
sinfer_add_op_test(sinfer_mtp_pack_test
  SOURCES ops/test_mtp_pack.cpp
  LIBRARIES sinfer_ops)
sinfer_add_op_test(sinfer_mtp_round_test
  SOURCES ops/test_mtp_round.cpp
  LIBRARIES sinfer_ops)
sinfer_add_op_test(sinfer_speculative_round_test
  SOURCES ops/test_speculative_round.cpp
  LIBRARIES sinfer_ops)
sinfer_add_linear_test(sinfer_linear_q4_a16_test ops/linear/test_q4_a16.cpp)
sinfer_add_linear_test(sinfer_linear_ggml_k_test ops/linear/ggml/test_ggml_k.cpp)
sinfer_add_linear_test(sinfer_linear_q5_a16_test ops/linear/test_q5_a16.cpp)
sinfer_add_linear_test(sinfer_linear_q6_a16_test ops/linear/test_q6_a16.cpp)
sinfer_add_linear_test(sinfer_linear_w8_a16_test ops/linear/test_w8_a16.cpp)
sinfer_add_linear_test(sinfer_linear_nvfp4_a16_test ops/linear/test_nvfp4_a16.cpp)
sinfer_add_linear_test(sinfer_linear_nvfp4_a4_test ops/linear/test_nvfp4_a4.cpp)
sinfer_add_op_test(sinfer_linear_nvfp4_cublaslt_test SOURCES ops/linear/test_nvfp4_cublaslt.cpp  LIBRARIES sinfer_ops CUDA::cublasLt)
sinfer_add_linear_test(sinfer_linear_fp8_a16_test ops/linear/test_fp8_a16.cpp)
sinfer_add_linear_test(sinfer_linear_fp8_a8_test ops/linear/test_fp8_a8.cpp)
sinfer_add_linear_test(sinfer_linear_fp8_generic_test ops/linear/test_fp8_generic.cpp)
sinfer_add_op_test(sinfer_linear_bf16_a16_test
  SOURCES ops/linear/test_bf16_a16.cpp
  LIBRARIES sinfer_ops)
sinfer_add_op_test(sinfer_attn_input_proj_test
  SOURCES ops/test_attn_input_proj.cpp
  LIBRARIES sinfer_ops)
sinfer_add_op_test(sinfer_gdn_input_proj_test
  SOURCES ops/test_gdn_input_proj.cpp
  LIBRARIES sinfer_ops)
sinfer_add_op_test(sinfer_gdn_input_proj_conv_snapshot_test
  SOURCES ops/test_gdn_input_proj_conv_snapshot.cpp
  LIBRARIES sinfer_ops)
sinfer_add_op_test(sinfer_gdn_input_proj_conv_record_test
  SOURCES ops/test_gdn_input_proj_conv_record.cpp
  LIBRARIES sinfer_ops)
sinfer_add_op_test(sinfer_gated_delta_net_replay_record_test
  SOURCES ops/test_gated_delta_net_replay_record.cpp
  LIBRARIES sinfer_ops)
sinfer_add_op_test(sinfer_gdn_replay_fold_test
  SOURCES ops/test_gdn_replay_fold.cpp
  LIBRARIES sinfer_ops)
sinfer_add_op_test(sinfer_kimi_delta_net_replay_test
  SOURCES ops/test_kimi_delta_net_replay.cpp
  LIBRARIES sinfer_ops)
sinfer_add_fused_linear_test(
  sinfer_linear_add_q5_a16_test
  ops/linear_add/test_q5_a16.cpp
  ops/linear_add/linear_add_test_common.cpp)
sinfer_add_fused_linear_test(
  sinfer_linear_add_bf16_a16_test
  ops/linear_add/test_bf16_a16.cpp
  ops/linear_add/linear_add_test_common.cpp)
sinfer_add_fused_linear_test(
  sinfer_linear_add_w8_a16_test
  ops/linear_add/test_w8_a16.cpp
  ops/linear_add/linear_add_test_common.cpp)
sinfer_add_op_test(sinfer_linear_add_nvfp4_test
  SOURCES ops/linear_add/test_nvfp4.cpp
  LIBRARIES sinfer_ops)
sinfer_add_op_test(sinfer_linear_add_fp8_test
  SOURCES ops/linear_add/test_fp8.cpp
  LIBRARIES sinfer_ops)
sinfer_add_op_test(sinfer_linear_fp8_block_test
  SOURCES ops/linear/fp8_block/test_fp8_block.cpp
  LIBRARIES sinfer_ops)
sinfer_add_fused_linear_test(
  sinfer_linear_pair_w8_a16_test
  ops/linear_pair/test_w8_a16.cpp
  ops/linear_pair/linear_pair_test_common.cpp)
sinfer_add_fused_linear_test(
  sinfer_linear_swiglu_q4_a16_test
  ops/linear_swiglu/test_q4_a16.cpp
  ops/linear_swiglu/linear_swiglu_test_common.cpp)
sinfer_add_fused_linear_test(
  sinfer_linear_swiglu_w8_a16_test
  ops/linear_swiglu/test_w8_a16.cpp
  ops/linear_swiglu/linear_swiglu_test_common.cpp)
sinfer_add_fused_linear_test(
  sinfer_linear_swiglu_nvfp4_test
  ops/linear_swiglu/test_nvfp4.cpp
  ops/linear_swiglu/linear_swiglu_test_common.cpp)
sinfer_add_fused_linear_test(
  sinfer_linear_swiglu_fp8_test
  ops/linear_swiglu/test_fp8.cpp
  ops/linear_swiglu/linear_swiglu_test_common.cpp)

sinfer_add_test(sinfer_qwen3_5_visual_scatter_test
  SOURCES targets/qwen3_5/test_visual_scatter.cpp
  LIBRARIES sinfer_engine sinfer_core)
set_tests_properties(sinfer_qwen3_5_visual_scatter_test PROPERTIES SKIP_RETURN_CODE 77)

# These tests deliberately compile against the exact target's internal contracts; the target does
# not export impl/ to ordinary consumers.
set(sinfer_qwen3_5_internal_tests
  sinfer_qwen3_5_visual_scatter_test)
foreach(test_target IN LISTS sinfer_qwen3_5_internal_tests)
  target_include_directories(${test_target} PRIVATE
    ${SERVE_ROOT}/targets/qwen3_5/impl
    ${SERVE_ROOT}/targets/qwen3_5/export
    ${SERVE_ROOT}/family/export)
endforeach()


# The aggregate: `cmake --build <dir> --target serve-tests` builds every registered test,
# and `ctest` then has a binary for each of them.
get_property(_sinfer_serve_test_targets GLOBAL PROPERTY SINFER_SERVE_TEST_TARGETS)
if(_sinfer_serve_test_targets)
  add_custom_target(serve-tests DEPENDS ${_sinfer_serve_test_targets})
  list(LENGTH _sinfer_serve_test_targets _sinfer_serve_test_count)
  message(STATUS "serve tests: ${_sinfer_serve_test_count} targets under `serve-tests`")
endif()
