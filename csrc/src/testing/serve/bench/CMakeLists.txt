function(sinfer_add_op_bench name)
  cmake_parse_arguments(arg "" "" "SOURCES" ${ARGN})
  add_executable(${name} ${arg_SOURCES})
  target_include_directories(${name} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/ops
    ${SERVE_ROOT}
    ${SERVE_ROOT})
  target_link_libraries(${name} PRIVATE sinfer_shared)
  target_compile_options(${name} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>)
endfunction()

sinfer_add_op_bench(sinfer_marlin_w8_bench SOURCES ops/marlin_w8_bench.cu)
set_source_files_properties(ops/marlin_w8_bench.cu
  PROPERTIES COMPILE_OPTIONS "$<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>")
sinfer_add_op_bench(sinfer_silu_mul_bench SOURCES ops/silu_mul_bench.cu)
sinfer_add_op_bench(sinfer_residual_add_bench SOURCES ops/residual_add_bench.cu)
sinfer_add_op_bench(sinfer_sigmoid_mul_bench SOURCES ops/sigmoid_mul_bench.cu)
sinfer_add_op_bench(sinfer_rmsnorm_bench SOURCES ops/rmsnorm_bench.cu)
sinfer_add_op_bench(sinfer_l2norm_bench SOURCES ops/l2norm_bench.cu)
sinfer_add_op_bench(sinfer_gdn_gating_bench SOURCES ops/gdn_gating_bench.cu)
sinfer_add_op_bench(sinfer_gdn_gating_proj_bench SOURCES ops/gdn_gating_proj_bench.cu)
sinfer_add_op_bench(sinfer_gated_delta_net_bench SOURCES ops/gated_delta_net_bench.cu)
sinfer_add_op_bench(sinfer_gdn_replay_bench SOURCES ops/gdn_replay_bench.cu)
sinfer_add_op_bench(sinfer_gdn_layer_bench SOURCES ops/gdn_layer_bench.cu)
sinfer_add_op_bench(sinfer_rope_bench SOURCES ops/rope_bench.cu)
sinfer_add_op_bench(sinfer_add_bias_bench SOURCES ops/add_bias_bench.cu)
sinfer_add_op_bench(sinfer_gelu_bench SOURCES ops/gelu_bench.cu)
sinfer_add_op_bench(sinfer_layer_norm_bench SOURCES ops/layer_norm_bench.cu)
sinfer_add_op_bench(sinfer_vision_pos_embed_bench SOURCES ops/vision_pos_embed_bench.cu)
sinfer_add_op_bench(sinfer_vision_tower_tune_bench SOURCES ops/vision_tower_tune_bench.cu)
sinfer_add_op_bench(sinfer_scatter_bench SOURCES ops/scatter_bench.cu)
sinfer_add_op_bench(sinfer_packed_softmax_attention_bench
  SOURCES ops/packed_softmax_attention_bench.cu)
sinfer_add_op_bench(sinfer_embedding_bench SOURCES ops/embedding_bench.cu)
sinfer_add_op_bench(sinfer_position_bench SOURCES ops/position_bench.cu)
sinfer_add_op_bench(sinfer_cast_bench SOURCES ops/cast_bench.cu)
sinfer_add_op_bench(sinfer_argmax_bench SOURCES ops/argmax_bench.cu)
sinfer_add_op_bench(sinfer_causal_conv1d_silu_bench SOURCES ops/causal_conv1d_silu_bench.cu)
sinfer_add_op_bench(sinfer_linear_bench SOURCES ops/linear_bench.cu)
sinfer_add_op_bench(sinfer_output_stage_bench SOURCES ops/output_stage_bench.cu)
sinfer_add_op_bench(sinfer_gdn_input_proj_bench SOURCES ops/gdn_input_proj_bench.cu)
sinfer_add_op_bench(sinfer_gdn_input_proj_conv_snapshot_bench
  SOURCES ops/gdn_input_proj_conv_snapshot_bench.cu)
sinfer_add_op_bench(sinfer_attn_input_proj_bench SOURCES ops/attn_input_proj_bench.cu)
sinfer_add_op_bench(sinfer_w8_linear_swiglu_bench SOURCES ops/w8_linear_swiglu_bench.cu)
# surogate vendor patch (PATCHES.md #13): qwen3.5-0.8b W8 route sweep.
sinfer_add_op_bench(sinfer_q08_route_sweep_bench SOURCES ops/q08_route_sweep_bench.cu)
# surogate vendor patch (PATCHES.md #17 WIP): W8A8-int IMMA prefill probe.
sinfer_add_op_bench(sinfer_w8a8_imma_probe_bench SOURCES ops/w8a8_imma_probe_bench.cu)
sinfer_add_op_bench(sinfer_nvfp4_linear_swiglu_bench SOURCES ops/nvfp4_linear_swiglu_bench.cu)
sinfer_add_op_bench(sinfer_fp8_linear_swiglu_bench SOURCES ops/fp8_linear_swiglu_bench.cu)
sinfer_add_op_bench(sinfer_q4_linear_swiglu_bench SOURCES ops/q4_linear_swiglu_bench.cu)
sinfer_add_op_bench(sinfer_w8_linear_add_bench SOURCES ops/w8_linear_add_bench.cu)
sinfer_add_op_bench(sinfer_q5_linear_add_bench SOURCES ops/q5_linear_add_bench.cu)
sinfer_add_op_bench(sinfer_bf16_linear_add_bench SOURCES ops/bf16_linear_add_bench.cu)
sinfer_add_op_bench(sinfer_nvfp4_linear_add_bench SOURCES ops/nvfp4_linear_add_bench.cu)
sinfer_add_op_bench(sinfer_fp8_linear_add_bench SOURCES ops/fp8_linear_add_bench.cu)
sinfer_add_op_bench(sinfer_causal_softmax_attention_bench
  SOURCES ops/causal_softmax_attention_bench.cu)
sinfer_add_op_bench(sinfer_context_softmax_attention_bench
  SOURCES ops/context_softmax_attention_bench.cu)
sinfer_add_op_bench(sinfer_sliding_window_attention_bench
  SOURCES ops/sliding_window_attention_bench.cu)
sinfer_add_op_bench(sinfer_kv_cache_append_bench SOURCES ops/kv_cache_append_bench.cu)
sinfer_add_op_bench(sinfer_prepare_masked_block_bench
  SOURCES ops/prepare_masked_block_bench.cu)
sinfer_add_op_bench(sinfer_sampling_select_bench SOURCES ops/sampling_select_bench.cu)
sinfer_add_op_bench(sinfer_sparse_moe_bench SOURCES ops/sparse_moe_bench.cu)
sinfer_add_op_bench(sinfer_linear_pair_bench SOURCES ops/linear_pair_bench.cu)
sinfer_add_op_bench(sinfer_mtp_pack_bench SOURCES ops/mtp_pack_bench.cu)
sinfer_add_op_bench(sinfer_proposal_remap_bench SOURCES ops/proposal_remap_bench.cu)
sinfer_add_op_bench(sinfer_cpu_expert_compute_bench SOURCES ops/cpu_expert_compute_bench.cpp)

# End-to-end product benchmark: public Engine API and native .sinfer artifacts only.
add_executable(sinfer_bench targets/qwen3_5/sinfer_bench.cpp targets/qwen3_5/sinfer_bench_support.cpp)
target_include_directories(sinfer_bench PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${SERVE_ROOT}
  ${SERVE_ROOT}
  ${SUROGATE_VENDOR_DIR})
target_compile_definitions(sinfer_bench PRIVATE SINFER_SOURCE_DIR="${SERVE_ROOT}")
target_link_libraries(sinfer_bench PRIVATE sinfer_shared CUDA::cudart)

# Native target-private MTP proposal/verification microbenchmark. It enters through the target
# Program facade and executes the production round schedule over a real .sinfer artifact.
add_executable(sinfer_qwen3_5_mtp_round_bench
  targets/qwen3_5/target_mtp_round_bench.cpp)
target_include_directories(sinfer_qwen3_5_mtp_round_bench PRIVATE
  ${SERVE_ROOT}
  ${SERVE_ROOT}
  ${SUROGATE_VENDOR_DIR}
  ${SERVE_ROOT}/targets/qwen3_5/export
  ${SERVE_ROOT}/targets/qwen3_5/export)
target_link_libraries(sinfer_qwen3_5_mtp_round_bench PRIVATE sinfer_shared)

# Complete production DFlash steady round: confirmed-context append, next proposal, target
# verification/acceptance, and host publication over a real 35B artifact.
add_executable(sinfer_qwen3_5_moe_dflash_round_bench
  targets/qwen3_5_moe/dflash_round_bench.cpp)
target_include_directories(sinfer_qwen3_5_moe_dflash_round_bench PRIVATE
  ${SERVE_ROOT}
  ${SERVE_ROOT}
  ${SUROGATE_VENDOR_DIR}
  ${SERVE_ROOT}/targets/qwen3_5_moe/export
  ${SERVE_ROOT}/targets/qwen3_5/export)
target_link_libraries(sinfer_qwen3_5_moe_dflash_round_bench PRIVATE sinfer_shared)
