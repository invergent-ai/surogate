target_sources(sinfer_engine PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/impl/load/host_decode.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/impl/load/host_bank.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/impl/moe/expert_cache.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/impl/moe/banked_experts.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/impl/frontend/chat_template.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/impl/frontend/digest.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/impl/frontend/frontend.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/impl/frontend/media_cache.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/impl/frontend/processor.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/impl/frontend/lfm2_processor.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/impl/frontend/resources.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/impl/frontend/tokenizer.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/impl/runtime/prefix_identity.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/impl/runtime/target_support.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/impl/state/decoder_state.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/impl/state/round_state.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/impl/runtime/visual_scatter.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/impl/vision/bindings.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/impl/vision/control.cpp)

target_include_directories(sinfer_engine PRIVATE
  ${SERVE_ROOT})
