#pragma once

// Include this once from an exact target translation unit after defining
// SINFER_FAMILY_VARIANT and SINFER_FAMILY_RUNTIME_NS. The body is shared source; the selected
// Variant is compile-time data and the only target-dependent calls are its three closed leaves.

#include "family/impl/runtime/layouts.h"
#include "family/impl/runtime/dflash_context.h"
#include "family/impl/runtime/text_context.h"
#include "family/impl/runtime/vision_context.h"
#include "family/impl/runtime/schedule.h"
#include "family/impl/runtime/program.h"
#include "family/impl/runtime/api_impl.h"

#include "family/impl/runtime/layouts_impl.h"
#include "family/impl/runtime/dflash_context_impl.h"
#include "family/impl/runtime/text_context_impl.h"
#include "family/impl/runtime/vision_context_impl.h"
#include "family/impl/runtime/text_prefill_impl.h"
#include "family/impl/runtime/graph_impl.h"
#include "family/impl/runtime/speculative_target_impl.h"
#include "family/impl/runtime/dflash_impl.h"
#include "family/impl/runtime/decode_impl.h"
#include "family/impl/runtime/mtp_impl.h"
#include "family/impl/runtime/request_plan_impl.h"
#include "family/impl/runtime/program_impl.h"
