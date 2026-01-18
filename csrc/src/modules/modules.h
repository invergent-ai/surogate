// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//

#ifndef SUROGATE_SRC_MODULES_MODULES_H
#define SUROGATE_SRC_MODULES_MODULES_H

/**
 * @file modules.h
 * @brief Convenience header including all module definitions
 *
 * This header provides a single include point for all primitive and composite
 * modules in the SUROGATE module system.
 *
 * Usage:
 * @code
 * #include "modules/modules.h"
 *
 * using namespace modules;
 *
 * LinearModule linear({.in_features = 768, .out_features = 3072});
 * RMSNormModule norm({.hidden_size = 768, .epsilon = 1e-5f});
 * AttentionModule attn({.hidden_size = 768, .num_query_heads = 12, ...});
 * @endcode
 */

// Core module infrastructure
#include "module_concept.h"
#include "module_base.h"

// Primitive modules
#include "primitives/rmsnorm.h"
#include "primitives/swiglu.h"
#include "primitives/embedding.h"
#include "primitives/attention.h"
#include "primitives/mlp.h"

// Composite modules
#include "composite/transformer_block.h"
// #include "composite/moe_block.h"  // TODO: implement

// Backward hooks for LoRA integration
#include "backward_hooks.h"

// State management
#include "weight_schema.h"
#include "weight_manager.h"
#include "gradient_manager.h"
#include "optimizer_state.h"
#include "run_state.h"

// Model configuration and factory
#include "model_config.h"
#include "model/modular_model.h"
#include "model_factory.h"

// LoRA adapter support
#include "lora/lora_config.h"
#include "lora/lora_weights.h"
#include "lora/lora_model.h"

// MoE modules
#include "moe/router.h"
#include "moe/expert.h"
#include "moe/moe_block.h"

#endif // SUROGATE_SRC_MODULES_MODULES_H
