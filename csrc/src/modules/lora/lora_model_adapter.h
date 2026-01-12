// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//

#ifndef SUROGATE_SRC_MODULES_LORA_LORA_MODEL_ADAPTER_H
#define SUROGATE_SRC_MODULES_LORA_LORA_MODEL_ADAPTER_H

#include <filesystem>
#include <fstream>
#include <nlohmann/json.hpp>
#include "lora_model_core.h"
#include "lora_model_utils.h"

namespace modules {

template<typename Block>
void ModularLoRAModel<Block>::export_adapter(const std::string& directory, NCCLCommunicator& comm, const std::string& base_model_path) {
    if (!lora_enabled()) return;
    namespace fs = std::filesystem;
    fs::path dir(directory);
    if (comm.rank() == 0) { fs::create_directories(dir); }
    comm.barrier();
    mLoRAWeights->export_to_file((dir / "adapter_model.safetensors").string(), comm);
    if (comm.rank() == 0) {
        nlohmann::json adapter_config;
        adapter_config["base_model_name_or_path"] = base_model_path;
        adapter_config["peft_type"] = "LORA";
        adapter_config["task_type"] = "CAUSAL_LM";
        adapter_config["r"] = mLoRAConfig.rank;
        adapter_config["lora_alpha"] = mLoRAConfig.alpha;
        adapter_config["lora_dropout"] = mLoRAConfig.dropout;
        adapter_config["fan_in_fan_out"] = false;
        adapter_config["bias"] = "none";
        adapter_config["use_rslora"] = mLoRAConfig.use_rs_lora;
        adapter_config["target_modules"] = detail::targets_to_peft_names(mLoRAConfig);
        std::ofstream config_file(dir / "adapter_config.json");
        config_file << adapter_config.dump(2);
    }
}

template<typename Block>
void ModularLoRAModel<Block>::import_adapter(const std::string& file_name, NCCLCommunicator& comm) {
    if (!lora_enabled()) return;
    mLoRAWeights->import_from_file(file_name, comm);
}

template<typename Block>
void ModularLoRAModel<Block>::save_lora_checkpoint(const std::string& checkpoint_dir, NCCLCommunicator& comm) {
    if (!lora_enabled()) return;
    export_adapter(checkpoint_dir, comm);
}

template<typename Block>
void ModularLoRAModel<Block>::load_lora_checkpoint(const std::string& checkpoint_dir, NCCLCommunicator& comm) {
    if (!lora_enabled()) return;
    namespace fs = std::filesystem;
    fs::path adapter_file = fs::path(checkpoint_dir) / "adapter_model.safetensors";
    if (fs::exists(adapter_file)) {
        import_adapter(adapter_file.string(), comm);
    }
}

} // namespace modules

#endif // SUROGATE_SRC_MODULES_LORA_LORA_MODEL_ADAPTER_H
