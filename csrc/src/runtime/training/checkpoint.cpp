// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// Copyright (c) 2025, IST Austria, developed by Erik Schultheis
// SPDX-License-Identifier: Apache-2.0
//

#include "checkpoint.h"

#include <filesystem>

#include <nlohmann/json.hpp>
#include <fmt/core.h>

#include "dataloader.h"
#include "model.h"
#include "config/pretrained_config.h"
#include "utilities/comm.h"
#include "utilities/safetensors.h"
#include "utilities/tensor.h"

/**
 * @brief Build the full checkpoint path for a given training step.
 *
 * This appends a `step_XXXXXXXX` directory name (zero-padded to 8 digits) to the
 * provided checkpoint directory.
 *
 * @param checkpoint_directory Base directory in which checkpoints are stored.
 * @param step Training step number used to form the subdirectory name.
 * @return Full path to the checkpoint directory for @p step.
 */
std::string get_checkpoint_path(std::string checkpoint_directory, int step) {
    checkpoint_directory += fmt::format("/step_{:08}", step);
    return checkpoint_directory;
}

/**
 * @brief Save a sharded checkpoint (weights + optimizer state) for the current rank.
 *
 * Writes per-rank safetensors shards for model weights and Adam optimizer state into
 * a `step_XXXXXXXX` directory. Rank 0 additionally writes a `checkpoint.json` metadata
 * file after all ranks have finished writing their shards.
 *
 * The function synchronizes ranks before and after writing shard files to ensure
 * checkpoint consistency.
 *
 * @param target Base checkpoint directory (the step subdirectory is appended).
 * @param step Training step number used to name the checkpoint directory.
 * @param model Model instance providing weights, optimizer state, and RNG state.
 * @param loader Optional dataloader to persist/restore input iteration state; may be null.
 * @param comm NCCL communicator used for rank/world information and barriers.
 * @return The full path to the created checkpoint directory for @p step.
 *
 * @throws std::filesystem::filesystem_error If directory creation or file operations fail.
 * @throws std::exception Propagates errors thrown by safetensors writing utilities.
 */
std::string save_checkpoint(std::string target, int step, IModel& model, const DataLoader* loader, NCCLCommunicator& comm) {
    comm.barrier();

    nlohmann::json meta_data;
    const bool is_lora = model.lora_enabled();

    // Multi-node: only the root node (node_rank 0) performs filesystem I/O.
    // In data-parallel training, all nodes have identical weights, so saving from
    // one node is sufficient. Non-root nodes only participate in NCCL barriers.
    // Shards are indexed by local_rank (0..num_local_gpus-1) so the checkpoint
    // is self-contained on a single node's filesystem.
    const bool is_root_node = comm.node_rank() == 0;
    const int shard_rank = comm.local_rank();
    const int shard_count = comm.num_local_gpus();

    target = get_checkpoint_path(std::move(target), step);
    if (is_root_node) {
        std::filesystem::create_directories(target);
    }

    // For LoRA training, only save the adapter - base model weights are frozen
    if (!is_lora) {
        if (is_root_node) {
            // weights
            // TODO don't duplicate weights if they are unsharded
            write_safetensors(target + fmt::format("/weights.shard_{:03}_of_{:03}.safetensors", shard_rank, shard_count), model.weights());

            // sharded optimizer state
            write_safetensors(target + fmt::format("/adam.m.shard_{:03}_of_{:03}.safetensors", shard_rank, shard_count), model.opt_momentum());
            write_safetensors(target + fmt::format("/adam.v.shard_{:03}_of_{:03}.safetensors", shard_rank, shard_count), model.opt_variance());

            bool has_scales = false;
            model.opt_momentum_scales().iterate_tensors([&has_scales](const std::string& name, const TensorShard& tensor){
                if(tensor.Data != nullptr) {
                    has_scales = true;
                }
            });
            if(has_scales) {
                write_safetensors(target + fmt::format("/adam.m.scales.shard_{:03}_of_{:03}.safetensors", shard_rank, shard_count), model.opt_momentum_scales());
            }

            bool has_v_scales = false;
            model.opt_variance_scales().iterate_tensors([&has_v_scales](const std::string&, const TensorShard& tensor) {
                if (tensor.Data != nullptr) {
                    has_v_scales = true;
                }
            });
            if (has_v_scales) {
                write_safetensors(target + fmt::format("/adam.v.scales.shard_{:03}_of_{:03}.safetensors", shard_rank, shard_count), model.opt_variance_scales());
            }
        }

        // Export full model weights in HuggingFace-compatible format (config.json + model.safetensors)
        // This allows loading the checkpoint directly with HuggingFace transformers
        // All ranks participate in NCCL barriers inside export_weights; only rank 0 writes.
        if (is_root_node) {
            save_pretrained_config(*model.get_run_state().Config, (target + "/config.json").c_str());
        }
        model.export_weights(target + "/model.safetensors", comm);
    }

    // Save LoRA adapter if this is a LoRA model (includes adapter weights and optimizer state)
    // All ranks participate in NCCL barriers; only rank 0's node writes.
    model.save_lora_checkpoint(target, comm);

    comm.barrier();  // only write checkpoint.json once we know all the shard files are saved

    if (comm.rank() == 0) {
        if(loader) {
            meta_data["data-loader"] = nlohmann::json::object({
                  {"seed",        loader->seed()},
                  {"chunk_index", loader->chunk_index()},
                  {"file_index",  loader->file_index()},
                  {"epoch",       loader->epoch()}
            });
        }

        meta_data["run"] = nlohmann::json::object({
            {"step", step},
            {"rng", model.rng_state()},
            {"past_losses.values", model.get_run_state().LossOutliers.mValues},
            {"past_losses.index", model.get_run_state().LossOutliers.mIndex},
            {"past_losses.size", model.get_run_state().LossOutliers.mWindowSize},
            {"past_norms.values", model.get_run_state().NormOutliers.mValues},
            {"past_norms.index", model.get_run_state().NormOutliers.mIndex},
            {"past_norms.size", model.get_run_state().NormOutliers.mWindowSize},
        });

        // in order to ensure that the mSum / mSumSq are bitwise identical, force a re-evaluation
        // here. we will do the same re-evaluation after loading the checkpoint.
        model.get_run_state().LossOutliers.re_evaluate();
        model.get_run_state().NormOutliers.re_evaluate();

        meta_data["distributed"] = nlohmann::json::object({
            {"world", shard_count},
            {"ep_size", comm.ep_size()},
        });

        std::ofstream file(target + "/checkpoint.json");
        file << std::setw(2) << meta_data;
    }

    return target;
}

/**
 * @brief Load a sharded checkpoint (weights + optimizer state) for the current rank.
 *
 * Reads `checkpoint.json` to validate distributed world size and to restore RNG (and
 * optionally dataloader) state, then loads per-rank safetensors shards for weights and
 * optimizer state. Calls `model.on_restore_checkpoint(comm)` after tensors are loaded.
 *
 * @param source Base checkpoint directory (the step subdirectory is appended).
 * @param step Training step number identifying which checkpoint to load.
 * @param model Model instance to populate with restored weights/optimizer/RNG state.
 * @param loader Optional dataloader to restore iteration state into; may be null.
 * @param comm NCCL communicator used for rank/world information and barriers.
 *
 * @throws std::runtime_error If the checkpoint directory or checkpoint.json does not exist,
 *         if checkpoint.json cannot be opened/parsed, or if the checkpoint world size
 *         differs from the current communicator world size.
 * @throws std::exception Propagates errors thrown by safetensors loading utilities.
 */
void load_checkpoint(std::string source, int step, IModel& model, DataLoader* loader, NCCLCommunicator& comm) {
    comm.barrier();
    source = get_checkpoint_path(std::move(source), step);
    if(!std::filesystem::exists(source)) {
        throw std::runtime_error("Checkpoint not found: " + source);
    }

    std::ifstream file(source + "/checkpoint.json");
    if(!file.is_open()) {
        throw std::runtime_error(fmt::format("could not open config file {}", source + "/checkpoint.json"));
    }

    nlohmann::json meta_data = nlohmann::json::parse(file);

    // Checkpoints are saved using local_rank / num_local_gpus shard indexing,
    // so the recorded "world" matches num_local_gpus (not global world_size).
    const int shard_rank = comm.local_rank();
    const int shard_count = comm.num_local_gpus();
    if(int ws = meta_data["distributed"]["world"].get<int>(); ws != shard_count) {
        throw std::runtime_error(
            fmt::format("Loading checkpoints with different shard count is not supported: Current num_local_gpus: {}, checkpoint world: {}",
                        shard_count, ws));
    }

    // Validate EP size matches (expert weights are distributed, so ep_size must be the same)
    const int checkpoint_ep_size = meta_data["distributed"].value("ep_size", 1);
    if (checkpoint_ep_size != comm.ep_size()) {
        throw std::runtime_error(
            fmt::format("Loading checkpoints with different EP size is not supported: "
                        "current ep_size={}, checkpoint ep_size={}",
                        comm.ep_size(), checkpoint_ep_size));
    }

    model.set_rng_state(meta_data["run"]["rng"].get<std::vector<std::byte>>());

    if (loader) {
        const auto& dl = meta_data["data-loader"];
        loader->set_state(dl["seed"].get<std::uint64_t>(), dl["epoch"].get<int>(), dl["file_index"].get<int>(), dl["chunk_index"].get<int>());
    }

    // Check if this is a LoRA-only checkpoint (no base model weights saved)
    const std::string weights_file = source + fmt::format("/weights.shard_{:03}_of_{:03}.safetensors", shard_rank, shard_count);
    const bool has_base_weights = std::filesystem::exists(weights_file);

    if (has_base_weights) {
        // Load base model weights
        load_safetensors(weights_file, model.weights(), false);

        // Pre-allocate optimizer state buffers before loading.
        // This ensures the 8-bit AdamW state tensors are allocated so load_safetensors can fill them.
        model.prepare_optimizer_for_checkpoint_load();

        // load optimizer shards
        load_safetensors(source + fmt::format("/adam.m.shard_{:03}_of_{:03}.safetensors", shard_rank, shard_count), model.opt_momentum(), false);
        load_safetensors(source + fmt::format("/adam.v.shard_{:03}_of_{:03}.safetensors", shard_rank, shard_count), model.opt_variance(), false);

        bool has_scales = false;
        model.opt_momentum_scales().iterate_tensors([&has_scales](const std::string& name, const TensorShard& tensor){
            if(tensor.Data != nullptr) {
                has_scales = true;
            }
        });
        if(has_scales) {
            load_safetensors(source + fmt::format("/adam.m.scales.shard_{:03}_of_{:03}.safetensors", shard_rank, shard_count), model.opt_momentum_scales(), false);
        }

        bool has_v_scales = false;
        model.opt_variance_scales().iterate_tensors([&has_v_scales](const std::string&, const TensorShard& tensor) {
            if (tensor.Data != nullptr) {
                has_v_scales = true;
            }
        });
        if (has_v_scales) {
            load_safetensors(source + fmt::format("/adam.v.scales.shard_{:03}_of_{:03}.safetensors", shard_rank, shard_count), model.opt_variance_scales(), false);
        }
    }

    // Load LoRA adapter if this is a LoRA model and adapter exists
    model.load_lora_checkpoint(source, comm);

    model.on_restore_checkpoint(comm);
}

/**
 * @brief Read the world size stored in a checkpoint's metadata.
 *
 * @param checkpoint_directory Base directory in which checkpoints are stored.
 * @param step Training step number identifying which checkpoint to inspect.
 * @return World size recorded in `checkpoint.json` for the specified checkpoint.
 *
 * @throws std::runtime_error If the checkpoint directory does not exist or checkpoint.json
 *         cannot be opened/parsed.
 */
int get_checkpoint_world_size(std::string checkpoint_directory, int step) {
    std::string path = get_checkpoint_path(checkpoint_directory, step);
    if(!std::filesystem::exists(path)) {
        throw std::runtime_error("Checkpoint not found: " + path);
    }

    std::ifstream file(path + "/checkpoint.json");
    if(!file.is_open()) {
        throw std::runtime_error(fmt::format("could not open config file {}", path + "/checkpoint.json"));
    }
    nlohmann::json meta_data = nlohmann::json::parse(file);
    return meta_data["distributed"]["world"].get<int>();
}

/**
 * @brief List all available checkpoint step numbers in a directory.
 *
 * Scans immediate subdirectories of @p checkpoint_directory for names matching
 * `step_...` and returns the parsed integer step values (only steps > 0).
 *
 * @param checkpoint_directory Base directory in which checkpoints are stored.
 * @return Vector of discovered checkpoint steps (unsorted).
 */
std::vector<int> get_all_checkpoints(const std::string& checkpoint_directory) {
    std::filesystem::path path(checkpoint_directory);
    if (!exists(path)) {
        return {};
    }
    std::filesystem::directory_iterator end_iter;
    std::vector<int> checkpoints;
    for(auto it = std::filesystem::directory_iterator(path); it != end_iter; ++it) {
        if(std::filesystem::is_directory(*it)) {
            std::string name = it->path().filename().string();
            if(name.starts_with("step_")) {
                int step = std::stoi(name.substr(5));
                if(step > 0) {
                    checkpoints.push_back(step);
                }
            }
        }
    }
    return checkpoints;
}

/**
 * @brief Find the latest (maximum step) checkpoint in a directory.
 *
 * @param checkpoint_directory Base directory in which checkpoints are stored.
 * @return Maximum checkpoint step found, or -1 if no checkpoints exist.
 */
int find_latest_checkpoint(const std::string& checkpoint_directory) {
    auto checkpoints = get_all_checkpoints(checkpoint_directory);
    return checkpoints.empty() ? -1 : *std::max_element(checkpoints.begin(), checkpoints.end());
}


/**
 * @brief Remove older checkpoints while keeping the newest N and optionally preserving "major" ones.
 *
 * Collects all checkpoints, optionally excludes "major" checkpoints (steps divisible by
 * @p major_every) from deletion, then removes the oldest remaining checkpoints until only
 * @p n_to_keep remain.
 *
 * @param checkpoint_directory Base directory in which checkpoints are stored.
 * @param n_to_keep Number of (non-major) checkpoints to keep (newest retained).
 * @param major_every If > 0, checkpoints where `step % major_every == 0` are preserved.
 * @return List of filesystem paths that were removed.
 *
 * @throws std::filesystem::filesystem_error If removal fails.
 */
std::vector<std::string> clean_old_checkpoints(const std::string& checkpoint_directory, int n_to_keep, int major_every) {
    auto checkpoints = get_all_checkpoints(checkpoint_directory);
    if(checkpoints.size() <= n_to_keep) {
        return {};
    }

    std::vector<std::string> removed;
    // leave major checkpoints untouched
    if(major_every > 0) {
        std::erase_if(checkpoints, [&](int step) { return step % major_every == 0; });
    }
    std::sort(checkpoints.begin(), checkpoints.end());
    for(int i = 0; i < checkpoints.size() - n_to_keep; ++i) {
        std::string path = get_checkpoint_path(checkpoint_directory, checkpoints[i]);
        removed.push_back(path);
        std::filesystem::remove_all(path);
    }

    return removed;
}
