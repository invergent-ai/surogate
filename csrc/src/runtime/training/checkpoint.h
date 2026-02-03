// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// Copyright (c) 2025, IST Austria, developed by Erik Schultheis
// SPDX-License-Identifier: Apache-2.0
//

#ifndef SUROGATE_TRAINING_CHECKPOINT_H
#define SUROGATE_TRAINING_CHECKPOINT_H

#include <string>
#include <vector>

class DataLoader;
class NCCLCommunicator;
class IModel;

//! Constructs full path for checkpoint at given step
std::string get_checkpoint_path(std::string checkpoint_directory, int step);

//! Saves distributed model state and training metadata
std::string save_checkpoint(std::string checkpoint_directory, int step, IModel& model,
                            const DataLoader* loader, NCCLCommunicator& comm);

//! Restores model and training state from checkpoint.
void load_checkpoint(std::string checkpoint_directory, int step, IModel& model,
                     DataLoader* loader, NCCLCommunicator& comm);

//! Gets the world size for which a checkpoint was created
int get_checkpoint_world_size(std::string checkpoint_directory, int step);

//! Lists available checkpoint step numbers.
std::vector<int> get_all_checkpoints(const std::string& checkpoint_directory);

//! Returns latest checkpoint step number, -1 if none exist
int find_latest_checkpoint(const std::string& checkpoint_directory);

//! Removes old checkpoints while preserving the latest `n_to_keep`, and any checkpoint whose
//! step number is a multiple of `major_every`. Note: `major_every` needs to be specified
//! in units of steps, not in units of minor checkpoints.
std::vector<std::string> clean_old_checkpoints(const std::string& checkpoint_directory, int n_to_keep, int major_every);


#endif //SUROGATE_TRAINING_CHECKPOINT_H
