// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0

#pragma once

namespace testing_config {

struct TestSizeConfig {
    int B = 2;
    int T = 64;
    int C = 768;
    int Nq = 4;
    int Nkv = 4;
};

inline TestSizeConfig& mutable_cfg() {
    static TestSizeConfig cfg{};
    return cfg;
}

inline void set_test_config(const TestSizeConfig& cfg) {
    if(cfg.Nq % cfg.Nkv != 0) {
        fprintf(stderr, "ERROR: Nq must be divisible by Nkv\n");
        exit(EXIT_FAILURE);
    }
    mutable_cfg() = cfg;
}

inline const TestSizeConfig& get_test_config() {
    return mutable_cfg();
}

} // namespace testing_config
