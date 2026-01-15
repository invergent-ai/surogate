// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0

#include <catch2/catch_all.hpp>

#include "training/logging.h"
#include "utilities/dtype.h"

TEST_CASE("TrainingRunLogger::log_sol_estimate does not throw on unsupported dtypes") {
    TrainingRunLogger logger("unit-test-sol.json", /*rank=*/0, TrainingRunLogger::SILENT);

    // ETensorDType::BYTE is intentionally unsupported by the SOL estimator.
    // The logger must not crash (or divide-by-zero) when SOL is unavailable.
    std::vector<std::pair<ETensorDType, long>> ops = {
        {ETensorDType::BYTE, 1},
        {ETensorDType::BYTE, 1},
        {ETensorDType::BYTE, 1},
    };

    REQUIRE_NOTHROW(logger.log_sol_estimate(ops, /*world_size=*/8));
}

