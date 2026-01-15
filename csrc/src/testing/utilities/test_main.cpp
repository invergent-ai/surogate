// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0

#define CATCH_CONFIG_RUNNER
#include <catch2/catch_all.hpp>

#include <CLI/CLI.hpp>
#include <vector>

#include "test_config.h"

int main(int argc, char** argv) {
    testing_config::TestSizeConfig cfg{};
    CLI::App app{"SUROGATE unit tests"};
    app.allow_extras();

    app.add_option("-B, --batch", cfg.B, "Batch size (B)");
    app.add_option("-T, --seq",   cfg.T, "Sequence length (T)");
    app.add_option("-C, --channels", cfg.C, "Channel size (C)");
    app.add_option("--query-heads", cfg.Nq, "Query heads (Nq)");
    app.add_option("--kv-heads", cfg.Nkv, "Key/Value heads (Nkv)");

    std::vector<std::string> remaining;
    try {
        app.parse(argc, argv);
        remaining = app.remaining();
    } catch (const CLI::ParseError &e) {
        return app.exit(e);
    }

    testing_config::set_test_config(cfg);

    // Forward remaining args to Catch2
    std::vector<const char*> args;
    args.reserve(1 + remaining.size());
    args.push_back(argv[0]);
    for (const auto& s : remaining) {
        args.push_back(s.c_str());
    }

    return Catch::Session().run((int)args.size(), const_cast<char**>(args.data()));
}
