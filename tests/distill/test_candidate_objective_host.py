"""Exercise the actual native all-rank preflight without loading a GPU model."""

import shutil
import subprocess
from pathlib import Path

import pytest


def test_native_candidate_objective_and_mask_validation(tmp_path):
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("Requires C++ compiler for native host preflight")
    root = Path(__file__).resolve().parents[2]
    source = tmp_path / "check.cpp"
    source.write_text(r"""
#include "kernels/candidate_objective.h"
#include <cassert>
#include <vector>
#include <functional>
void rejects(const std::function<void()>& fn) {
    bool failed = false;
    try { fn(); } catch (const std::invalid_argument&) { failed = true; }
    assert(failed);
}
int main() {
    assert(parse_candidate_objective("cross_entropy") == CandidateObjective::CrossEntropy);
    assert(parse_candidate_objective("brier") == CandidateObjective::Brier);
    assert(parse_candidate_objective("rps") == CandidateObjective::Rps);
    rejects([] { parse_candidate_objective("forward_kl"); });
    rejects([] { validate_candidate_objective(static_cast<CandidateObjective>(9)); });
    const int targets[]{4, -100};
    int ids[]{9, -1, 4, 2, -99, -99, -99, -99};
    for (auto objective : {CandidateObjective::CrossEntropy, CandidateObjective::Brier, CandidateObjective::Rps}) {
        validate_candidate_rows(targets, ids, 2, 4, 10, objective);
    }
    auto test = [&] { validate_candidate_rows(targets, ids, 2, 4, 10, CandidateObjective::Rps); };
    ids[0] = 4; rejects(test); // duplicate
    ids[0] = 10; rejects(test); // out of vocabulary
    ids[0] = -2; rejects(test); // only -1 is padding
    ids[0] = 9; ids[2] = 5; rejects(test); // missing gold
    ids[0] = -1; ids[2] = 4; ids[3] = -1; rejects(test); // only one allowed
    rejects([&] { validate_candidate_rows(nullptr, ids, 2, 4, 10, CandidateObjective::Rps); });
    std::vector<int> many(512, -1);
    for (int i=0;i<255;++i) many[2*i] = i;
    const int gold[]{3};
    validate_candidate_rows(gold, many.data(), 1, 512, 300, CandidateObjective::Rps);
    many[511] = 255;
    validate_candidate_rows(gold, many.data(), 1, 512, 300, CandidateObjective::CrossEntropy);
    rejects([&] { validate_candidate_rows(gold, many.data(), 1, 512, 300, CandidateObjective::Brier); });
    rejects([&] { validate_candidate_rows(gold, many.data(), 1, 512, 300, CandidateObjective::Rps); });
}
""")
    executable = tmp_path / "check"
    subprocess.run(
        [compiler, "-std=c++17", "-I", str(root / "csrc/src"), str(source), "-o", str(executable)],
        check=True,
        capture_output=True,
    )
    subprocess.run([str(executable)], check=True, capture_output=True)
