// The cards a process may touch: the pure plan behind the narrowing of CUDA_VISIBLE_DEVICES.
#include "product/cuda_visibility/cuda_visibility.h"

#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

int failures = 0;

void expect(bool ok, const char* what) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", what);
        ++failures;
    }
}

} // namespace

int main() {
    using sinfer::product::plan_cuda_visibility;
    // Physical indices, no variable set: the requested cards in order, numbered 0..n-1.
    {
        const auto plan = plan_cuda_visibility(nullptr, {5, 6, 7});
        expect(plan.visible == "5,6,7", "physical list");
        expect((plan.renumbered == std::vector<int>{0, 1, 2}), "physical renumbering");
        expect(plan.changed, "unset variable counts as changed");
    }
    // A single card.
    {
        const auto plan = plan_cuda_visibility(nullptr, {4});
        expect(plan.visible == "4" && plan.renumbered == std::vector<int>{0}, "single card");
    }
    // Through an existing variable: indices select its entries, in request order, deduplicated.
    {
        const std::string existing = "2,5,6,7";
        const auto plan            = plan_cuda_visibility(&existing, {1, 2, 3});
        expect(plan.visible == "5,6,7", "translated through the existing list");
        expect((plan.renumbered == std::vector<int>{0, 1, 2}), "translated renumbering");
        expect(plan.changed, "narrowed from a wider list");
    }
    {
        const std::string existing = "GPU-aaaa,GPU-bbbb";
        const auto plan            = plan_cuda_visibility(&existing, {1, 0, 1});
        expect(plan.visible == "GPU-bbbb,GPU-aaaa", "uuid entries kept verbatim, first appearance wins");
        expect((plan.renumbered == std::vector<int>{0, 1, 0}), "duplicates map to their first slot");
    }
    // Already exactly right: nothing changes.
    {
        const std::string existing = "3,4";
        const auto plan            = plan_cuda_visibility(&existing, {0, 1});
        expect(plan.visible == "3,4" && !plan.changed, "already narrowed");
    }
    // Outside the existing list is an error, not a silent widening.
    {
        const std::string existing = "2,5";
        bool threw                 = false;
        try {
            (void)plan_cuda_visibility(&existing, {2});
        } catch (const std::invalid_argument&) {
            threw = true;
        }
        expect(threw, "index outside CUDA_VISIBLE_DEVICES is refused");
    }
    if (failures == 0) { std::printf("OK cuda visibility plan\n"); }
    return failures == 0 ? 0 : 1;
}
