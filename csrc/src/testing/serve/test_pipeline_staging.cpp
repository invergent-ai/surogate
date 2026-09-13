#include "runtime/engine/pipeline_instance.h"

#include <cstdlib>
#include <iostream>
#include <new>

namespace {
constexpr std::size_t kBoundaryBytes = 4U << 20;
thread_local bool reject_staging = false;
struct Program {
    std::size_t stage_boundary_bytes() const { return kBoundaryBytes; }
    int stage_boundary_columns() const { return 512; }
    int speculative_round_width() const { return 1; }
};
struct Stage {
    struct Package {
        using PreparedPrompt = int;
        using RequestBasePlan = int;
        using RequestPlan = int;
    };
    Program* program;
};
}

void* operator new(std::size_t bytes) {
    if (reject_staging && bytes >= kBoundaryBytes) { throw std::bad_alloc(); }
    if (void* value = std::malloc(bytes ? bytes : 1)) { return value; }
    throw std::bad_alloc();
}
void operator delete(void* value) noexcept { std::free(value); }
void operator delete(void* value, std::size_t) noexcept { std::free(value); }

int main() {
    setenv("SUROGATE_SERVE_PIPELINE_GROUPS", "8", 1);
    Program programs[8];
    std::vector<Stage> stages;
    std::vector<Stage*> pointers;
    for (auto& program : programs) { stages.push_back({&program}); }
    for (auto& stage : stages) { pointers.push_back(&stage); }
    try {
        // Stage export/import storage exists independently. Constructing the
        // flight-based executor must not allocate 56 extra boundary buffers.
        reject_staging = true;
        sinfer::runtime::PipelineProgram<Stage> pipeline(pointers, {0,1,2,3,4,5,6,7});
        reject_staging = false;
        if (pipeline.group_count() != 8) { return 1; }
        std::cout << "pipeline construction needs no grouped staging payloads\n";
    } catch (const std::bad_alloc&) {
        reject_staging = false;
        std::cerr << "pipeline allocated unused boundary staging at construction\n";
        return 1;
    }
}
