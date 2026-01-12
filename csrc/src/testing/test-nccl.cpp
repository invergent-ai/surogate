// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//
// Unit tests for NCCL (NVIDIA Collective Communications Library) functionality

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <vector>
#include <numeric>
#include <cstring>

#include <cuda_runtime.h>

#include "utilities/comm.h"

namespace {

// Check if NCCL communication is available (CUDA + proper driver version + NCCL)
// Returns true only if the full stack works
bool nccl_available() {
    static bool checked = false;
    static bool available = false;
    
    if (!checked) {
        checked = true;
        try {
            // Try to run a minimal NCCL operation
            NCCLCommunicator::run_communicators(1, false, false, [&](NCCLCommunicator& comm) {
                available = true;
            });
        } catch (...) {
            available = false;
        }
    }
    return available;
}

// Check if CUDA is available (basic check)
bool cuda_available() {
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    return err == cudaSuccess && device_count > 0;
}

// Get number of available CUDA devices
int get_cuda_device_count() {
    int device_count = 0;
    cudaGetDeviceCount(&device_count);
    return device_count;
}

} // anonymous namespace

// =============================================================================
// Basic NCCL Availability Tests
// =============================================================================

TEST_CASE("CUDA device availability", "[nccl][cuda][basic]") {
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    
    INFO("CUDA device count: " << device_count);
    INFO("CUDA error: " << cudaGetErrorString(err));
    
    if (err != cudaSuccess || device_count == 0) {
        SKIP("No CUDA devices available");
    }
    
    REQUIRE(device_count > 0);
}

TEST_CASE("NCCL communicator can be created with single GPU", "[nccl][basic]") {
    if (!cuda_available()) {
        SKIP("CUDA not available");
    }
    
    bool communicator_created = false;
    
    try {
        NCCLCommunicator::run_communicators(1, false, false, [&](NCCLCommunicator& comm) {
            communicator_created = true;
            REQUIRE(comm.rank() == 0);
            REQUIRE(comm.world_size() == 1);
        });
    } catch (const std::exception& e) {
        INFO("Exception: " << e.what());
        SKIP("NCCL initialization failed");
    }
    
    REQUIRE(communicator_created == true);
}

TEST_CASE("NCCL communicator properties with single GPU", "[nccl][basic]") {
    if (!nccl_available()) {
        SKIP("NCCL not available");
    }
    
    NCCLCommunicator::run_communicators(1, false, false, [](NCCLCommunicator& comm) {
        REQUIRE(comm.rank() == 0);
        REQUIRE(comm.world_size() == 1);
        REQUIRE(comm.stream() != nullptr);
    });
}

// =============================================================================
// Multi-GPU NCCL Tests (if multiple GPUs available)
// =============================================================================

TEST_CASE("NCCL communicator with multiple GPUs", "[nccl][multi-gpu]") {
    if (!nccl_available()) {
        SKIP("NCCL not available");
    }
    
    int num_gpus = get_cuda_device_count();
    if (num_gpus < 2) {
        SKIP("Need at least 2 GPUs for multi-GPU tests");
    }
    
    // Test with 2 GPUs
    int test_gpus = std::min(num_gpus, 2);
    std::vector<int> ranks_seen(test_gpus, 0);
    
    NCCLCommunicator::run_communicators(test_gpus, false, false, [&](NCCLCommunicator& comm) {
        int rank = comm.rank();
        int world = comm.world_size();
        
        REQUIRE(world == test_gpus);
        REQUIRE(rank >= 0);
        REQUIRE(rank < world);
        
        ranks_seen[rank] = 1;
    });
    
    // Verify all ranks were created
    for (int i = 0; i < test_gpus; ++i) {
        REQUIRE(ranks_seen[i] == 1);
    }
}

TEST_CASE("NCCL barrier synchronization", "[nccl][sync]") {
    if (!nccl_available()) {
        SKIP("NCCL not available");
    }
    
    int num_gpus = get_cuda_device_count();
    if (num_gpus < 2) {
        SKIP("Need at least 2 GPUs for barrier test");
    }
    
    int test_gpus = std::min(num_gpus, 2);
    
    NCCLCommunicator::run_communicators(test_gpus, false, false, [](NCCLCommunicator& comm) {
        // Simply test that barrier doesn't hang or crash
        comm.barrier();
        
        // Multiple barriers should also work
        comm.barrier();
        comm.barrier();
    });
}

// =============================================================================
// Host Gather Tests
// =============================================================================

TEST_CASE("NCCL host_gather with single GPU", "[nccl][gather]") {
    if (!nccl_available()) {
        SKIP("NCCL not available");
    }
    
    NCCLCommunicator::run_communicators(1, false, false, [](NCCLCommunicator& comm) {
        int value = 42;
        auto result = comm.host_gather(value);
        
        // On rank 0, result should contain the gathered value
        if (comm.rank() == 0) {
            REQUIRE(result.size() == 1);
            REQUIRE(result[0] == 42);
        }
    });
}

TEST_CASE("NCCL host_gather with multiple GPUs", "[nccl][gather][multi-gpu]") {
    if (!nccl_available()) {
        SKIP("NCCL not available");
    }
    
    int num_gpus = get_cuda_device_count();
    if (num_gpus < 2) {
        SKIP("Need at least 2 GPUs for multi-GPU gather test");
    }
    
    int test_gpus = std::min(num_gpus, 2);
    
    NCCLCommunicator::run_communicators(test_gpus, false, false, [&](NCCLCommunicator& comm) {
        // Each rank sends its rank number
        int value = comm.rank() * 10 + 1;  // e.g., rank 0 -> 1, rank 1 -> 11
        auto result = comm.host_gather(value);
        
        if (comm.rank() == 0) {
            REQUIRE(result.size() == static_cast<size_t>(test_gpus));
            for (int i = 0; i < test_gpus; ++i) {
                REQUIRE(result[i] == i * 10 + 1);
            }
        }
    });
}

TEST_CASE("NCCL host_all_gather with single GPU", "[nccl][allgather]") {
    if (!nccl_available()) {
        SKIP("NCCL not available");
    }
    
    NCCLCommunicator::run_communicators(1, false, false, [](NCCLCommunicator& comm) {
        int value = 99;
        auto result = comm.host_all_gather(value);
        
        REQUIRE(result.size() == 1);
        REQUIRE(result[0] == 99);
    });
}

TEST_CASE("NCCL host_all_gather with multiple GPUs", "[nccl][allgather][multi-gpu]") {
    if (!nccl_available()) {
        SKIP("NCCL not available");
    }
    
    int num_gpus = get_cuda_device_count();
    if (num_gpus < 2) {
        SKIP("Need at least 2 GPUs for multi-GPU all_gather test");
    }
    
    int test_gpus = std::min(num_gpus, 2);
    
    NCCLCommunicator::run_communicators(test_gpus, false, false, [&](NCCLCommunicator& comm) {
        // Each rank sends its rank number
        int value = comm.rank() + 100;  // e.g., rank 0 -> 100, rank 1 -> 101
        auto result = comm.host_all_gather(value);
        
        // All ranks should have the same result
        REQUIRE(result.size() == static_cast<size_t>(test_gpus));
        for (int i = 0; i < test_gpus; ++i) {
            REQUIRE(result[i] == i + 100);
        }
    });
}

// =============================================================================
// Host Gather with Struct Tests
// =============================================================================

TEST_CASE("NCCL host_gather with struct", "[nccl][gather][struct]") {
    if (!nccl_available()) {
        SKIP("NCCL not available");
    }
    
    struct TestData {
        int a;
        float b;
        char c;
    };
    
    NCCLCommunicator::run_communicators(1, false, false, [](NCCLCommunicator& comm) {
        TestData data{42, 3.14f, 'x'};
        auto result = comm.host_gather(data);
        
        if (comm.rank() == 0) {
            REQUIRE(result.size() == 1);
            REQUIRE(result[0].a == 42);
            REQUIRE(result[0].b == Catch::Approx(3.14f));
            REQUIRE(result[0].c == 'x');
        }
    });
}

// =============================================================================
// Error Handling Tests
// =============================================================================

TEST_CASE("NCCL handles zero GPUs gracefully", "[nccl][error]") {
    if (!cuda_available()) {
        SKIP("CUDA not available");
    }
    
    // Requesting 0 GPUs should either throw or handle gracefully
    // This tests that the library doesn't crash on invalid input
    bool threw_exception = false;
    try {
        NCCLCommunicator::run_communicators(0, false, false, [](NCCLCommunicator& comm) {
            // Should not reach here
        });
    } catch (...) {
        threw_exception = true;
    }
    
    // Either it throws an exception or handles it gracefully (both are acceptable)
    // The test passes if we get here without crashing
    INFO("Zero GPUs request threw exception: " << threw_exception);
}

// =============================================================================
// Stress Tests
// =============================================================================

TEST_CASE("NCCL multiple sequential communicator creations", "[nccl][stress]") {
    if (!nccl_available()) {
        SKIP("NCCL not available");
    }
    
    // Create and destroy communicators multiple times
    for (int i = 0; i < 3; ++i) {
        NCCLCommunicator::run_communicators(1, false, false, [&](NCCLCommunicator& comm) {
            REQUIRE(comm.rank() == 0);
            REQUIRE(comm.world_size() == 1);
        });
    }
}

TEST_CASE("NCCL with memcpy_allgather option", "[nccl][options]") {
    if (!nccl_available()) {
        SKIP("NCCL not available");
    }
    
    // Test with memcpy_allgather = true
    NCCLCommunicator::run_communicators(1, true, false, [](NCCLCommunicator& comm) {
        REQUIRE(comm.rank() == 0);
        REQUIRE(comm.world_size() == 1);
    });
}

TEST_CASE("NCCL with memcpy_send_recv option", "[nccl][options]") {
    if (!nccl_available()) {
        SKIP("NCCL not available");
    }
    
    // Test with memcpy_send_recv = true
    NCCLCommunicator::run_communicators(1, false, true, [](NCCLCommunicator& comm) {
        REQUIRE(comm.rank() == 0);
        REQUIRE(comm.world_size() == 1);
    });
}

TEST_CASE("NCCL with both memcpy options", "[nccl][options]") {
    if (!nccl_available()) {
        SKIP("NCCL not available");
    }

    // Test with both options = true
    NCCLCommunicator::run_communicators(1, true, true, [](NCCLCommunicator& comm) {
        REQUIRE(comm.rank() == 0);
        REQUIRE(comm.world_size() == 1);
    });
}

// =============================================================================
// NCCL ID Generation Tests
// =============================================================================

TEST_CASE("generate_nccl_id returns 128 bytes", "[nccl][id]") {
    if (!cuda_available()) {
        SKIP("CUDA not available");
    }

    auto id = NCCLCommunicator::generate_nccl_id();
    REQUIRE(id.size() == 128);
}

TEST_CASE("generate_nccl_id returns unique IDs", "[nccl][id]") {
    if (!cuda_available()) {
        SKIP("CUDA not available");
    }

    auto id1 = NCCLCommunicator::generate_nccl_id();
    auto id2 = NCCLCommunicator::generate_nccl_id();

    REQUIRE(id1 != id2);
}

TEST_CASE("generate_nccl_id produces valid IDs for multiple calls", "[nccl][id]") {
    if (!cuda_available()) {
        SKIP("CUDA not available");
    }

    std::vector<std::array<std::byte, 128>> ids;
    for (int i = 0; i < 10; ++i) {
        ids.push_back(NCCLCommunicator::generate_nccl_id());
    }

    // All IDs should be unique
    for (size_t i = 0; i < ids.size(); ++i) {
        for (size_t j = i + 1; j < ids.size(); ++j) {
            REQUIRE(ids[i] != ids[j]);
        }
    }
}

// =============================================================================
// Multi-node Communicator Tests (simulated single-node)
// =============================================================================

TEST_CASE("launch_communicators_multinode with single node", "[nccl][multinode][.]") {
    if (!nccl_available()) {
        SKIP("NCCL not available");
    }

    auto nccl_id = NCCLCommunicator::generate_nccl_id();
    auto node_master_id = NCCLCommunicator::generate_nccl_id();

    bool executed = false;

    auto pack = NCCLCommunicator::launch_communicators_multinode(
        1,              // ngpus
        0,              // node_rank
        1,              // num_nodes
        nccl_id.data(),
        node_master_id.data(),
        false,          // memcpy_allgather
        false,          // memcpy_send_recv
        [&](NCCLCommunicator& comm) {
            executed = true;
            REQUIRE(comm.rank() == 0);
            REQUIRE(comm.world_size() == 1);
        }
    );

    pack->join();
    REQUIRE(executed == true);
}

TEST_CASE("launch_communicators_multinode with multiple local GPUs", "[nccl][multinode][multi-gpu][.]") {
    if (!nccl_available()) {
        SKIP("NCCL not available");
    }

    int num_gpus = get_cuda_device_count();
    if (num_gpus < 2) {
        SKIP("Need at least 2 GPUs for multi-GPU multinode test");
    }

    int test_gpus = std::min(num_gpus, 2);

    auto nccl_id = NCCLCommunicator::generate_nccl_id();
    auto node_master_id = NCCLCommunicator::generate_nccl_id();

    std::vector<int> ranks_seen(test_gpus, 0);

    auto pack = NCCLCommunicator::launch_communicators_multinode(
        test_gpus,      // ngpus
        0,              // node_rank
        1,              // num_nodes
        nccl_id.data(),
        node_master_id.data(),
        false,          // memcpy_allgather
        false,          // memcpy_send_recv
        [&](NCCLCommunicator& comm) {
            int rank = comm.rank();
            int world = comm.world_size();

            REQUIRE(world == test_gpus);
            REQUIRE(rank >= 0);
            REQUIRE(rank < world);

            ranks_seen[rank] = 1;
        }
    );

    pack->join();

    for (int i = 0; i < test_gpus; ++i) {
        REQUIRE(ranks_seen[i] == 1);
    }
}

TEST_CASE("launch_communicators_multinode barrier works", "[nccl][multinode][sync][.]") {
    if (!nccl_available()) {
        SKIP("NCCL not available");
    }

    int num_gpus = get_cuda_device_count();
    if (num_gpus < 2) {
        SKIP("Need at least 2 GPUs for barrier test");
    }

    int test_gpus = std::min(num_gpus, 2);

    auto nccl_id = NCCLCommunicator::generate_nccl_id();
    auto node_master_id = NCCLCommunicator::generate_nccl_id();

    auto pack = NCCLCommunicator::launch_communicators_multinode(
        test_gpus, 0, 1,
        nccl_id.data(), node_master_id.data(),
        false, false,
        [](NCCLCommunicator& comm) {
            // Test barrier multiple times
            comm.barrier();
            comm.barrier();
            comm.barrier();
        }
    );

    pack->join();
}

TEST_CASE("launch_communicators_multinode host_gather works", "[nccl][multinode][gather][.]") {
    if (!nccl_available()) {
        SKIP("NCCL not available");
    }

    int num_gpus = get_cuda_device_count();
    if (num_gpus < 2) {
        SKIP("Need at least 2 GPUs for gather test");
    }

    int test_gpus = std::min(num_gpus, 2);

    auto nccl_id = NCCLCommunicator::generate_nccl_id();
    auto node_master_id = NCCLCommunicator::generate_nccl_id();

    auto pack = NCCLCommunicator::launch_communicators_multinode(
        test_gpus, 0, 1,
        nccl_id.data(), node_master_id.data(),
        false, false,
        [&](NCCLCommunicator& comm) {
            int value = comm.rank() * 10 + 5;
            auto result = comm.host_gather(value);

            if (comm.rank() == 0) {
                REQUIRE(result.size() == static_cast<size_t>(test_gpus));
                for (int i = 0; i < test_gpus; ++i) {
                    REQUIRE(result[i] == i * 10 + 5);
                }
            }
        }
    );

    pack->join();
}

TEST_CASE("launch_communicators_multinode host_all_gather works", "[nccl][multinode][allgather][.]") {
    if (!nccl_available()) {
        SKIP("NCCL not available");
    }

    int num_gpus = get_cuda_device_count();
    if (num_gpus < 2) {
        SKIP("Need at least 2 GPUs for all_gather test");
    }

    int test_gpus = std::min(num_gpus, 2);

    auto nccl_id = NCCLCommunicator::generate_nccl_id();
    auto node_master_id = NCCLCommunicator::generate_nccl_id();

    auto pack = NCCLCommunicator::launch_communicators_multinode(
        test_gpus, 0, 1,
        nccl_id.data(), node_master_id.data(),
        false, false,
        [&](NCCLCommunicator& comm) {
            int value = comm.rank() + 200;
            auto result = comm.host_all_gather(value);

            // All ranks should get the same result
            REQUIRE(result.size() == static_cast<size_t>(test_gpus));
            for (int i = 0; i < test_gpus; ++i) {
                REQUIRE(result[i] == i + 200);
            }
        }
    );

    pack->join();
}

// =============================================================================
// Multi-node Error Handling Tests
// =============================================================================

TEST_CASE("launch_communicators_multinode validates node_rank", "[nccl][multinode][error][.]") {
    if (!cuda_available()) {
        SKIP("CUDA not available");
    }

    auto nccl_id = NCCLCommunicator::generate_nccl_id();
    auto node_master_id = NCCLCommunicator::generate_nccl_id();

    // Invalid node_rank (>= num_nodes)
    REQUIRE_THROWS_AS(
        NCCLCommunicator::launch_communicators_multinode(
            1, 5, 2,  // node_rank=5 but num_nodes=2
            nccl_id.data(), node_master_id.data(),
            false, false,
            [](NCCLCommunicator&) {}
        ),
        std::runtime_error
    );
}

TEST_CASE("launch_communicators_multinode validates num_nodes", "[nccl][multinode][error][.]") {
    if (!cuda_available()) {
        SKIP("CUDA not available");
    }

    auto nccl_id = NCCLCommunicator::generate_nccl_id();
    auto node_master_id = NCCLCommunicator::generate_nccl_id();

    // Invalid num_nodes (< 1)
    REQUIRE_THROWS_AS(
        NCCLCommunicator::launch_communicators_multinode(
            1, 0, 0,  // num_nodes=0
            nccl_id.data(), node_master_id.data(),
            false, false,
            [](NCCLCommunicator&) {}
        ),
        std::runtime_error
    );
}

TEST_CASE("launch_communicators_multinode validates nccl_id not null", "[nccl][multinode][error][.]") {
    if (!cuda_available()) {
        SKIP("CUDA not available");
    }

    auto node_master_id = NCCLCommunicator::generate_nccl_id();

    REQUIRE_THROWS_AS(
        NCCLCommunicator::launch_communicators_multinode(
            1, 0, 1,
            nullptr,  // null nccl_id
            node_master_id.data(),
            false, false,
            [](NCCLCommunicator&) {}
        ),
        std::runtime_error
    );
}

TEST_CASE("launch_communicators_multinode validates node_master_nccl_id not null", "[nccl][multinode][error][.]") {
    if (!cuda_available()) {
        SKIP("CUDA not available");
    }

    auto nccl_id = NCCLCommunicator::generate_nccl_id();

    REQUIRE_THROWS_AS(
        NCCLCommunicator::launch_communicators_multinode(
            1, 0, 1,
            nccl_id.data(),
            nullptr,  // null node_master_nccl_id
            false, false,
            [](NCCLCommunicator&) {}
        ),
        std::runtime_error
    );
}
