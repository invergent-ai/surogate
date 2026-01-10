// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// Copyright (c) 2025, IST Austria, developed by Erik Schultheis
// SPDX-License-Identifier: Apache-2.0
//

#ifndef SUROGATE_SRC_UTILITIES_COMM_H
#define SUROGATE_SRC_UTILITIES_COMM_H

#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <vector>

#include <cuda_bf16.h>

namespace std
{
    class jthread;
}

struct Tensor;
struct TensorShard;

typedef struct ncclComm* ncclComm_t;
typedef struct CUevent_st* cudaEvent_t;
typedef struct CUstream_st* cudaStream_t;
class NCCLCommunicator;

class CommunicatorThreadsPack {
public:
    virtual ~CommunicatorThreadsPack() = default;
    virtual void join() = 0;
    virtual bool has_exception() const = 0;
};


class NCCLCommunicator {
public:
    NCCLCommunicator(int rank, int world, const void* nccl_id, int local_device = -1);
    virtual ~NCCLCommunicator();

    // Cpu-side barrier (includes cross-node sync if multi-node)
    virtual void barrier() = 0;

    // Local barrier (only syncs threads on this node)
    virtual void local_barrier() = 0;

    void begin_transaction(cudaEvent_t ready);
    void begin_transaction(cudaStream_t wait_for_stream);
    void schedule_reduce_scatter(Tensor& tensor);
    void schedule_all_gather(const TensorShard& src, Tensor& tgt);
    // like all-to-all, except the local shard will *not* be preserved, and results will be shifted cyclically
    void schedule_destructive_all_to_all(const Tensor& tensor);
    void execute_transaction(cudaEvent_t signal);

    void reduce_loss(float* loss, cudaStream_t stream);
    void reduce_norm(float* norm_squared, cudaStream_t stream);
    void all_reduce_sum_int(int* values, int n, cudaStream_t stream);

    void reduce_max(float* values, int n = 1, cudaStream_t stream=nullptr);

    //! All-reduce tensor data in-place using average (for gradient averaging in data parallelism)
    void all_reduce_avg(Tensor& tensor, cudaStream_t stream);

    void wait_on_comms(cudaStream_t compute_stream);

    [[nodiscard]] int rank() const { return mRank; }
    [[nodiscard]] int world_size() const { return mWorld; }
    [[nodiscard]] int local_rank() const { return mLocalRank; }

    [[nodiscard]] cudaStream_t stream() const { return mCommsStream; }

    //! On the root rank, returns a vector of (memcpyable) T objects that
    //! have been gathered from all ranks.
    template<typename T>
    std::vector<T> host_gather(const T& object) {
        static_assert(std::is_trivially_copyable_v<T>, "Cannot communicate type with non-trivial copy operator");
        std::vector<T> result;
        if(rank() == 0) {
            result.resize(world_size());
        }

        gather_bytes_host(reinterpret_cast<std::byte*>(result.data()), reinterpret_cast<const std::byte*>(&object), sizeof(T));
        return result;
    }

    template<typename T>
    std::vector<T> host_all_gather(const T& object) {
        static_assert(std::is_trivially_copyable_v<T>, "Cannot communicate type with non-trivial copy operator");
        std::vector<T> result(world_size());
        all_gather_bytes_host(reinterpret_cast<std::byte*>(result.data()), reinterpret_cast<const std::byte*>(&object), sizeof(T));
        return result;
    }

    /**
     * @brief Run distributed training with one thread per local GPU (blocking).
     *
     * Single-node: Launches threads for each GPU, uses std::barrier for synchronization.
     * Multi-node: When MPI environment is detected (OMPI_COMM_WORLD_SIZE), coordinates
     * across nodes using MPI while maintaining threaded intra-node communication.
     *
     * @param ngpus Number of local GPUs to use (0 = auto-detect all available).
     * @param memcpy_allgather Enable memcpy-based all-gather emulation.
     * @param memcpy_send_recv Enable memcpy-based send/recv emulation.
     * @param work Callable invoked once per GPU with that GPU's communicator.
     */
    static void run_communicators(int ngpus, bool memcpy_allgather, bool memcpy_send_recv, std::function<void(NCCLCommunicator& comm)> work);

    /**
     * @brief Launch communicator threads and return a joinable pack (non-blocking).
     *
     * Same as run_communicators but returns immediately with a joinable pack.
     * Used by Python bindings to run training in background threads.
     *
     * @param ngpus Number of local GPUs to use (0 = auto-detect all available).
     * @param memcpy_allgather Enable memcpy-based all-gather emulation.
     * @param memcpy_send_recv Enable memcpy-based send/recv emulation.
     * @param work Callable invoked once per GPU with that GPU's communicator.
     * @return Joinable pack that can be used to wait for completion.
     */
    static std::unique_ptr<CommunicatorThreadsPack> launch_communicators(int ngpus, bool memcpy_allgather, bool memcpy_send_recv, std::function<void(NCCLCommunicator& comm)> work);

    /**
     * @brief Launch communicator threads with externally-provided NCCL IDs (for Ray multi-node).
     *
     * Used for multi-node training where NCCL IDs are coordinated externally (e.g., via Ray).
     * Creates two NCCL communicators:
     * 1. Global communicator spanning all GPUs across all nodes (for gradient sync)
     * 2. Node master communicator spanning only local_rank=0 on each node (for host gather/barrier)
     *
     * @param ngpus Number of local GPUs on this node.
     * @param node_rank This node's rank (0 to num_nodes-1).
     * @param num_nodes Total number of nodes.
     * @param nccl_id Shared NCCL unique ID for global communicator (128 bytes, same on all nodes).
     * @param node_master_nccl_id Shared NCCL unique ID for node master communicator (128 bytes, same on all nodes).
     * @param memcpy_allgather Enable memcpy-based all-gather emulation.
     * @param memcpy_send_recv Enable memcpy-based send/recv emulation.
     * @param work Callable invoked once per GPU with that GPU's communicator.
     * @return Joinable pack that can be used to wait for completion.
     */
    static std::unique_ptr<CommunicatorThreadsPack> launch_communicators_multinode(
        int ngpus,
        int node_rank,
        int num_nodes,
        const void* nccl_id,
        const void* node_master_nccl_id,
        bool memcpy_allgather,
        bool memcpy_send_recv,
        std::function<void(NCCLCommunicator& comm)> work
    );

    /**
     * @brief Generate a new NCCL unique ID.
     * @return 128-byte unique ID that must be shared across all nodes.
     */
    static std::array<std::byte, 128> generate_nccl_id();

protected:
    void terminate_nccl();

    void scatter_grad(float* value, std::size_t size);
    void scatter_grad(nv_bfloat16* value, std::size_t size);

    virtual void gather_weight(const std::byte* src, std::byte* tgt, std::size_t size);
    virtual void send(const std::byte* src, int peer, std::size_t size);
    virtual void recv(std::byte* tgt, int peer, std::size_t size);


    virtual void gather_bytes_host(std::byte* recv, const std::byte* object, std::size_t size) = 0;
    virtual void all_gather_bytes_host(std::byte* recv, const std::byte* object, std::size_t size) = 0;

    struct CommandBuffer;
    virtual void on_execute_transaction(const CommandBuffer&) = 0;
    virtual void on_finish_transaction(cudaEvent_t signal) = 0;
    virtual void _launch_queue_throttle_sync() = 0;
private:
    ncclComm_t mNcclComm;
    int mRank;
    int mWorld;
    int mLocalRank;  // Local device index (same as rank for single-node, device index for multi-node)

    cudaEvent_t mCommsSync;
    cudaStream_t mCommsStream;

    struct CommandVisitor;
    std::unique_ptr<CommandBuffer> mCmdBuf;

    friend struct CommandVisitor;
};

#endif //SUROGATE_SRC_UTILITIES_COMM_H
