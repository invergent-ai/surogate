// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// Copyright (c) 2025, IST Austria, developed by Erik Schultheis
// SPDX-License-Identifier: Apache-2.0
//

#include "comm.h"

#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <utility>
#include <variant>
#include <future>
#include <thread>
#include <barrier>

#include <nccl.h>
#include <fmt/core.h>

#include "gpu_info.h"
#include "kernels/kernels.h"
#include "tensor.h"
#include "utils.h"

/**
 * @brief Throws a std::runtime_error if an NCCL call returned an error.
 *
 * @param status NCCL status code returned by an NCCL API call.
 * @param file Source file where the failing call was made.
 * @param line Source line where the failing call was made.
 *
 * @throws std::runtime_error Always thrown when @p status != ncclSuccess.
 */
void nccl_check(ncclResult_t status, const char* file, int line) {
    if (status != ncclSuccess) {
        throw std::runtime_error(fmt::format("NCCL error at {}:{}: {}", file, line, ncclGetErrorString(status)));
    }
}
#define ncclCheck(err) (nccl_check(err, __FILE__, __LINE__))

struct NCCLCommunicator::CommandBuffer
{
    struct Gather {
        std::byte* Src;
        std::byte* Dst;
        std::size_t Bytes;
    };

    struct ScatterReduce {
        ETensorDType DType;
        std::byte* Tensor;
        std::size_t Elements;
    };


    struct Send {
        const std::byte* Tensor;
        std::size_t Bytes;
        int Target;
    };

    struct Recv {
        std::byte* Tensor;
        std::size_t Bytes;
        int Source;
    };

    struct AllReduce {
        ETensorDType DType;
        std::byte* Tensor;
        std::size_t Elements;
    };

    std::vector<std::variant<Gather, ScatterReduce, Send, Recv, AllReduce>> Commands;
    cudaEvent_t Ready = nullptr;
};

/**
 * @brief Construct an NCCLCommunicator for a given rank and world size.
 *
 * Sets the CUDA device to the local device index, initializes the NCCL communicator
 * using @p nccl_id, and creates a dedicated comms stream and sync event on that device.
 *
 * @param rank Global rank in the NCCL communicator (0 to world-1).
 * @param world Total number of ranks in the communicator.
 * @param nccl_id Pointer to an ncclUniqueId shared across all ranks.
 * @param local_device CUDA device index to use (defaults to rank for single-node).
 *
 * @throws std::runtime_error If NCCL initialization fails.
 */
NCCLCommunicator::NCCLCommunicator(int rank, int world, const void* nccl_id, int local_device) :
    mRank(rank), mWorld(world), mLocalRank(local_device >= 0 ? local_device : rank), mNcclComm(nullptr), mCmdBuf(std::make_unique<CommandBuffer>())
{
    // Use local_device for CUDA device selection (supports multi-node where device != global rank)
    CUDA_CHECK(cudaSetDevice(mLocalRank));
    ncclCheck(ncclCommInitRank(&mNcclComm, mWorld, *reinterpret_cast<const ncclUniqueId*>(nccl_id), mRank));

    // must be created _after_ we set the device
    mCommsStream = create_named_stream("nccl_stream");
    mCommsSync = create_named_event("nccl_sync");  // todo disable timing for max perf
}

#include <pthread.h>

/**
 * @brief Destructor that attempts to terminate NCCL without hanging the main thread.
 *
 * NCCL finalization can hang (notably with Python bindings). To localize hangs, NCCL teardown
 * is attempted in a helper thread with a timeout; on timeout, the future is intentionally leaked.
 *
 * Always destroys the internal CUDA event/stream (best-effort; never throws).
 */
NCCLCommunicator::~NCCLCommunicator() {
    // When used with the python bindings, ncclCommFinalize() can hang forever;
    // I haven't found a fix, so here we just make sure that the hang gets localized
    // to a helper thread (which we leak, but generally ~NCCLCommunicator is expected
    // to run at program shutdown anyway)
    auto terminate_future = std::async(std::launch::async, [this]() {
        this->terminate_nccl();
    });

    if (terminate_future.wait_for(std::chrono::seconds(2)) == std::future_status::timeout) {
        fprintf(stderr, "NCCL termination timed out, detaching\n");
        // this *will* leak resources, but at least we're not hanging forever
        new auto(std::move(terminate_future));
    }
    // Destructor must not throw, especially during stack unwinding.
    // Best-effort cleanup: report CUDA errors and continue.
    if (mCommsSync) {
        const cudaError_t st = cudaEventDestroy(mCommsSync);
        if (st != cudaSuccess) {
            fprintf(stderr, "WARNING: cudaEventDestroy(nccl_sync) failed: %s\n", cudaGetErrorString(st));
            fflush(stderr);
            (void)cudaGetLastError();
        }
        mCommsSync = nullptr;
    }
    if (mCommsStream) {
        const cudaError_t st = cudaStreamDestroy(mCommsStream);
        if (st != cudaSuccess) {
            fprintf(stderr, "WARNING: cudaStreamDestroy(nccl_stream) failed: %s\n", cudaGetErrorString(st));
            fflush(stderr);
            (void)cudaGetLastError();
        }
        mCommsStream = nullptr;
    }
}

/**
 * @brief Performs NCCL teardown (finalize/destroy or abort depending on state).
 *
 * If no exception is active and NCCL reports no async error, performs a "nice" shutdown:
 * synchronize streams/devices, finalize, then destroy communicator.
 * Otherwise aborts the communicator.
 *
 * @note Intended to be called from a helper thread in the destructor to avoid process-wide hangs.
 */
void NCCLCommunicator::terminate_nccl() {
    ncclResult_t result;
    ncclCheck(ncclCommGetAsyncError(mNcclComm, &result));
    // do "nice" shutdown if we're in a good state,
    // just abort if there is something weird going on.
    if (std::uncaught_exceptions() == 0 && result == ncclSuccess) {
        CUDA_CHECK(cudaStreamSynchronize(mCommsStream));
        CUDA_CHECK(cudaDeviceSynchronize());

        // Destroy EP sub-communicators before the global comm
        if (mWeightTransferComm) {
            ncclCheck(ncclCommFinalize(mWeightTransferComm));
            ncclCheck(ncclCommDestroy(mWeightTransferComm));
            mWeightTransferComm = nullptr;
        }
        if (mDPComm) {
            ncclCheck(ncclCommFinalize(mDPComm));
            ncclCheck(ncclCommDestroy(mDPComm));
            mDPComm = nullptr;
        }
        if (mEPComm) {
            ncclCheck(ncclCommFinalize(mEPComm));
            ncclCheck(ncclCommDestroy(mEPComm));
            mEPComm = nullptr;
        }

        ncclCheck(ncclCommFinalize(mNcclComm));
        ncclCheck(ncclCommDestroy(mNcclComm));
    } else {
        if (mWeightTransferComm) { ncclCommAbort(mWeightTransferComm); mWeightTransferComm = nullptr; }
        if (mDPComm) { ncclCommAbort(mDPComm); mDPComm = nullptr; }
        if (mEPComm) { ncclCommAbort(mEPComm); mEPComm = nullptr; }
        ncclCheck(ncclCommAbort(mNcclComm));
    }
}

/**
 * @brief Begin a transaction by specifying an event that indicates inputs are ready.
 *
 * @param ready CUDA event that must be completed before communication work may start.
 *
 * @throws std::runtime_error If the internal command buffer is not empty.
 */
void NCCLCommunicator::begin_transaction(cudaEvent_t ready) {
    if (!mCmdBuf->Commands.empty()) {
        throw std::runtime_error("start_comms: Buffer not empty");
    }
    mCmdBuf->Ready = ready;
}

/**
 * @brief Begin a transaction by recording a readiness event on a given stream.
 *
 * Records an internal event on @p wait_for_stream and uses it as the transaction "ready" marker.
 *
 * @param wait_for_stream Stream on which preceding compute producing the communication inputs was enqueued.
 */
void NCCLCommunicator::begin_transaction(cudaStream_t wait_for_stream) {
    CUDA_CHECK(cudaEventRecord(mCommsSync, wait_for_stream));
    begin_transaction(mCommsSync);
}

/**
 * @brief Visitor that executes buffered communication commands by dispatching to NCCLCommunicator methods.
 */
struct NCCLCommunicator::CommandVisitor {
    NCCLCommunicator* Comm;

    /**
     * @brief Execute a Gather command via NCCL all-gather (or derived override).
     * @param cmd Gather command containing src/dst pointers and byte size.
     */
    void operator()(CommandBuffer::Gather& cmd) const {
        Comm->gather_weight(cmd.Src, cmd.Dst, cmd.Bytes);
    }

    /**
     * @brief Execute a ScatterReduce command via NCCL reduce-scatter.
     * @param cmd ScatterReduce command containing dtype, tensor pointer, and element count.
     *
     * @throws std::runtime_error If @p cmd.DType is not supported.
     */
    void operator()(CommandBuffer::ScatterReduce& cmd) const {
        switch (cmd.DType) {
        case ETensorDType::FP32:
            Comm->scatter_grad(reinterpret_cast<float*>(cmd.Tensor), cmd.Elements);
            break;
        case ETensorDType::BF16:
            Comm->scatter_grad(reinterpret_cast<nv_bfloat16*>(cmd.Tensor), cmd.Elements);
            break;
        default:
            throw std::runtime_error("scatter: Unsupported dtype");
        }
    }

    /**
     * @brief Execute a Send command (point-to-point send).
     * @param cmd Send command containing source pointer, byte size, and target rank.
     */
    void operator()(CommandBuffer::Send& cmd) const {
        Comm->send(cmd.Tensor, cmd.Target, cmd.Bytes);
    }

    /**
     * @brief Execute a Recv command (point-to-point receive).
     * @param cmd Recv command containing destination pointer, byte size, and source rank.
     */
    void operator()(CommandBuffer::Recv& cmd) const {
        Comm->recv(cmd.Tensor, cmd.Source, cmd.Bytes);
    }

    /**
     * @brief Execute an AllReduce command via NCCL all-reduce with average.
     * @param cmd AllReduce command containing dtype, tensor pointer, and element count.
     *
     * @throws std::runtime_error If @p cmd.DType is not supported.
     */
    void operator()(CommandBuffer::AllReduce& cmd) const {
        Comm->all_reduce_avg_impl(cmd.Tensor, cmd.Elements, cmd.DType);
    }

};

/**
 * @brief Execute all scheduled commands in the current transaction and signal completion.
 *
 * Calls the transaction hooks (on_execute_transaction / on_finish_transaction), executes each buffered command,
 * and performs launch-queue throttling syncs before and after to avoid multi-rank deadlocks.
 *
 * Launch Queue Throttling Strategy:
 * - BEFORE transaction: Ensures all ranks are ready to begin enqueuing collective operations together.
 *   This prevents a fast rank from enqueueing its collectives while slower ranks are still processing
 *   previous work, which could lead to queue exhaustion before all ranks reach the collective barrier.
 *
 * - AFTER transaction: Ensures all ranks have completed enqueuing the collective operations before any
 *   rank continues with subsequent work. This prevents a fast rank from filling the launch queue with
 *   post-transaction kernels, which would block slower ranks from enqueuing future collectives.
 *
 * Note: In multi-process mode (MPI), _launch_queue_throttle_sync() is a no-op because each process
 * has its own independent launch queue. This mechanism only applies to multi-threaded mode where
 * all GPU worker threads share a single per-process CUDA launch queue.
 *
 * Optimization: Throttling is only applied when the transaction contains NCCL collective operations
 * (ScatterReduce or Gather), since only these operations have implicit global barriers that can
 * cause deadlocks. Point-to-point operations (Send/Recv) don't require throttling.
 *
 * @param signal CUDA event that will be recorded on the comms stream to signal completion of the transaction.
 *
 * @throws std::runtime_error Propagates errors from command execution or hooks.
 */
void NCCLCommunicator::execute_transaction(cudaEvent_t signal) {
    // Check if this transaction contains NCCL collective operations that require throttling
    bool has_collectives = std::any_of(mCmdBuf->Commands.begin(), mCmdBuf->Commands.end(),
        [](const auto& cmd) {
            return std::holds_alternative<CommandBuffer::ScatterReduce>(cmd) ||
                   std::holds_alternative<CommandBuffer::Gather>(cmd) ||
                   std::holds_alternative<CommandBuffer::AllReduce>(cmd);
        });

    // Synchronize CPU threads before enqueuing collective operations
    if (has_collectives) {
        _launch_queue_throttle_sync();
    }

    on_execute_transaction(*mCmdBuf);

    CommandVisitor visitor{this};
    for (auto& cmd: mCmdBuf->Commands) {
        std::visit(visitor, cmd);
    }

    on_finish_transaction(signal);

    // Synchronize CPU threads after enqueuing collective operations
    if (has_collectives) {
        _launch_queue_throttle_sync();
    }

    mCmdBuf->Commands.clear();
}

/**
 * @brief Schedule an in-place reduce-scatter of gradients for a tensor.
 *
 * @param tensor Tensor whose device buffer is reduced across ranks and scattered so each rank keeps its shard.
 *
 * @throws std::runtime_error If @p tensor.Data is null.
 */
void NCCLCommunicator::schedule_reduce_scatter(Tensor& tensor) {
    if (tensor.Data == nullptr) {
        throw std::runtime_error("scatter: Source tensor is null");
    }

    mCmdBuf->Commands.emplace_back(CommandBuffer::ScatterReduce{.DType = tensor.DType, .Tensor = tensor.Data, .Elements = tensor.nelem()});
}

/**
 * @brief Schedule an all-gather of a sharded tensor into a full target tensor.
 *
 * @param src Source shard (device pointer and dtype describe the local shard).
 * @param tgt Target full tensor receiving concatenated shards (device pointer must be valid).
 *
 * @throws std::runtime_error If source/target pointers are null or dtypes mismatch.
 */
void NCCLCommunicator::schedule_all_gather(const TensorShard& src, Tensor& tgt) {
    if (src.Data == nullptr) {
        throw std::runtime_error("gather: Source tensor is null");
    }

    if (tgt.Data == nullptr) {
        throw std::runtime_error("gather: Target tensor is null");
    }

    if (src.DType != tgt.DType) {
        fprintf(stderr, "[DEBUG] all_gather dtype mismatch: src.DType=%d tgt.DType=%d\n", (int)src.DType, (int)tgt.DType);
        throw std::runtime_error("gather: Mismatched dtypes");
    }

    mCmdBuf->Commands.emplace_back(CommandBuffer::Gather{.Src = src.Data, .Dst = tgt.Data, .Bytes = tgt.bytes()});
}

/**
 * @brief Schedule an in-place all-reduce with average for a tensor (for batching in transactions).
 *
 * @param tensor Tensor whose device buffer is all-reduced in-place.
 *
 * @throws std::runtime_error If @p tensor.Data is null.
 */
void NCCLCommunicator::schedule_all_reduce_avg(Tensor& tensor) {
    if (tensor.Data == nullptr) {
        throw std::runtime_error("schedule_all_reduce_avg: Tensor is null");
    }

    mCmdBuf->Commands.emplace_back(CommandBuffer::AllReduce{.DType = tensor.DType, .Tensor = tensor.Data, .Elements = tensor.nelem()});
}

/**
 * @brief All-reduce a single scalar loss value across ranks using average.
 *
 * @param loss Device pointer to a single float (input/output in-place).
 * @param stream CUDA stream to enqueue the NCCL all-reduce on.
 */
void NCCLCommunicator::reduce_loss(float* loss, cudaStream_t stream) {
    if (mWorld == 1) return;
    ncclCheck(ncclAllReduce(loss, loss, 1, ncclFloat, ncclAvg, mNcclComm, stream));
}

/**
 * @brief All-reduce an array of floats across ranks using maximum.
 *
 * @param values Device pointer to @p n floats (input/output in-place).
 * @param n Number of float elements.
 * @param stream CUDA stream to enqueue on; if null, uses the internal comms stream.
 */
void NCCLCommunicator::reduce_max(float* values, int n, cudaStream_t stream) {
    ncclCheck(ncclAllReduce(values, values, n, ncclFloat, ncclMax, mNcclComm, stream ? stream : mCommsStream));
}

/**
 * @brief All-reduce a single scalar value across ranks using sum (e.g., norm-squared accumulation).
 *
 * @param norm_squared Device pointer to a single float (input/output in-place).
 * @param stream CUDA stream to enqueue the NCCL all-reduce on.
 */
void NCCLCommunicator::reduce_norm(float* norm_squared, cudaStream_t stream) {
    ncclCheck(ncclAllReduce(norm_squared, norm_squared, 1, ncclFloat, ncclSum, mNcclComm, stream));
}

/**
 * @brief All-reduce INT32 values across ranks using sum.
 *
 * Used for aggregating counters like valid-token counts when masking is enabled.
 *
 * @param values Device pointer to @p n int32 values (input/output in-place).
 * @param n Number of int32 elements.
 * @param stream CUDA stream to enqueue the NCCL all-reduce on.
 */
void NCCLCommunicator::all_reduce_sum_int(int* values, int n, cudaStream_t stream) {
    if (mWorld == 1) return;
    ncclCheck(ncclAllReduce(values, values, n, ncclInt32, ncclSum, mNcclComm, stream));
}

// ============================================================================
// Expert Parallelism (EP) process groups
// ============================================================================

/**
 * @brief Initialize EP sub-communicators for 2D parallelism (DP x EP).
 *
 * Creates three NCCL sub-communicators:
 * - EP comm: ranks sharing data but owning different experts (size = ep_size)
 * - DP comm: ranks owning the same experts but processing different data (size = dp_size)
 * - Weight transfer comm: same ranks as EP, separate comm for overlap with A2A
 *
 * Rank mapping: ep_rank = global_rank % ep_size, dp_rank = global_rank / ep_size
 *
 * @param ep_size Number of EP ranks (must divide world_size; 1 = no EP, skip creation)
 *
 * @throws std::runtime_error If ep_size doesn't divide world_size.
 */
void NCCLCommunicator::init_ep_groups(int ep_size) {
    if (ep_size <= 1) {
        mEPSize = 1;
        mEPRank = 0;
        mDPSize = mWorld;
        mDPRank = mRank;
        return;
    }

    if (mWorld % ep_size != 0) {
        throw std::runtime_error(fmt::format(
            "init_ep_groups: ep_size ({}) must divide world_size ({})", ep_size, mWorld));
    }

    mEPSize = ep_size;
    mDPSize = mWorld / ep_size;
    mEPRank = mRank % ep_size;
    mDPRank = mRank / ep_size;

    // Generate unique NCCL IDs for each group via host all-gather.
    // Rank 0 of each group generates the ID, then broadcasts via all-gather.

    // EP groups: ranks {dp_rank * ep_size .. dp_rank * ep_size + ep_size - 1}
    // The "leader" of each EP group is the rank with ep_rank == 0, i.e. rank = dp_rank * ep_size
    ncclUniqueId ep_id{};
    if (mEPRank == 0) {
        ncclCheck(ncclGetUniqueId(&ep_id));
    }
    // Broadcast EP ID within EP group: all ranks in same EP group need the same ID.
    // Use host_all_gather to share, then pick the one from the group leader.
    auto all_ep_ids = host_all_gather(ep_id);
    int ep_leader = mDPRank * ep_size;  // global rank of EP group leader
    ep_id = all_ep_ids[ep_leader];

    ncclCheck(ncclCommInitRank(&mEPComm, mEPSize, ep_id, mEPRank));

    // DP groups: ranks {ep_rank, ep_rank + ep_size, ep_rank + 2*ep_size, ...}
    // The "leader" of each DP group is the rank with dp_rank == 0, i.e. rank = ep_rank
    ncclUniqueId dp_id{};
    if (mDPRank == 0) {
        ncclCheck(ncclGetUniqueId(&dp_id));
    }
    auto all_dp_ids = host_all_gather(dp_id);
    int dp_leader = mEPRank;  // global rank of DP group leader
    dp_id = all_dp_ids[dp_leader];

    ncclCheck(ncclCommInitRank(&mDPComm, mDPSize, dp_id, mDPRank));

    // Weight transfer comm: same group as EP, separate NCCL comm for overlap
    ncclUniqueId wt_id{};
    if (mEPRank == 0) {
        ncclCheck(ncclGetUniqueId(&wt_id));
    }
    auto all_wt_ids = host_all_gather(wt_id);
    wt_id = all_wt_ids[ep_leader];

    ncclCheck(ncclCommInitRank(&mWeightTransferComm, mEPSize, wt_id, mEPRank));

    if (mRank == 0) {
        fprintf(stderr, "[EP] Initialized EP groups: ep_size=%d, dp_size=%d, world_size=%d\n",
                mEPSize, mDPSize, mWorld);
    }
}

/**
 * @brief Variable-split all-to-all using grouped ncclSend/ncclRecv on the EP comm.
 *
 * NCCL has no native variable-split all-to-all. This implements it via:
 * ncclGroupStart(); for each peer: ncclSend + ncclRecv; ncclGroupEnd();
 *
 * @param send Source buffer, split contiguously per send_splits.
 * @param recv Destination buffer, split contiguously per recv_splits.
 * @param send_splits Array of ep_size() ints: elements to send to each EP peer.
 * @param recv_splits Array of ep_size() ints: elements to receive from each EP peer.
 * @param elem_size Size of each element in bytes.
 * @param stream CUDA stream for the NCCL operations.
 */
void NCCLCommunicator::all_to_all_single(const std::byte* send, std::byte* recv,
                                          const int* send_splits, const int* recv_splits,
                                          int elem_size, cudaStream_t stream) {
    if (!mEPComm) {
        throw std::runtime_error("all_to_all_single: EP comm not initialized (call init_ep_groups first)");
    }

    ncclCheck(ncclGroupStart());
    std::size_t send_offset = 0;
    std::size_t recv_offset = 0;
    for (int peer = 0; peer < mEPSize; ++peer) {
        std::size_t send_bytes = static_cast<std::size_t>(send_splits[peer]) * elem_size;
        std::size_t recv_bytes = static_cast<std::size_t>(recv_splits[peer]) * elem_size;
        if (send_bytes > 0) {
            ncclCheck(ncclSend(send + send_offset, send_bytes, ncclInt8, peer, mEPComm, stream));
        }
        if (recv_bytes > 0) {
            ncclCheck(ncclRecv(recv + recv_offset, recv_bytes, ncclInt8, peer, mEPComm, stream));
        }
        send_offset += send_bytes;
        recv_offset += recv_bytes;
    }
    ncclCheck(ncclGroupEnd());
}

/**
 * @brief All-reduce INT32 values across EP group using sum.
 *
 * Used for aggregating expert token counts across EP ranks.
 *
 * @param values Device pointer to int32 values (input/output in-place).
 * @param n Number of int32 elements.
 * @param stream CUDA stream.
 */
void NCCLCommunicator::all_reduce_sum_int_ep(int* values, int n, cudaStream_t stream) {
    if (!mEPComm || mEPSize <= 1) return;
    ncclCheck(ncclAllReduce(values, values, n, ncclInt32, ncclSum, mEPComm, stream));
}

/**
 * @brief All-reduce tensor in-place using average on the DP comm.
 *
 * Used for gradient averaging across data-parallel ranks (expert weight gradients
 * only need to be synchronized within the DP group, not globally).
 *
 * @param tensor Tensor to all-reduce in-place.
 * @param stream CUDA stream.
 */
void NCCLCommunicator::all_reduce_avg_dp(Tensor& tensor, cudaStream_t stream) {
    if (!mDPComm || mDPSize <= 1) return;

    ncclDataType_t nccl_dtype;
    switch (tensor.DType) {
        case ETensorDType::FP32:
            nccl_dtype = ncclFloat;
            break;
        case ETensorDType::BF16:
            nccl_dtype = ncclBfloat16;
            break;
        case ETensorDType::FP16:
            nccl_dtype = ncclFloat16;
            break;
        default:
            throw std::runtime_error(fmt::format(
                "NCCLCommunicator::all_reduce_avg_dp: unsupported tensor dtype {}",
                dtype_to_str(tensor.DType)));
    }

    ncclCheck(ncclAllReduce(tensor.Data, tensor.Data, tensor.nelem(), nccl_dtype, ncclAvg, mDPComm, stream));
}

void NCCLCommunicator::send_wt(const void* data, std::size_t bytes, int peer, cudaStream_t stream) {
    if (!mWeightTransferComm) {
        throw std::runtime_error("send_wt: weight_transfer_comm not initialized");
    }
    if (bytes > 0) {
        ncclCheck(ncclSend(data, bytes, ncclInt8, peer, mWeightTransferComm, stream));
    }
}

void NCCLCommunicator::recv_wt(void* data, std::size_t bytes, int peer, cudaStream_t stream) {
    if (!mWeightTransferComm) {
        throw std::runtime_error("recv_wt: weight_transfer_comm not initialized");
    }
    if (bytes > 0) {
        ncclCheck(ncclRecv(data, bytes, ncclInt8, peer, mWeightTransferComm, stream));
    }
}

void NCCLCommunicator::weight_transfer_group_start() {
    ncclCheck(ncclGroupStart());
}

void NCCLCommunicator::weight_transfer_group_end() {
    ncclCheck(ncclGroupEnd());
}

/**
 * @brief All-reduce tensor data in-place using average.
 *
 * Used for gradient averaging in data parallelism (e.g., LoRA training).
 * Supports FP32 and BF16 tensors.
 *
 * @param tensor Tensor to all-reduce in-place (must be on device).
 * @param stream CUDA stream to enqueue the NCCL all-reduce on.
 *
 * @throws std::runtime_error if tensor dtype is not supported.
 */
void NCCLCommunicator::all_reduce_avg(Tensor& tensor, cudaStream_t stream) {
    if (mWorld == 1) return; // No-op for single GPU

    ncclDataType_t nccl_dtype;
    switch (tensor.DType) {
        case ETensorDType::FP32:
            nccl_dtype = ncclFloat;
            break;
        case ETensorDType::BF16:
            nccl_dtype = ncclBfloat16;
            break;
        case ETensorDType::FP16:
            nccl_dtype = ncclFloat16;
            break;
        default:
            fprintf(stderr, "[DEBUG] all_reduce_avg ERROR: unsupported dtype %d\n", (int)tensor.DType);
            throw std::runtime_error(fmt::format(
                "NCCLCommunicator::all_reduce_avg: unsupported tensor dtype {}",
                dtype_to_str(tensor.DType)));
    }

    ncclCheck(ncclAllReduce(tensor.Data, tensor.Data, tensor.nelem(), nccl_dtype, ncclAvg, mNcclComm, stream));
}

/**
 * @brief Internal implementation for all-reduce with average (used by transaction visitor).
 *
 * Executes on the internal comms stream within an NCCL group.
 *
 * @param data Device pointer to tensor data (input/output in-place).
 * @param elements Number of elements in the tensor.
 * @param dtype Data type of the tensor.
 *
 * @throws std::runtime_error if dtype is not supported.
 */
void NCCLCommunicator::all_reduce_avg_impl(std::byte* data, std::size_t elements, ETensorDType dtype) {
    ncclDataType_t nccl_dtype;
    switch (dtype) {
        case ETensorDType::FP32:
            nccl_dtype = ncclFloat;
            break;
        case ETensorDType::BF16:
            nccl_dtype = ncclBfloat16;
            break;
        case ETensorDType::FP16:
            nccl_dtype = ncclFloat16;
            break;
        default:
            throw std::runtime_error(fmt::format(
                "NCCLCommunicator::all_reduce_avg_impl: unsupported dtype {}",
                static_cast<int>(dtype)));
    }

    ncclCheck(ncclAllReduce(data, data, elements, nccl_dtype, ncclAvg, mNcclComm, mCommsStream));
}

/**
 * @brief Reduce-scatter FP32 gradients across ranks using average.
 *
 * Input is interpreted as a full buffer of @p size elements; output shard is written to the local shard region.
 *
 * @param value Device pointer to the full buffer (input) and shard region (output).
 * @param size Total number of float elements in the full buffer (must be divisible by world size).
 */
void NCCLCommunicator::scatter_grad(float* value, std::size_t size) {
    assert(size % mWorld == 0);
    size_t shard_size = size / mWorld;
    ptrdiff_t shard_offset = (ptrdiff_t)shard_size * mRank;
    ncclCheck(ncclReduceScatter(
        value, value + shard_offset,
        shard_size,
        ncclFloat, ncclAvg,
        mNcclComm, mCommsStream
    ));
}

/**
 * @brief Reduce-scatter BF16 gradients across ranks using average.
 *
 * Input is interpreted as a full buffer of @p size elements; output shard is written to the local shard region.
 *
 * @param value Device pointer to the full buffer (input) and shard region (output).
 * @param size Total number of BF16 elements in the full buffer (must be divisible by world size).
 */
void NCCLCommunicator::scatter_grad(nv_bfloat16* value, std::size_t size) {
    assert(size % mWorld == 0);
    size_t shard_size = size / mWorld;
    ptrdiff_t shard_offset = (ptrdiff_t)shard_size * mRank;
    ncclCheck(ncclReduceScatter(
        value, value + shard_offset,
        shard_size,
        ncclBfloat16, ncclAvg,
        mNcclComm, mCommsStream
    ));
}

/**
 * @brief All-gather a sharded weight buffer into a full buffer.
 *
 * @param src Device pointer to the local shard (or full buffer if in-place).
 * @param dst Device pointer to the full destination buffer.
 * @param size Total byte size of the full buffer (must be divisible by world size).
 *
 * @note If @p src == @p dst, performs an in-place all-gather by offsetting @p src to the local shard.
 */
void NCCLCommunicator::gather_weight(const std::byte* src, std::byte* dst,  std::size_t size) {
    assert(size % mWorld == 0);
    size_t shard_size = size / mWorld;
    if(src == dst) {
        src += shard_size * mRank; // in-place
    }
    ncclCheck(ncclAllGather(src,
                            dst,
                            shard_size, ncclInt8,
                            mNcclComm, mCommsStream));
}

/**
 * @brief Enqueue a point-to-point send of raw bytes.
 *
 * @param src Device pointer to bytes to send.
 * @param peer Destination rank.
 * @param size Number of bytes to send.
 */
void NCCLCommunicator::send(const std::byte* src, int peer, std::size_t size) {
    ncclCheck(ncclSend(src, size, ncclInt8, peer, mNcclComm, mCommsStream));
}

/**
 * @brief Enqueue a point-to-point receive of raw bytes.
 *
 * @param dst Device pointer to receive buffer.
 * @param peer Source rank.
 * @param size Number of bytes to receive.
 */
void NCCLCommunicator::recv(std::byte* dst, int peer, std::size_t size) {
    ncclCheck(ncclRecv(dst, size, ncclInt8, peer, mNcclComm, mCommsStream));
}

/**
 * @brief Make a compute stream wait until the communicator sync event is complete.
 *
 * @param compute_stream CUDA stream that should wait on the internal comms sync event.
 */
void NCCLCommunicator::wait_on_comms(cudaStream_t compute_stream) {
    CUDA_CHECK(cudaStreamWaitEvent(compute_stream, mCommsSync, 0));
}

/**
 * @brief Schedule a destructive all-to-all rotation using explicit send/recv pairs.
 *
 * Splits @p tensor into @c world_size shards and schedules point-to-point exchanges such that each rank
 * sends shard slices to peers and overwrites local storage positions with received shards.
 *
 * @param tensor Tensor whose device storage is partitioned and exchanged; contents are overwritten in-place.
 */
void NCCLCommunicator::schedule_destructive_all_to_all(const Tensor& tensor) {
    std::size_t shard_size = (ptrdiff_t)tensor.bytes() / world_size();
    for(int n = 1; n < world_size(); ++n) {
        int dst = (n + rank()) % world_size();
        int src = (rank() - n + world_size()) % world_size();
        int store = (rank() + n - 1 + world_size()) % world_size();
        mCmdBuf->Commands.emplace_back(CommandBuffer::Send{
            .Tensor = tensor.Data + dst * shard_size,
            .Bytes = shard_size,
            .Target = dst
            }
            );
        mCmdBuf->Commands.emplace_back(CommandBuffer::Recv{
            .Tensor = tensor.Data + store * shard_size,
            .Bytes = shard_size,
            .Source = src
        });
    }
}

// ============================================================================
// Unified Threaded Communicator (with optional MPI for multi-node)
// ============================================================================

/**
 * @brief Thread-based NCCL communicator variant supporting single-node and multi-node.
 *
 * Single-node: Uses std::barrier for CPU synchronization and shared memory for host data exchange.
 * Multi-node: When MPI is available and multiple nodes detected, uses MPI for inter-node
 * coordination while maintaining threaded intra-node communication.
 *
 * Can optionally replace some NCCL ops (all-gather and/or send/recv) with device-to-device
 * memcpy coordinated via barriers.
 */
class NCCLCommunicatorImpl : public NCCLCommunicator {
public:
    struct SharedState {
        std::unique_ptr<std::barrier<>> Barrier;
        std::vector<std::byte*> Buffer;     // one pointer per thread
        std::vector<std::exception_ptr> Exceptions;
        std::mutex Mutex;
        int NumNodes = 1;                   // number of nodes (1 = single node)
        int NodeRank = 0;                   // this node's rank
        int LocalGPUs = 0;                  // GPUs per node

        // GPU staging buffers for host gather/all-gather in multi-node mode (Ray)
        // Allocated once at init, reused for all operations
        void* d_gather_send = nullptr;      // size = kMaxGatherObjectSize * LocalGPUs
        void* d_gather_recv = nullptr;      // size = kMaxGatherObjectSize * world_size
        static constexpr size_t kMaxGatherObjectSize = 4096;

        // Device buffer for NCCL barrier (ncclAllReduce requires device memory)
        char* d_barrier_buf = nullptr;      // 1 byte for barrier AllReduce

        // Node master NCCL communicator for cross-node operations in Ray mode
        ncclComm_t NodeMasterComm = nullptr;

        ~SharedState() {
            if (d_gather_send) cudaFree(d_gather_send);
            if (d_gather_recv) cudaFree(d_gather_recv);
            if (d_barrier_buf) cudaFree(d_barrier_buf);
            if (NodeMasterComm) ncclCommDestroy(NodeMasterComm);
        }
    };

    /**
     * @brief Construct a communicator for a given local rank.
     *
     * @param local_rank Thread/rank index within this node (also used as CUDA device index).
     * @param global_rank Global rank across all nodes.
     * @param global_world Total number of GPUs across all nodes.
     * @param memcpy_allgather If true, all-gather may be implemented via host-coordinated D2D memcpy.
     * @param memcpy_send_recv If true, point-to-point send/recv may be implemented via host-coordinated D2D memcpy.
     * @param nccl_id Pointer to shared ncclUniqueId used for ncclCommInitRank.
     * @param state Shared synchronization/exchange state shared by all local ranks.
     */
    NCCLCommunicatorImpl(int local_rank, int global_rank, int global_world,
                         bool memcpy_allgather, bool memcpy_send_recv,
                         const void* nccl_id, std::shared_ptr<SharedState> state);

    /**
     * @brief Drops out of the shared barrier on destruction (if present).
     */
    ~NCCLCommunicatorImpl() override;

    /**
     * @brief Barrier across all ranks (local threads + inter-node MPI if multi-node).
     */
    void barrier() override;

    /**
     * @brief Throttle launch queue by synchronizing local CPU threads.
     */
    void _launch_queue_throttle_sync() override;

    /**
     * @brief Gather fixed-size host byte blobs onto global rank 0.
     */
    void gather_bytes_host(std::byte* recv, const std::byte* object, std::size_t size) override;

    /**
     * @brief All-gather fixed-size host byte blobs onto all ranks.
     */
    void all_gather_bytes_host(std::byte* recv, const std::byte* object, std::size_t size) override;

    /**
     * @brief All-gather weights either via NCCL or via D2D memcpy depending on configuration.
     */
    void gather_weight(const std::byte* src, std::byte* tgt, std::size_t size) override;

    /**
     * @brief Send bytes to @p peer using NCCL or deferred memcpy emulation.
     */
    void send(const std::byte* src, int peer, std::size_t size) override;

    /**
     * @brief Receive bytes from @p peer using NCCL or deferred memcpy emulation.
     */
    void recv(std::byte* tgt, int peer, std::size_t size) override;

    /**
     * @brief Transaction hook: decide whether this transaction uses NCCL and/or memcpy emulation.
     */
    void on_execute_transaction(const NCCLCommunicator::CommandBuffer& cmd) override;

    /**
     * @brief Transaction hook: complete NCCL group and/or perform memcpy-based send/recv matching.
     */
    void on_finish_transaction(cudaEvent_t signal) override;

    [[nodiscard]] int local_rank() const { return mLocalRank; }
    [[nodiscard]] bool is_node_master() const { return mLocalRank == 0; }

    void local_barrier() override;

private:

    std::shared_ptr<SharedState> mShare;
    int mLocalRank;
    bool mAllGatherUseMemcpy = false;
    bool mSendRecvUseMemcpy = true;

    // transaction status
    bool mUseMemcpy;
    bool mUseNCCL;

    struct sSendParams {
        const std::byte* Data;
        std::size_t Size;
        int Peer;
        bool Matched = false;
    };
    std::vector<sSendParams> mSendParams;

    struct sRecvParams {
        std::byte* Data;
        std::size_t Size;
        int Peer;
    };
    std::vector<sRecvParams> mRecvParams;
};

NCCLCommunicatorImpl::NCCLCommunicatorImpl(
    int local_rank, int global_rank, int global_world,
    bool memcpy_allgather, bool memcpy_send_recv,
    const void* nccl_id, std::shared_ptr<SharedState> state)
    : NCCLCommunicator(global_rank, global_world, nccl_id, local_rank)  // Pass local_rank as device
    , mShare(std::move(state))
    , mLocalRank(local_rank)
    , mAllGatherUseMemcpy(memcpy_allgather)
    , mSendRecvUseMemcpy(memcpy_send_recv)
{
}

NCCLCommunicatorImpl::~NCCLCommunicatorImpl() {
    if(mShare && mShare->Barrier) {
        mShare->Barrier->arrive_and_drop();
    }
}

void NCCLCommunicatorImpl::local_barrier() {
    mShare->Barrier->arrive_and_wait();
}

void NCCLCommunicatorImpl::barrier() {
    // Two-level barrier for multi-node
    local_barrier();

    if (mShare->NumNodes > 1) {
        // NCCL-based barrier: all-reduce a single byte on device memory
        // Only local_rank 0 on each node participates in cross-node NCCL
        if (is_node_master() && mShare->NodeMasterComm && mShare->d_barrier_buf) {
            ncclCheck(ncclAllReduce(mShare->d_barrier_buf, mShare->d_barrier_buf, 1, ncclChar, ncclSum,
                                    mShare->NodeMasterComm, stream()));
            CUDA_CHECK(cudaStreamSynchronize(stream()));
        }
        local_barrier();  // Sync local threads after cross-node barrier
    }
}

void NCCLCommunicatorImpl::_launch_queue_throttle_sync() {
    // For multi-node training: synchronize local threads to prevent launch queue exhaustion
    // For single-node training: skip barrier to avoid unnecessary CPU synchronization overhead
    if (mShare->NumNodes > 1) {
        local_barrier();
    }
}

void NCCLCommunicatorImpl::gather_bytes_host(std::byte* recv, const std::byte* object, std::size_t size) {
    if (mShare->NumNodes > 1) {
        // NCCL-based gather for multi-node
        // Only node masters (local_rank == 0) participate in cross-node NCCL
        int local_gpus = mShare->LocalGPUs;

        if (size > NCCLCommunicatorImpl::SharedState::kMaxGatherObjectSize) {
            throw std::runtime_error(fmt::format("gather_bytes_host: object size {} exceeds max {}",
                                                 size, NCCLCommunicatorImpl::SharedState::kMaxGatherObjectSize));
        }

        // Step 1: Local gather via shared memory (same as single-node)
        std::vector<std::byte> local_gather(local_gpus * size);
        if (is_node_master()) {
            mShare->Buffer[0] = local_gather.data();
        }
        local_barrier();
        std::memcpy(mShare->Buffer[0] + mLocalRank * size, object, size);
        local_barrier();

        // Step 2: NCCL gather from node masters to global rank 0
        if (is_node_master() && mShare->NodeMasterComm) {
            // Copy local data to GPU staging buffer
            CUDA_CHECK(cudaMemcpyAsync(mShare->d_gather_send, local_gather.data(),
                                       local_gpus * size, cudaMemcpyHostToDevice, stream()));

            // Use NCCL all-gather (no native gather, so gather = allgather + discard on non-root)
            ncclCheck(ncclAllGather(mShare->d_gather_send, mShare->d_gather_recv,
                                    local_gpus * size, ncclChar, mShare->NodeMasterComm, stream()));

            // Copy result back to host (only rank 0 needs it)
            if (rank() == 0) {
                CUDA_CHECK(cudaMemcpyAsync(recv, mShare->d_gather_recv,
                                           world_size() * size, cudaMemcpyDeviceToHost, stream()));
            }
            CUDA_CHECK(cudaStreamSynchronize(stream()));
        }
        local_barrier();
    } else {
        // Single-node: simple shared memory gather
        if (mLocalRank == 0) {
            mShare->Buffer[0] = recv;
        }
        local_barrier();
        std::memcpy(mShare->Buffer[0] + mLocalRank * size, object, size);
        local_barrier();
        if (mLocalRank == 0) {
            mShare->Buffer[0] = nullptr;
        }
    }
}

void NCCLCommunicatorImpl::all_gather_bytes_host(std::byte* recv, const std::byte* object, std::size_t size) {
    if (mShare->NumNodes > 1) {
        // NCCL-based all-gather for multi-node
        int local_gpus = mShare->LocalGPUs;
        int total_gpus = world_size();

        if (size > NCCLCommunicatorImpl::SharedState::kMaxGatherObjectSize) {
            throw std::runtime_error(fmt::format("all_gather_bytes_host: object size {} exceeds max {}",
                                                 size, NCCLCommunicatorImpl::SharedState::kMaxGatherObjectSize));
        }

        // Step 1: Local all-gather via shared memory
        local_barrier();
        mShare->Buffer[mLocalRank] = const_cast<std::byte*>(object);
        local_barrier();

        std::vector<std::byte> local_data(local_gpus * size);
        for (int i = 0; i < local_gpus; ++i) {
            std::memcpy(local_data.data() + i * size, mShare->Buffer[i], size);
        }
        local_barrier();

        // Step 2: NCCL all-gather from node masters
        if (is_node_master() && mShare->NodeMasterComm) {
            CUDA_CHECK(cudaMemcpyAsync(mShare->d_gather_send, local_data.data(),
                                       local_gpus * size, cudaMemcpyHostToDevice, stream()));

            ncclCheck(ncclAllGather(mShare->d_gather_send, mShare->d_gather_recv,
                                    local_gpus * size, ncclChar, mShare->NodeMasterComm, stream()));

            CUDA_CHECK(cudaMemcpyAsync(recv, mShare->d_gather_recv,
                                       total_gpus * size, cudaMemcpyDeviceToHost, stream()));
            CUDA_CHECK(cudaStreamSynchronize(stream()));
        }
        local_barrier();

        // Step 3: Broadcast result to all local threads
        if (is_node_master()) {
            mShare->Buffer[0] = recv;
        }
        local_barrier();
        if (!is_node_master()) {
            std::memcpy(recv, mShare->Buffer[0], total_gpus * size);
        }
        local_barrier();
        mShare->Buffer[mLocalRank] = nullptr;
    } else {
        // Single-node: simple shared memory all-gather
        local_barrier();
        mShare->Buffer[mLocalRank] = const_cast<std::byte*>(object);
        local_barrier();
        for (int i = 0; i < world_size(); ++i) {
            std::memcpy(recv + i * size, mShare->Buffer[i], size);
        }
        local_barrier();
        mShare->Buffer[mLocalRank] = nullptr;
    }
}

void NCCLCommunicatorImpl::send(const std::byte* src, int peer, std::size_t size) {
    if (!mSendRecvUseMemcpy) {
        NCCLCommunicator::send(src, peer, size);
    } else {
        mSendParams.emplace_back(sSendParams{src, size, peer});
    }
}

void NCCLCommunicatorImpl::recv(std::byte* tgt, int peer, std::size_t size) {
    if (!mSendRecvUseMemcpy) {
        NCCLCommunicator::recv(tgt, peer, size);
    } else {
        mRecvParams.emplace_back(sRecvParams{tgt, size, peer});
    }
}

void NCCLCommunicatorImpl::gather_weight(const std::byte* src, std::byte* tgt, std::size_t size) {
    if (mAllGatherUseMemcpy) {
        auto wgt_list = host_all_gather(src);
        std::size_t shard_size = size / world_size();
        for (int i = 0; i < world_size(); ++i) {
            if (tgt + shard_size * i != wgt_list[i]) {
                CUDA_CHECK(cudaMemcpyAsync(tgt + shard_size * i, wgt_list[i], shard_size, cudaMemcpyDeviceToDevice, stream()));
            }
        }
    } else {
        NCCLCommunicator::gather_weight(src, tgt, size);
    }
}

void NCCLCommunicatorImpl::on_execute_transaction(const NCCLCommunicator::CommandBuffer& cmd) {
    mUseMemcpy = false;
    mUseNCCL = false;
    for (auto& c : cmd.Commands) {
        if (std::holds_alternative<CommandBuffer::ScatterReduce>(c)) {
            mUseNCCL = true;
        }
        if (std::holds_alternative<CommandBuffer::AllReduce>(c)) {
            mUseNCCL = true;
        }
        if (std::holds_alternative<CommandBuffer::Gather>(c)) {
            if (!mAllGatherUseMemcpy) mUseNCCL = true;
            if (mAllGatherUseMemcpy) mUseMemcpy = true;
        }
        if (std::holds_alternative<CommandBuffer::Send>(c)) {
            if (!mSendRecvUseMemcpy) mUseNCCL = true;
            if (mSendRecvUseMemcpy) mUseMemcpy = true;
        }
    }

    assert(mUseNCCL || mUseMemcpy);

    if (mUseMemcpy) {
        // ensure every worker has set-up commands.Ready to the most recent version
        local_barrier();
        // get the ready event from all workers
        auto event_list = host_all_gather(cmd.Ready);
        // make sure to block the comms thread until the data is ready on every worker
        for (auto event : event_list) {
            CUDA_CHECK(cudaStreamWaitEvent(stream(), event, 0));
        }
    }

    if (mUseNCCL) {
        CUDA_CHECK(cudaStreamWaitEvent(stream(), cmd.Ready, 0));
        ncclCheck(ncclGroupStart());
    }
}

void NCCLCommunicatorImpl::on_finish_transaction(cudaEvent_t signal) {
    if (!mRecvParams.empty()) {
        // get send-queues from peers
        std::vector<std::vector<sSendParams>*> send_params = host_all_gather(&mSendParams);
        std::vector<cudaEvent_t> sync_events = host_all_gather(signal);
        // ok, now iterate all recv's
        for (auto& recv_param : mRecvParams) {
            // find matching send
            for (auto& send : *send_params.at(recv_param.Peer)) {
                if (send.Peer != rank() || send.Matched) continue;
                // copy data
                if (recv_param.Size != send.Size) {
                    throw std::runtime_error("Size mismatch in recv/send");
                }
                CUDA_CHECK(cudaMemcpyAsync(recv_param.Data, send.Data, recv_param.Size, cudaMemcpyDeviceToDevice, stream()));
                send.Matched = true;
                break;
            }

            CUDA_CHECK(cudaEventRecord(signal, stream()));
            local_barrier();  // assumes _all_ workers have the same number of receives!
            for (int j = 0; j < world_size(); ++j) {
                if (j != rank()) {
                    CUDA_CHECK(cudaStreamWaitEvent(stream(), sync_events[j], 0));
                }
            }
        }

        local_barrier();
        mRecvParams.clear();
        mSendParams.clear();
    }
    if (mUseNCCL) {
        ncclCheck(ncclGroupEnd());
    }

    CUDA_CHECK(cudaEventRecord(signal, stream()));
}

// ============================================================================
// Thread Pack for managing worker threads
// ============================================================================

class CommunicatorThreadsPackImpl : public CommunicatorThreadsPack {
public:
    CommunicatorThreadsPackImpl(std::vector<std::jthread> threads,
                                std::shared_ptr<NCCLCommunicatorImpl::SharedState> state)
        : mThreads(std::move(threads)), mState(std::move(state)) {}

    ~CommunicatorThreadsPackImpl() override {
        try {
            join_impl();
        } catch (...) {
            // Swallow in destructor
        }
    }

    void join() override {
        join_impl();
    }

    bool has_exception() const override {
        std::lock_guard<std::mutex> lock(mState->Mutex);
        for (size_t t = 0; t < mThreads.size(); ++t) {
            if (mState->Exceptions[t]) {
                return true;
            }
        }
        return false;
    }

private:
    void join_impl() {
        check_exceptions();
        for (auto& t : mThreads) {
            if (t.joinable()) {
                t.join();
            }
        }
        check_exceptions();
    }

    void check_exceptions() {
        std::lock_guard<std::mutex> lock(mState->Mutex);
        for (size_t t = 0; t < mThreads.size(); ++t) {
            if (auto error = mState->Exceptions[t]; error) {
                fprintf(stderr, "Thread %zu exited with uncaught exception\n", t);
                fflush(stderr);
                mState->Exceptions[t] = nullptr;
                std::rethrow_exception(error);
            }
        }
    }

    std::vector<std::jthread> mThreads;
    std::shared_ptr<NCCLCommunicatorImpl::SharedState> mState;
};

// ============================================================================
// Main Entry Points
// ============================================================================

namespace {

/**
 * @brief Helper to launch communicator threads for single-node training.
 *
 * @param ngpus Number of local GPUs.
 * @param memcpy_allgather Enable memcpy-based all-gather emulation.
 * @param memcpy_send_recv Enable memcpy-based send/recv emulation.
 * @param work Callable invoked once per GPU with that GPU's communicator.
 * @return Thread pack for caller to manage.
 */
std::unique_ptr<CommunicatorThreadsPackImpl>
launch_communicators_impl(int ngpus, bool memcpy_allgather, bool memcpy_send_recv,
                          std::function<void(NCCLCommunicator& comm)> work) {
    // Detect available GPUs
    int gpus_available = 0;
    CUDA_CHECK(cudaGetDeviceCount(&gpus_available));
    if (ngpus == 0) {
        ngpus = gpus_available;
    }
    if (ngpus > gpus_available) {
        throw std::runtime_error(fmt::format("Requested {} GPUs, but only {} available", ngpus, gpus_available));
    }

    // Generate NCCL unique ID
    ncclUniqueId nccl_id;
    ncclCheck(ncclGetUniqueId(&nccl_id));

    // Create shared state for local threads
    auto shared_state = std::make_shared<NCCLCommunicatorImpl::SharedState>();
    shared_state->Barrier = std::make_unique<std::barrier<>>(ngpus);
    shared_state->Buffer.resize(ngpus);
    shared_state->Exceptions.resize(ngpus);
    shared_state->NumNodes = 1;
    shared_state->NodeRank = 0;
    shared_state->LocalGPUs = ngpus;

    // Launch worker threads
    std::vector<std::jthread> threads;
    threads.reserve(ngpus);

    for (int local_rank = 0; local_rank < ngpus; ++local_rank) {
        threads.emplace_back([=, &work]() {
            try {
                if (!set_cpu_affinity()) {
                    fprintf(stderr, "WARNING: Failed to set CPU affinity for local rank %d\n", local_rank);
                }

                NCCLCommunicatorImpl comm(local_rank, local_rank, ngpus,
                                          memcpy_allgather, memcpy_send_recv,
                                          &nccl_id, shared_state);
                work(comm);
                shared_state->Barrier->arrive_and_wait();
            } catch (...) {
                std::lock_guard<std::mutex> lock(shared_state->Mutex);
                shared_state->Exceptions[local_rank] = std::current_exception();
            }
        });
    }

    return std::make_unique<CommunicatorThreadsPackImpl>(std::move(threads), shared_state);
}

} // anonymous namespace

/**
 * @brief Run distributed training with one thread per local GPU (blocking).
 */
void NCCLCommunicator::run_communicators(int ngpus, bool memcpy_allgather, bool memcpy_send_recv,
                                          std::function<void(NCCLCommunicator& comm)> work) {
    auto pack = launch_communicators_impl(ngpus, memcpy_allgather, memcpy_send_recv, std::move(work));
    pack->join();
}

/**
 * @brief Launch communicator threads and return a joinable pack (non-blocking).
 *
 * For single-node operation. Use launch_communicators_multinode() for multi-node via Ray.
 */
std::unique_ptr<CommunicatorThreadsPack> NCCLCommunicator::launch_communicators(
    int ngpus, bool memcpy_allgather, bool memcpy_send_recv,
    std::function<void(NCCLCommunicator& comm)> work) {

    return launch_communicators_impl(ngpus, memcpy_allgather, memcpy_send_recv, std::move(work));
}

/**
 * @brief Generate a new NCCL unique ID.
 */
std::array<std::byte, 128> NCCLCommunicator::generate_nccl_id() {
    ncclUniqueId id;
    ncclCheck(ncclGetUniqueId(&id));
    std::array<std::byte, 128> result;
    static_assert(sizeof(ncclUniqueId) == 128, "ncclUniqueId size mismatch");
    std::memcpy(result.data(), &id, 128);
    return result;
}

/**
 * @brief Launch communicator threads with externally-provided NCCL IDs (for Ray multi-node).
 *
 * Similar to launch_communicators_impl but:
 * - Uses externally-provided nccl_id and node_master_nccl_id (coordinated via Ray)
 * - Uses provided node_rank, num_nodes
 * - Computes global_rank = node_rank * ngpus + local_rank
 * - Computes global_world = num_nodes * ngpus
 * - Initializes node master communicator and staging buffers for cross-node operations
 */
std::unique_ptr<CommunicatorThreadsPack> NCCLCommunicator::launch_communicators_multinode(
    int ngpus,
    int node_rank,
    int num_nodes,
    const void* nccl_id,
    const void* node_master_nccl_id,
    bool memcpy_allgather,
    bool memcpy_send_recv,
    std::function<void(NCCLCommunicator& comm)> work) {

    // Detect available GPUs
    int gpus_available = 0;
    CUDA_CHECK(cudaGetDeviceCount(&gpus_available));
    if (ngpus == 0) {
        ngpus = gpus_available;
    }
    if (ngpus > gpus_available) {
        throw std::runtime_error(fmt::format("Requested {} GPUs, but only {} available", ngpus, gpus_available));
    }

    // Validate parameters
    if (node_rank < 0 || node_rank >= num_nodes) {
        throw std::runtime_error(fmt::format("Invalid node_rank {} for num_nodes {}", node_rank, num_nodes));
    }
    if (num_nodes < 1) {
        throw std::runtime_error(fmt::format("Invalid num_nodes {}", num_nodes));
    }
    if (nccl_id == nullptr) {
        throw std::runtime_error("nccl_id cannot be null");
    }
    if (node_master_nccl_id == nullptr) {
        throw std::runtime_error("node_master_nccl_id cannot be null");
    }

    int global_world = num_nodes * ngpus;

    if (node_rank == 0) {
        fprintf(stderr, "Ray multi-node training: %d nodes, %d GPUs per node, %d total GPUs\n",
                num_nodes, ngpus, global_world);
    }

    // Create shared state for local threads
    auto shared_state = std::make_shared<NCCLCommunicatorImpl::SharedState>();
    shared_state->Barrier = std::make_unique<std::barrier<>>(ngpus);
    shared_state->Buffer.resize(ngpus);
    shared_state->Exceptions.resize(ngpus);
    shared_state->NumNodes = num_nodes;
    shared_state->NodeRank = node_rank;
    shared_state->LocalGPUs = ngpus;

    // Allocate staging buffers on device 0 for host gather/all-gather operations
    CUDA_CHECK(cudaSetDevice(0));
    size_t send_size = NCCLCommunicatorImpl::SharedState::kMaxGatherObjectSize * ngpus;
    size_t recv_size = NCCLCommunicatorImpl::SharedState::kMaxGatherObjectSize * global_world;
    CUDA_CHECK(cudaMalloc(&shared_state->d_gather_send, send_size));
    CUDA_CHECK(cudaMalloc(&shared_state->d_gather_recv, recv_size));

    // Allocate device buffer for NCCL barrier (ncclAllReduce requires device memory)
    CUDA_CHECK(cudaMalloc(&shared_state->d_barrier_buf, sizeof(char)));
    CUDA_CHECK(cudaMemset(shared_state->d_barrier_buf, 0, sizeof(char)));

    // NOTE: NodeMasterComm is initialized inside the worker threads by local_rank=0
    // because ncclCommInitRank is a collective operation that must be called
    // simultaneously by all participating ranks (one per node). Moving this to the
    // main thread would cause a deadlock since nodes may reach this point at different times.

    // Launch worker threads
    std::vector<std::jthread> threads;
    threads.reserve(ngpus);

    for (int local_rank = 0; local_rank < ngpus; ++local_rank) {
        int global_rank = node_rank * ngpus + local_rank;

        threads.emplace_back([=, &work, nccl_id_copy = nccl_id, node_master_nccl_id_copy = node_master_nccl_id]() {
            try {
                if (!set_cpu_affinity()) {
                    fprintf(stderr, "WARNING: Failed to set CPU affinity for local rank %d\n", local_rank);
                }

                NCCLCommunicatorImpl comm(local_rank, global_rank, global_world,
                                          memcpy_allgather, memcpy_send_recv,
                                          nccl_id_copy, shared_state);

                // Initialize NodeMasterComm for cross-node operations IMMEDIATELY after the
                // per-GPU communicator init. The ncclCommInitRank above is a collective operation
                // that synchronizes all 8 GPUs across both nodes, so all local_rank=0 threads
                // exit that call at approximately the same time. We leverage this synchronization
                // to ensure the NodeMasterComm init (also a collective) is called simultaneously
                // by both node masters without needing an explicit cross-node barrier.
                //
                // IMPORTANT: No local barrier before this! A local barrier would desynchronize
                // the cross-node timing because nodes have different numbers of threads that
                // might reach the barrier at different speeds.
                if (local_rank == 0 && num_nodes > 1) {
                    // Stay on device 0 (already set by NCCLCommunicatorImpl constructor for local_rank 0)
                    ncclCheck(ncclCommInitRank(&shared_state->NodeMasterComm, num_nodes,
                                               *reinterpret_cast<const ncclUniqueId*>(node_master_nccl_id_copy), node_rank));
                }

                // Now sync all local threads before proceeding to work
                shared_state->Barrier->arrive_and_wait();

                work(comm);

                shared_state->Barrier->arrive_and_wait();
            } catch (...) {
                std::lock_guard<std::mutex> lock(shared_state->Mutex);
                shared_state->Exceptions[local_rank] = std::current_exception();
            }
        });
    }

    return std::make_unique<CommunicatorThreadsPackImpl>(std::move(threads), shared_state);
}
