#pragma once
#include "core/arena.h"
#include "core/device.h"
#include "api/ops/sampling.h"
#include "runtime/contract/constraint.h"
#include <exception>
#include <cstring>
#include <algorithm>

namespace sinfer::family::detail {
// Stable host callback storage for captured speculative rounds. Drafts are copied
// after proposal/verification; masking explores a fork, never the live grammar.
class SpeculativeConstraintRound {
    std::size_t width_, words_;
    PinnedHostBuffer drafts_;
    PinnedHostBuffer masks_;
    Tensor device_masks_;
    std::vector<TokenConstraintState*> states_;
    std::exception_ptr error_;
    static void CUDART_CB fill(void* opaque) noexcept {
        auto& self = *static_cast<SpeculativeConstraintRound*>(opaque);
        try {
            const auto* drafts = static_cast<const TokenId*>(self.drafts_.data());
            auto* masks = static_cast<int32_t*>(self.masks_.data());
            for (std::size_t row = 0; row < self.states_.size(); ++row) {
                if (auto* state = self.states_[row]) {
                    state->fill_draft_masks({drafts + row * (self.width_ - 1), self.width_ - 1},
                        {masks + row * self.width_ * self.words_, self.width_ * self.words_});
                }
            }
        } catch (...) { self.error_ = std::current_exception(); }
    }
public:
    SpeculativeConstraintRound(Tensor masks, std::size_t width, std::size_t words, std::size_t batch)
        : width_(width), words_(words), drafts_((width - 1) * batch * sizeof(TokenId)),
          masks_(width * words * batch * sizeof(int32_t)), device_masks_(masks), states_(batch, nullptr) {
        std::memset(masks_.data(), 0xff, masks_.size());
    }

    void set_width(std::size_t width) { width_ = width; }
    void reset() { std::fill(states_.begin(), states_.end(), nullptr); error_ = {}; }
    void stage(std::size_t row, TokenConstraintState* state, ops::SamplingConfig& config) {
        states_.at(row) = state;
        if (state) {
            config.token_bitmask = static_cast<const int32_t*>(device_masks_.data) + row * width_ * words_;
            config.token_bitmask_stride = static_cast<int32_t>(words_);
        }
    }
    void enqueue(const Tensor& drafts, cudaStream_t stream) {
        if (width_ > 1) {
            CUDA_CHECK(cudaMemcpyAsync(drafts_.data(), drafts.data, drafts.bytes(),
                                       cudaMemcpyDeviceToHost, stream));
        }
        CUDA_CHECK(cudaLaunchHostFunc(stream, &fill, this));
        CUDA_CHECK(cudaMemcpyAsync(device_masks_.data, masks_.data(),
            static_cast<std::size_t>(drafts.ne[1]) * width_ * words_ * sizeof(int32_t), cudaMemcpyHostToDevice, stream));
    }
    void check() { if (error_) { std::rethrow_exception(error_); } }
};
} // namespace sinfer::family::detail
