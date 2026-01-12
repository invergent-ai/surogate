#pragma once

// Accessors + simple methods

template<typename Block>
float ModularTransformerModel<Block>::get_loss() const {
    if (!mRunState) return 0.0f;

    // Get raw loss from IRunState base class (populated by reduce_loss)
    float raw_loss = mRunState->get_loss();

    // Normalize by valid token count (similar to LLamaModel::get_loss)
    // ValidTokenCount was reduced across ranks in backward_with_hook
    int valid_tokens;
    CUDA_CHECK(cudaMemcpy(&valid_tokens, mRunState->ValidTokenCount.Data, sizeof(int), cudaMemcpyDeviceToHost));

    if (valid_tokens > 0) {
        // ValidTokenCount is reduced across ranks (sum). Loss is reduced with ncclAvg, so
        // divide by the average valid tokens per rank for correct mean CE.
        float avg_valid = static_cast<float>(valid_tokens) / static_cast<float>(std::max(1, mRunState->WorldSize));
        return raw_loss / avg_valid;
    } else {
        return 0.0f;
    }
}

template<typename Block>
ITensorContainer& ModularTransformerModel<Block>::weights() {
    return *mWeights;
}

template<typename Block>
ITensorContainer& ModularTransformerModel<Block>::opt_momentum() {
    // 8-bit optimizer doesn't expose state as ITensorContainer
    static EmptyTensorContainer empty;
    return empty;
}

template<typename Block>
ITensorContainer& ModularTransformerModel<Block>::opt_momentum_scales() {
    // 8-bit optimizer doesn't use FP8 scales
    static EmptyTensorContainer empty;
    return empty;
}

template<typename Block>
ITensorContainer& ModularTransformerModel<Block>::opt_variance() {
    // 8-bit optimizer doesn't expose state as ITensorContainer
    static EmptyTensorContainer empty;
    return empty;
}

template<typename Block>
ITensorContainer& ModularTransformerModel<Block>::opt_variance_scales() {
    // 8-bit optimizer doesn't use FP8 scales
    static EmptyTensorContainer empty;
    return empty;
}

template<typename Block>
std::vector<std::byte> ModularTransformerModel<Block>::rng_state() const {
    std::stringstream tmp;
    static_cast<std::ostream&>(tmp) << mOptimizerRNG;
    auto view = tmp.rdbuf()->view();
    std::vector<std::byte> state;
    state.reserve(view.size());
    std::transform(view.begin(), view.end(), std::back_inserter(state),
                   [](char c) { return static_cast<std::byte>(c); });
    return state;
}

template<typename Block>
void ModularTransformerModel<Block>::set_rng_state(const std::vector<std::byte>& state) {
    std::stringstream tmp;
    tmp.write(reinterpret_cast<const char*>(state.data()), state.size());
    static_cast<std::istream&>(tmp) >> mOptimizerRNG;
}

template<typename Block>
std::string_view ModularTransformerModel<Block>::model_type() const {
    return mConfig.model_name();
}

template<typename Block>
IRunState& ModularTransformerModel<Block>::get_run_state() const {
    if (!mRunState) {
        throw std::logic_error("ModularTransformerModel::get_run_state() called before allocate_run_state()");
    }
    // ModularRunState inherits from IRunState, so this is safe
    return *mRunState;
}

