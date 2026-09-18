// Copyright (c) 2026 Invergent SA. SPDX-License-Identifier: Apache-2.0
// FastConformer and TDT equations follow NVIDIA NeMo (Apache-2.0); see NOTICE.
#include "model.h"
#include <ATen/TensorIndexing.h>
#include <algorithm>
#include <cmath>
#include <chrono>
#include <cstring>
#include <fstream>
#include <limits>
#include <numeric>
#include <unordered_map>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <dlfcn.h>

namespace sinfer::speech {
using namespace at::indexing;

struct Weights::Mapping {
    int fd      = -1;
    size_t size = 0;
    void* data  = MAP_FAILED;

    ~Mapping() {
        if (data != MAP_FAILED) munmap(data, size);
        if (fd >= 0) close(fd);
    }
};

Weights::Weights(const std::string& path, at::Device device)
    : mapping_(std::make_shared<Mapping>()) {
    auto& m = *mapping_;
    m.fd    = open(path.c_str(), O_RDONLY);

    struct stat s {};

    if (m.fd < 0 || fstat(m.fd, &s) || s.st_size < 8)
        throw std::runtime_error("cannot read speech weights: " + path);
    m.size = s.st_size;
    m.data = mmap(nullptr, m.size, PROT_READ, MAP_PRIVATE, m.fd, 0);
    if (m.data == MAP_FAILED) throw std::runtime_error("cannot map speech weights");
    uint64_t n;
    std::memcpy(&n, m.data, 8);
    if (n > m.size - 8 || n > 16777216) throw std::runtime_error("invalid speech tensor header");
    auto header = json::parse(static_cast<char*>(m.data) + 8, static_cast<char*>(m.data) + 8 + n);
    for (auto& [name, entry] : header.items()) {
        if (name == "__metadata__") continue;
        auto shape   = entry.at("shape").get<std::vector<int64_t>>();
        auto offsets = entry.at("data_offsets").get<std::vector<uint64_t>>();
        auto dtype   = entry.at("dtype").get<std::string>();
        auto type    = dtype == "F32"   ? at::kFloat
                       : dtype == "I32" ? at::kInt
                       : dtype == "I64" ? at::kLong
                                        : at::kByte;
        if (type == at::kByte)
            throw std::runtime_error("unsupported speech tensor dtype: " + dtype);
        uint64_t bytes = c10::elementSize(type);
        for (auto dim : shape) {
            if (dim < 0 || (dim && bytes > m.size / static_cast<uint64_t>(dim)))
                throw std::runtime_error("invalid tensor shape");
            bytes *= dim;
        }
        if (offsets.size() != 2 || offsets[1] < offsets[0] || offsets[1] > m.size - 8 - n ||
            offsets[1] - offsets[0] != bytes)
            throw std::runtime_error("invalid speech tensor bounds");
        tensors_[name] = at::from_blob(static_cast<char*>(m.data) + 8 + n + offsets[0], shape,
                                       at::TensorOptions().dtype(type))
                             .to(device);
    }
}

static at::Device select_device(const std::string& device) {
    if (device == "cpu") return at::Device(at::kCPU);
    // CPU installations never load the CUDA backend. Retain the handle for
    // process lifetime: its registered dispatch kernels must remain resident.
    static void* cuda = dlopen("libtorch_cuda.so", RTLD_NOW | RTLD_GLOBAL);
    if (!cuda)
        throw std::runtime_error("GPU speech serving requires a CUDA-enabled PyTorch installation; "
                                 "use --device cpu for CPU serving");
    return at::Device("cuda:" + device);
}

Model::Model(const std::string& dir, const std::string& device, CpuKernels kernels)
    : device_(select_device(device)), w_(dir + "/acoustic.safetensors", device_),
      lm_(dir + "/lm.safetensors", at::kCPU) {
    bool available = device_.is_cpu() && at::hasMKL();
    if (kernels == CpuKernels::Optimized && !available)
        throw std::invalid_argument(
            "Optimized CPU kernels require --device cpu and an MKL-enabled LibTorch build");
    cpu_optimized_ = kernels != CpuKernels::Reference && available;
    std::ifstream f(dir + "/speech.json");
    f >> config;
    if (config.at("version") != 1) throw std::runtime_error("unsupported speech artifact version");
    auto e = config.at("model").at("encoder");
    hidden = e.at("d_model");
    heads  = e.at("n_heads");
    layers = e.at("n_layers");
    attention_cache_.resize(layers);
    left   = e.at("att_context_size")[0];
    right  = e.at("att_context_size")[1];
    kernel = e.at("conv_kernel_size");
    vocab  = config.at("model").at("decoder").at("vocab_size");
    if (!tokenizer_.Load(dir + "/tokenizer.model").ok() || tokenizer_.GetPieceSize() != vocab)
        throw std::runtime_error("invalid speech tokenizer");
}

Tensor Model::linear(const Tensor& x, const std::string& p, bool bias) const {
    if (cpu_optimized_ && p.find("feed_forward") != std::string::npos && p.ends_with("linear2"))
        return packed_linear_[p].run(x, w_[p + ".weight"],
                                     bias ? std::optional<Tensor>(w_[p + ".bias"]) : std::nullopt);
    return at::linear(x, w_[p + ".weight"],
                      bias ? std::optional<Tensor>(w_[p + ".bias"]) : std::nullopt);
}

std::array<Tensor, 3> Model::project_attention(const Tensor& query, const Tensor& kv,
                                               const std::string& p, int layer,
                                               bool streaming) const {
    // MKL's single-row GEMV takes a different reduction path after concatenation.
    // Preserve the original calls for that boundary case.
    if (!cpu_optimized_ || kv.size(1) == 1)
        return {linear(query, p + "linear_q"), linear(kv, p + "linear_k"),
                linear(kv, p + "linear_v")};
    auto& cache = attention_cache_[layer];
    if (!cache.weight.defined()) {
        cache.weight = at::cat(
            {w_[p + "linear_q.weight"], w_[p + "linear_k.weight"], w_[p + "linear_v.weight"]});
        cache.bias =
            at::cat({w_[p + "linear_q.bias"], w_[p + "linear_k.bias"], w_[p + "linear_v.bias"]});
    }
    if (streaming) {
        auto both = at::linear(kv, cache.weight.narrow(0, hidden, 2 * hidden),
                               cache.bias.narrow(0, hidden, 2 * hidden));
        return {linear(query, p + "linear_q"), both.narrow(2, 0, hidden),
                both.narrow(2, hidden, hidden)};
    }
    auto all = at::linear(query, cache.weight, cache.bias);
    return {all.narrow(2, 0, hidden), all.narrow(2, hidden, hidden),
            all.narrow(2, 2 * hidden, hidden)};
}

Tensor Model::norm(const Tensor& x, const std::string& p) const {
    return at::layer_norm(x, {hidden}, w_[p + ".weight"], w_[p + ".bias"], 1e-5, false);
}

Tensor Model::conv(const Tensor& x, const std::string& p, int groups) const {
    return at::conv1d(x, w_[p + ".weight"], w_[p + ".bias"], {1}, at::IntArrayRef{0}, {1}, groups);
}

Tensor Model::mel_frames(const Tensor& x) const {
    auto spectrum = at::stft(x, 512, 160, 400, w_["preprocessor.featurizer.window"], false,
                             "constant", false, true, true);
    // Match NeMo's sqrt followed by square (including its floating-point rounding).
    auto magnitude = at::sqrt(at::view_as_real(spectrum).pow(2).sum(-1)).pow(2);
    return at::log(at::matmul(w_["preprocessor.featurizer.fb"], magnitude) + std::pow(2., -24));
}

Tensor Model::mel(const Tensor& samples) const {
    auto x = samples.to(device_).flatten();
    if (x.numel() < 1) return empty_features();
    auto pre = at::cat({x.slice(0, 0, 1), x.slice(0, 1) - 0.97 * x.slice(0, 0, -1)});
    // NeMo uses centered constant padding for both file and live features.
    auto padded   = at::constant_pad_nd(pre, {256, 256}, 0);
    auto features = mel_frames(padded).slice(2, 0, x.numel() / 160);
    if (!streaming() && features.size(2)) {
        auto mean     = features.sum(2, true) / features.size(2);
        auto centered = features - mean;
        auto std      = features.size(2) > 1
                            ? at::sqrt(centered.pow(2).sum(2, true) / (features.size(2) - 1))
                            : at::zeros_like(mean);
        features      = centered / (std + 1e-5);
    }
    return features;
}

Tensor Model::encode(const Tensor& features, EncoderState* state) const {
    if (state && !streaming())
        throw std::invalid_argument("offline speech encoder cannot carry streaming state");
    auto x = features.transpose(1, 2).unsqueeze(1);
    for (int stage = 0; stage < 3; ++stage) {
        int index        = stage == 0 ? 0 : stage == 1 ? 2 : 5;
        std::string p    = "encoder.pre_encode.conv." + std::to_string(index);
        int padding_left = streaming() ? 2 : 1;
        x                = at::conv2d(at::constant_pad_nd(x, {padding_left, 1, padding_left, 1}, 0),
                                      w_[p + ".weight"], w_[p + ".bias"], {2, 2}, at::IntArrayRef{0, 0}, {1, 1},
                       stage == 0 ? 1 : 256);
        if (stage) {
            p = "encoder.pre_encode.conv." + std::to_string(index + 1);
            x = at::conv2d(x, w_[p + ".weight"], w_[p + ".bias"]);
        }
        x = cpu_optimized_ ? at::relu_(x) : at::relu(x);
    }
    x = linear(x.transpose(1, 2).contiguous().flatten(2), "encoder.pre_encode.out");
    if (state && state->step) x = x.slice(1, 2);
    int64_t qlen = x.size(1), cache = state ? left : 0, length = qlen + cache, dim = hidden / heads;
    if (!qlen) return x.transpose(1, 2);
    auto options   = x.options();
    auto positions = at::arange(length - 1, -length, -1, options).unsqueeze(1);
    auto frequency = at::exp(at::arange(0, hidden, 2, options) * (-std::log(10000.) / hidden));
    auto angles    = positions * frequency;
    auto pos       = at::stack({at::sin(angles), at::cos(angles)}, -1).flatten(1).unsqueeze(0);
    // Keep separate bounded caches for full segments and streaming chunks.
    // Long recordings still work without retaining large positional tensors.
    const int position_slot = state ? 1 : 0;
    const bool cache_positions =
        cpu_optimized_ && (2 * length - 1) * hidden * sizeof(float) * layers <= 32 * 1024 * 1024;
    if (cpu_optimized_ && position_lengths_[position_slot] != length) {
        for (auto& cache : attention_cache_) cache.positions[position_slot] = Tensor();
        position_lengths_[position_slot] = length;
    }
    Tensor mask;
    if (state) {
        auto indices = at::arange(length, options.dtype(at::kLong));
        auto chunks  = at::floor_divide(indices, right + 1);
        auto diff    = chunks.unsqueeze(1) - chunks.unsqueeze(0);
        mask         = ((diff < 0) | (diff > left / (right + 1)) |
                (indices.unsqueeze(0) < left - state->valid))
                   .slice(0, cache)
                   .unsqueeze(0)
                   .unsqueeze(0);
        if (state->attention.empty())
            for (int i = 0; i < layers; ++i) {
                state->attention.push_back(at::zeros({1, left, hidden}, options));
                state->convolution.push_back(at::zeros({1, hidden, kernel - 1}, options));
            }
    }
    for (int layer = 0; layer < layers; ++layer) {
        auto p     = "encoder.layers." + std::to_string(layer) + ".";
        x          = x + 0.5 * linear(at::silu(linear(norm(x, p + "norm_feed_forward1"),
                                                      p + "feed_forward1.linear1")),
                                      p + "feed_forward1.linear2");
        auto query = norm(x, p + "norm_self_att");
        auto kv    = state ? at::cat({state->attention[layer], query}, 1) : query;
        if (state) state->attention[layer] = kv.slice(1, -left).clone();
        auto a                = p + "self_attn.";
        auto projected        = project_attention(query, kv, a, layer, state != nullptr);
        auto q                = projected[0].view({1, qlen, heads, dim});
        auto k                = projected[1].view({1, length, heads, dim}).transpose(1, 2);
        auto v                = projected[2].view({1, length, heads, dim}).transpose(1, 2);
        auto& cached_position = attention_cache_[layer].positions[position_slot];
        auto pe =
            cache_positions && cached_position.defined()
                ? cached_position
                : linear(pos, a + "linear_pos", false).view({1, -1, heads, dim}).transpose(1, 2);
        if (cache_positions && !cached_position.defined()) cached_position = pe;
        auto bd   = at::matmul((q + w_[a + "pos_bias_v"]).transpose(1, 2), pe.transpose(-2, -1));
        auto plen = bd.size(-1);
        if (cpu_optimized_) {
            bd = cpu::relative_shift(bd, length);
        } else {
            bd = at::constant_pad_nd(bd, {1, 0}, 0)
                     .view({1, heads, -1, qlen})
                     .slice(2, 1)
                     .reshape({1, heads, qlen, plen})
                     .slice(3, 0, length);
        }
        auto scores =
            (at::matmul((q + w_[a + "pos_bias_u"]).transpose(1, 2), k.transpose(-2, -1)) + bd) /
            std::sqrt(double(dim));
        if (state) scores = scores.masked_fill(mask, -10000.);
        auto probabilities = at::softmax(scores, -1);
        if (state) probabilities = probabilities.masked_fill(mask, 0.);
        x      = x + linear(at::matmul(probabilities, v).transpose(1, 2).reshape({1, qlen, hidden}),
                            a + "linear_out");
        auto c = p + "conv.";
        auto z = at::glu(conv(norm(x, p + "norm_conv").transpose(1, 2), c + "pointwise_conv1"), 1);
        int conv_left = streaming() ? kernel - 1 : (kernel - 1) / 2;
        z             = state ? at::cat({state->convolution[layer], z}, 2)
                              : at::constant_pad_nd(z, {conv_left, kernel - 1 - conv_left}, 0);
        if (state) state->convolution[layer] = z.slice(2, -(kernel - 1)).clone();
        z = conv(z, c + "depthwise_conv", hidden);
        z = at::batch_norm(z, w_[c + "batch_norm.weight"], w_[c + "batch_norm.bias"],
                           w_[c + "batch_norm.running_mean"], w_[c + "batch_norm.running_var"],
                           false, 0.1, 1e-5, false);
        x = x + conv(at::silu(z), c + "pointwise_conv2").transpose(1, 2);
        x = x + 0.5 * linear(at::silu(linear(norm(x, p + "norm_feed_forward2"),
                                             p + "feed_forward2.linear1")),
                             p + "feed_forward2.linear2");
        x = norm(x, p + "norm_out");
    }
    if (state) {
        state->valid = std::min<int64_t>(left, state->valid + qlen);
        ++state->step;
    }
    return x.transpose(1, 2);
}

Tensor Model::ctc(const Tensor& encoded) const {
    return at::log_softmax(conv(encoded, "ctc_decoder.decoder_layers.0").transpose(1, 2), -1);
}

std::pair<Tensor, Tensor> Model::predict(int token, const Tensor& h, const Tensor& c) const {
    auto input = w_["decoder.prediction.embed.weight"][token].view({1, -1});
    auto p     = std::string("decoder.prediction.dec_rnn.lstm.");
    auto gates = at::linear(input, w_[p + "weight_ih_l0"], w_[p + "bias_ih_l0"]) +
                 at::linear(h, w_[p + "weight_hh_l0"], w_[p + "bias_hh_l0"]);
    auto parts = gates.chunk(4, 1);
    auto next  = at::sigmoid(parts[1]) * c + at::sigmoid(parts[0]) * at::tanh(parts[2]);
    return {at::sigmoid(parts[3]) * at::tanh(next), next};
}

std::string Model::text(const std::vector<int>& ids) const {
    std::string result;
    auto status = tokenizer_.Decode(ids, &result);
    if (!status.ok()) throw std::runtime_error(status.ToString());
    return result;
}

std::string Model::tdt(const Tensor& encoded, PredictorState& state) const {
    if (!state.h.defined()) {
        int d                      = w_["decoder.prediction.embed.weight"].size(1);
        auto zero                  = at::zeros({1, d}, encoded.options());
        std::tie(state.h, state.c) = predict(vocab, zero, zero);
        state.prediction           = linear(state.h, "joint.pred");
    }
    auto enc       = linear(encoded.transpose(1, 2), "joint.enc");
    auto durations = config["model"]["model_defaults"]["tdt_durations"].get<std::vector<int>>();
    int time = state.time_jump, symbols = 0;
    while (time < enc.size(1)) {
        auto logits = linear(at::relu(enc.select(1, time) + state.prediction), "joint.joint_net.2")
                          .to(at::kCPU)
                          .contiguous();
        auto values = logits.data_ptr<float>();
        int token   = std::max_element(values, values + vocab + 1) - values;
        int duration =
            durations[std::max_element(values + vocab + 1, values + vocab + 1 + durations.size()) -
                      (values + vocab + 1)];
        if (token != vocab) {
            state.tokens.push_back(token);
            std::tie(state.h, state.c) = predict(token, state.h, state.c);
            state.prediction           = linear(state.h, "joint.pred");
        }
        if (token == vocab && duration == 0) duration = 1;
        if (duration == 0 && ++symbols >= 10) duration = 1;
        if (duration) {
            time += duration;
            symbols = 0;
        }
    }
    state.time_jump = time - enc.size(1);
    return text(state.tokens);
}

std::string Model::beam(const Tensor& log_probs) const {
    const size_t beam_size = streaming() ? 64 : 32;
    const float alpha      = streaming() ? 0.55f : 0.5f;
    const float beta       = streaming() ? 1.75f : 2.0f;
    // NeMo beam_batch: expand the full vocabulary, prune, then recombine
    // identical collapsed transcripts with the same last frame label.
    auto cpu            = log_probs.squeeze(0).to(at::kCPU).contiguous();
    const auto* values  = cpu.data_ptr<float>();
    const auto* arcs    = lm_["arcs_weights"].data_ptr<float>();
    const auto* backoff = lm_["backoff_weights"].data_ptr<float>();
    const auto* final   = lm_["final_weights"].data_ptr<float>();
    const auto* bounds  = lm_["start_end_arcs"].data_ptr<int>();
    const auto* labels  = lm_["ilabels"].data_ptr<int>();
    const auto* targets = lm_["to_states"].data_ptr<int>();
    const auto* backs   = lm_["backoff_to_states"].data_ptr<int>();

    struct Hyp {
        float score;
        int state, last;
        std::vector<int> tokens;
    };

    struct Candidate {
        float score;
        int parent, label, state;
    };

    std::vector<Hyp> beam{{0, config["lm"].value("separate_bos_state", true) ? 1 : 0, -1, {}}};
    std::vector<float> scores(vocab);
    std::vector<int> states(vocab);
    auto resolve_lm = [&](int current, std::vector<int>& row_states,
                          std::vector<float>& row_scores) {
        std::fill(row_states.begin(), row_states.end(), -1);
        float accumulated = 0;
        for (int order = 0; order <= config["lm"]["max_order"].get<int>(); ++order) {
            for (int j = bounds[2 * current]; j < bounds[2 * current + 1]; ++j) {
                int token = labels[j];
                if (token >= 0 && token < vocab && row_states[token] < 0) {
                    row_states[token] = targets[j];
                    row_scores[token] = accumulated + arcs[j];
                }
            }
            if (current == 0) break;
            accumulated += backoff[current];
            current = backs[current];
        }
    };

    struct LmRow {
        std::vector<int> states;
        std::vector<float> scores;
    };

    // Blank frames and competing hypotheses repeatedly visit the same LM state.
    // Cache its immutable backoff expansion within this request, with a fixed cap.
    std::unordered_map<int, LmRow> lm_rows;
    for (int t = 0; t < cpu.size(0); ++t) {
        std::vector<Candidate> candidates;
        candidates.reserve(beam.size() * (vocab + 1));
        for (int b = 0; b < int(beam.size()); ++b) {
            const std::vector<int>* row_states   = &states;
            const std::vector<float>* row_scores = &scores;
            if (cpu_optimized_) {
                auto found = lm_rows.find(beam[b].state);
                if (found == lm_rows.end()) {
                    if (lm_rows.size() >= beam_size * 4) lm_rows.clear();
                    LmRow row{std::vector<int>(vocab), std::vector<float>(vocab)};
                    resolve_lm(beam[b].state, row.states, row.scores);
                    found = lm_rows.emplace(beam[b].state, std::move(row)).first;
                }
                row_states = &found->second.states;
                row_scores = &found->second.scores;
            } else {
                resolve_lm(beam[b].state, states, scores);
            }
            for (int token = 0; token <= vocab; ++token) {
                bool extend = token != vocab && token != beam[b].last;
                if (extend && (*row_states)[token] < 0) continue;
                float score = beam[b].score + values[t * (vocab + 1) + token];
                if (extend) score += beta + alpha * (*row_scores)[token];
                candidates.push_back(
                    {score, b, token, extend ? (*row_states)[token] : beam[b].state});
            }
        }
        size_t count = std::min(beam_size, candidates.size());
        std::partial_sort(candidates.begin(), candidates.begin() + count, candidates.end(),
                          [](auto& a, auto& b) { return a.score > b.score; });
        std::vector<Hyp> next;
        for (size_t i = 0; i < count; ++i) {
            const auto& c = candidates[i];
            if (c.score <= candidates[0].score - 20) continue;
            auto tokens = beam[c.parent].tokens;
            if (c.label != vocab && c.label != beam[c.parent].last) tokens.push_back(c.label);
            auto same = std::find_if(next.begin(), next.end(), [&](auto& h) {
                return h.last == c.label && h.tokens == tokens;
            });
            if (same == next.end())
                next.push_back({c.score, c.state, c.label, std::move(tokens)});
            else
                same->score = std::max(same->score, c.score);
        }
        beam = std::move(next);
    }
    float best = -std::numeric_limits<float>::infinity();
    std::vector<int> tokens;
    for (auto& h : beam) {
        int state   = h.state;
        float score = final[state], acc = 0;
        for (int order = 0; score <= -1e4f && state && order < config["lm"]["max_order"].get<int>();
             ++order) {
            acc += backoff[state];
            state = backs[state];
            score = acc + final[state];
        }
        h.score += alpha * score;
        if (h.score > best) {
            best   = h.score;
            tokens = h.tokens;
        }
    }
    return text(tokens);
}

Stream::Stream(Model& model, const std::string& directory)
    : model_(model), vad_(torch::jit::load(directory + "/vad.jit", at::kCPU)) {
    reset();
}

void Stream::reset() {
    encoder_   = {};
    predictor_ = {};
    samples_ = frames_ = quiet_ = 0;
    previous_                   = 0;
    speech_                     = false;
    fft_                        = at::zeros({256}, at::TensorOptions().device(model_.device()));
    pending_                    = model_.empty_features();
    history_                    = pending_.clone();
    audio_.clear();
    partial_.clear();
    displayed_.clear();
    vad_.get_method("reset_states")({});
    start_ = total_;
}

void Stream::features(const Tensor& input, bool final) {
    auto x = input.to(model_.device());
    if (x.numel()) {
        auto before = at::cat({at::full({1}, previous_, x.options()), x.slice(0, 0, -1)});
        fft_        = at::cat({fft_, x - 0.97 * before});
        previous_   = x[-1].item<float>();
        samples_ += x.numel();
    }
    if (final) fft_ = at::constant_pad_nd(fft_, {0, 256}, 0);
    int64_t available = fft_.numel() < 512 ? 0 : (fft_.numel() - 512) / 160 + 1;
    int64_t count     = std::min(available, samples_ / 160 - frames_);
    if (count > 0) {
        pending_ =
            at::cat({pending_, model_.mel_frames(fft_.slice(0, 0, 512 + 160 * (count - 1)))}, 2);
        fft_ = fft_.slice(0, count * 160).clone();
        frames_ += count;
    }
    drain(final);
}

void Stream::drain(bool final) {
    while (pending_.size(2)) {
        int required = (encoder_.step ? 8 : 1) + 8 * model_.right;
        if (!final && pending_.size(2) < required) break;
        int count  = std::min<int64_t>(required, pending_.size(2));
        auto chunk = pending_.slice(2, 0, count);
        pending_   = pending_.slice(2, count);
        auto hist  = encoder_.step ? history_.slice(2, -9) : history_.slice(2, 0, 0);
        if (encoder_.step && hist.size(2) < 9)
            hist = at::constant_pad_nd(hist, {9 - hist.size(2), 0}, 0);
        auto encoded = model_.encode(at::cat({hist, chunk}, 2), &encoder_);
        if (!final || pending_.size(2)) encoded = encoded.slice(2, 0, model_.right + 1);
        if (encoded.size(2)) partial_ = model_.tdt(encoded, predictor_);
        history_ = at::cat({history_, chunk}, 2).slice(2, -9).clone();
    }
}

void Stream::finish_segment(const char* reason, std::vector<json>& events) {
    if (audio_.empty()) return;
    auto begin = std::chrono::steady_clock::now();
    features(at::empty({0}), true);
    auto samples = at::from_blob(audio_.data(), {int64_t(audio_.size())}, at::kFloat);
    // The published finalizer re-encodes the segment with full attention.
    auto final = samples.numel() < 160
                     ? std::string()
                     : model_.beam(model_.ctc(model_.encode(model_.mel(samples))));
    events.push_back({{"type", "final"},
                      {"text", final},
                      {"partial", partial_},
                      {"reason", reason},
                      {"start_seconds", start_ / 16000.},
                      {"audio_seconds", audio_.size() / 16000.},
                      {"finalize_ms", std::chrono::duration<double, std::milli>(
                                          std::chrono::steady_clock::now() - begin)
                                          .count()}});
    reset();
}

void Stream::packet(const float* data, size_t count, std::vector<json>& events) {
    audio_.insert(audio_.end(), data, data + count);
    total_ += count;
    auto x = at::from_blob(const_cast<float*>(data), {int64_t(count)}, at::kFloat);
    features(x, false);
    if (count < 512) x = at::constant_pad_nd(x, {0, 512 - int64_t(count)}, 0);
    float probability = vad_.forward({x.unsqueeze(0), int64_t(16000)}).toTensor().item<float>();
    if (probability >= 0.5) {
        speech_ = true;
        quiet_  = 0;
    } else if (probability < 0.35)
        quiet_ += count;
    if (speech_ && partial_ != displayed_) {
        events.push_back(
            {{"type", "partial"}, {"text", partial_}, {"audio_seconds", audio_.size() / 16000.}});
        displayed_ = partial_;
    }
    if (speech_ && quiet_ >= 10240)
        finish_segment("pause", events);
    else if (audio_.size() >= 960000)
        finish_segment("max_duration", events);
    else if (!speech_ && audio_.size() >= 160000)
        finish_segment("idle", events);
}

std::vector<json> Stream::accept(const std::vector<float>& pcm, bool finish) {
    if (closed_) throw std::runtime_error("speech stream is closed");
    for (float x : pcm)
        if (!std::isfinite(x) || std::abs(x) > 1.0f)
            throw std::invalid_argument("PCM samples must be finite and between -1 and 1");
    packet_.insert(packet_.end(), pcm.begin(), pcm.end());
    std::vector<json> events;
    size_t offset = 0;
    while (packet_.size() - offset >= 512) {
        packet(packet_.data() + offset, 512, events);
        offset += 512;
    }
    packet_.erase(packet_.begin(), packet_.begin() + offset);
    if (finish) {
        if (!packet_.empty()) {
            packet(packet_.data(), packet_.size(), events);
            packet_.clear();
        }
        finish_segment("eof", events);
        closed_ = true;
    }
    return events;
}
} // namespace sinfer::speech
