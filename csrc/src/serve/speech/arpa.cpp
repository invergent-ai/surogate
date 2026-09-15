// Copyright (c) 2026 Invergent SA. SPDX-License-Identifier: Apache-2.0
// Import NeMo token-ID ARPA LMs into the suffix-graph tensors used by the
// native decoder. BOS/EOS and unknown normalization follow NeMo; see NOTICE.
#include <nlohmann/json.hpp>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace {
constexpr int BOS = -1, EOS = -2, UNK = -3;

struct Gram {
    std::array<int, 4> tokens{};
    float weight = 0, backoff = 0;
};

struct State {
    int begin = 0, end = 0, backoff = 0;
    float weight = 0, final = -std::numeric_limits<float>::infinity();
};

struct Arc {
    int from, to, label;
    float weight;
};

// Prefix-free 12-bit packing, sufficient for four-gram models with <=4093
// token IDs. Histories contain at most three labels; BOS occupies one code.
uint64_t key(const int* tokens, int count) {
    uint64_t result = 1;
    for (int i = 0; i < count; ++i)
        result = (result << 12) | uint64_t(tokens[i] == BOS ? 4095 : tokens[i] + 1);
    return result;
}

int token(const std::string& s, int vocab) {
    if (s == "<s>") return BOS;
    if (s == "</s>") return EOS;
    if (s == "<unk>") return UNK;
    if (s.empty()) throw std::runtime_error("empty ARPA token");
    auto lead = static_cast<unsigned char>(s[0]);
    int size  = lead < 0x80                    ? 1
                : lead >= 0xc2 && lead < 0xe0  ? 2
                : lead >= 0xe0 && lead < 0xf0  ? 3
                : lead >= 0xf0 && lead <= 0xf4 ? 4
                                               : 0;
    if (!size || int(s.size()) != size)
        throw std::runtime_error("ARPA must use NeMo token-ID characters, not words");
    int code = lead & (size == 1 ? 127 : (1 << (7 - size)) - 1);
    for (int i = 1; i < size; ++i) {
        auto c = static_cast<unsigned char>(s[i]);
        if ((c & 0xc0) != 0x80) throw std::runtime_error("invalid ARPA UTF-8");
        code = (code << 6) | (c & 63);
    }
    if ((size == 2 && code < 128) || (size == 3 && code < 2048) || (size == 4 && code < 65536) ||
        code > 0x10ffff || (code >= 0xd800 && code <= 0xdfff))
        throw std::runtime_error("invalid ARPA UTF-8");
    int label = code - 100;
    if (label < 0 || label >= vocab)
        throw std::runtime_error("ARPA token ID is outside the acoustic vocabulary");
    return label;
}

float score(const std::string& s) {
    size_t end   = 0;
    double value = std::stod(s, &end);
    if (end != s.size() || std::isnan(value) || value == std::numeric_limits<double>::infinity())
        throw std::runtime_error("invalid ARPA score");
    return float(value / std::log10(std::exp(1.0)));
}

Gram parse(const std::string& line, int order, int vocab) {
    auto tab  = line.find('\t');
    auto next = line.find('\t', tab + 1);
    if (tab == std::string::npos) throw std::runtime_error("ARPA entries must be tab-separated");
    Gram gram;
    gram.weight = score(line.substr(0, tab));
    if (next != std::string::npos) gram.backoff = score(line.substr(next + 1));
    std::istringstream input(
        line.substr(tab + 1, next == std::string::npos ? next : next - tab - 1));
    std::string word;
    for (int i = 0; i < order; ++i) {
        if (!(input >> word)) throw std::runtime_error("ARPA entry has too few tokens");
        gram.tokens[i] = token(word, vocab);
        if (gram.tokens[i] == BOS && i != 0)
            throw std::runtime_error("BOS must begin an ARPA history");
        if (gram.tokens[i] == EOS && i != order - 1)
            throw std::runtime_error("EOS must end an ARPA entry");
        if (gram.tokens[i] == UNK && order != 1)
            throw std::runtime_error("unknown-token histories are unsupported");
    }
    if (input >> word) throw std::runtime_error("ARPA entry has too many tokens");
    return gram;
}

void convert(const std::string& path, const std::filesystem::path& output, int vocab) {
    if (vocab < 1 || vocab > 4093)
        throw std::runtime_error("ARPA vocabulary must contain 1..4093 tokens");
    std::ifstream input(path);
    if (!input) throw std::runtime_error("cannot read ARPA file");
    std::string line;
    std::array<size_t, 5> counts{};
    int max_order = 0;
    bool data     = false;
    while (std::getline(input, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.empty()) {
            if (data) break;
            continue;
        }
        if (!data) {
            if (line != "\\data\\") throw std::runtime_error("missing ARPA data header");
            data = true;
            continue;
        }
        int order;
        unsigned long long count;
        char extra;
        if (std::sscanf(line.c_str(), "ngram %d=%llu %c", &order, &count, &extra) != 2 ||
            order < 1 || order > 4 || counts[order] || !count)
            throw std::runtime_error("invalid ARPA counts (supported order: 2..4)");
        if (count > INT32_MAX) throw std::runtime_error("ARPA graph exceeds 32-bit indexing");
        counts[order] = count;
        max_order     = std::max(max_order, order);
    }
    if (max_order < 2) throw std::runtime_error("ARPA must contain at least bigrams");
    size_t max_states = 2 + vocab, max_arcs = vocab;
    for (int i = 1; i <= max_order; ++i) {
        if (!counts[i]) throw std::runtime_error("ARPA order is missing");
        if (i > 1) max_arcs += counts[i];
        if (i > 1 && i < max_order) max_states += counts[i];
    }
    if (max_states > INT32_MAX || max_arcs > INT32_MAX)
        throw std::runtime_error("ARPA graph exceeds 32-bit indexing");
    std::vector<State> states(2);
    states.reserve(max_states);
    std::vector<Arc> arcs(vocab);
    arcs.reserve(max_arcs);
    std::unordered_map<uint64_t, int> histories;
    histories.reserve(max_states);
    std::array<int, 1> bos{BOS};
    histories.emplace(key(bos.data(), 1), 1);
    int section = 0;
    size_t seen = 0;
    bool ended  = false;
    std::vector<Gram> unigrams;
    float unknown = -std::numeric_limits<float>::infinity();
    int known     = 0;
    auto finish   = [&]() {
        if (!section) return;
        if (seen != counts[section])
            throw std::runtime_error("ARPA count does not match section " +
                                       std::to_string(section));
        if (section != 1) return;
        std::sort(unigrams.begin(), unigrams.end(),
                    [](auto& a, auto& b) { return a.tokens[0] < b.tokens[0]; });
        bool has_bos = false, has_eos = false;
        int previous = INT32_MIN;
        for (auto& g : unigrams) {
            int id = g.tokens[0];
            if (id == previous) throw std::runtime_error("duplicate ARPA unigram");
            previous = id;
            if (id == UNK)
                unknown = g.weight;
            else if (id == BOS) {
                states[1].weight = g.backoff;
                has_bos          = true;
            } else if (id == EOS) {
                states[0].final = g.weight;
                has_eos         = true;
            } else {
                int state = states.size();
                states.push_back({0, 0, 0, g.backoff});
                histories.emplace(key(g.tokens.data(), 1), state);
                arcs[id] = {0, state, id, g.weight};
                ++known;
            }
        }
        if (!has_bos || !has_eos) throw std::runtime_error("ARPA requires BOS and EOS unigrams");
        if (vocab - known > 1) unknown -= std::log(float(vocab - known));
        for (int id = 0; id < vocab; ++id) {
            if (!histories.count(key(&id, 1))) arcs[id] = {0, 0, id, unknown};
        }
        unigrams.clear();
    };
    auto find = [&](const int* tokens, int count) {
        auto it = histories.find(key(tokens, count));
        if (it != histories.end()) return it->second;
        if (count == 1 && tokens[0] >= 0) return 0; // missing unigram follows the unknown arc
        throw std::runtime_error("ARPA prefix or suffix history is missing");
    };
    while (std::getline(input, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.empty()) continue;
        if (line == "\\end\\") {
            finish();
            ended = true;
            break;
        }
        if (line.front() == '\\') {
            finish();
            int order;
            char extra;
            if (std::sscanf(line.c_str(), "\\%d-grams:%c", &order, &extra) != 1 ||
                order != section + 1 || order > max_order)
                throw std::runtime_error("ARPA sections must occur in increasing order");
            section = order;
            seen    = 0;
            continue;
        }
        if (!section || ++seen > counts[section]) throw std::runtime_error("unexpected ARPA entry");
        auto g = parse(line, section, vocab);
        if (section == 1) {
            unigrams.push_back(g);
            continue;
        }
        int from = find(g.tokens.data(), section - 1), label = g.tokens[section - 1];
        if (label == EOS) {
            states[from].final = g.weight;
            continue;
        }
        int suffix = find(g.tokens.data() + 1, section - 1), to = suffix;
        if (section < max_order) {
            to = states.size();
            states.push_back({0, 0, suffix, g.backoff});
            if (!histories.emplace(key(g.tokens.data(), section), to).second)
                throw std::runtime_error("duplicate ARPA history");
        }
        arcs.push_back({from, to, label, g.weight});
    }
    if (!ended || section != max_order) throw std::runtime_error("ARPA is truncated");
    histories.clear();
    histories.rehash(0);
    std::sort(arcs.begin(), arcs.end(), [](auto& a, auto& b) {
        return a.from != b.from ? a.from < b.from : a.label < b.label;
    });
    for (size_t i = 0; i < arcs.size(); ++i) {
        auto& arc   = arcs[i];
        auto& state = states[arc.from];
        if (i && arcs[i - 1].from == arc.from && arcs[i - 1].label == arc.label)
            throw std::runtime_error("duplicate ARPA transition");
        if (!state.end) state.begin = i;
        state.end = i + 1;
    }
    // Write each tensor directly, without allocating a second graph-sized copy.
    nlohmann::json header;
    size_t offset = 0;
    auto field = [&](const char* name, const char* dtype, size_t count, std::vector<size_t> shape) {
        header[name] = {
            {"dtype", dtype}, {"shape", shape}, {"data_offsets", {offset, offset + count * 4}}};
        offset += count * 4;
    };
    auto na = arcs.size(), ns = states.size();
    field("arcs_weights", "F32", na, {na});
    field("to_states", "I32", na, {na});
    field("ilabels", "I32", na, {na});
    field("backoff_weights", "F32", ns, {ns});
    field("backoff_to_states", "I32", ns, {ns});
    field("final_weights", "F32", ns, {ns});
    field("start_end_arcs", "I32", ns * 2, {ns, 2});
    std::string h = header.dump();
    h.append((8 - h.size() % 8) % 8, ' ');
    uint64_t hsize = h.size();
    std::ofstream file(output / "lm.safetensors", std::ios::binary);
    file.exceptions(std::ios::badbit | std::ios::failbit);
    file.write(reinterpret_cast<char*>(&hsize), 8);
    file.write(h.data(), h.size());
    std::vector<uint32_t> buffer;
    buffer.reserve(65536);
    auto flush = [&]() {
        file.write(reinterpret_cast<const char*>(buffer.data()), buffer.size() * 4);
        buffer.clear();
    };
    auto emit = [&](auto value) {
        uint32_t bits;
        static_assert(sizeof(value) == 4);
        std::memcpy(&bits, &value, 4);
        buffer.push_back(bits);
        if (buffer.size() == buffer.capacity()) flush();
    };
    for (auto& a : arcs) emit(a.weight);
    flush();
    for (auto& a : arcs) emit(a.to);
    flush();
    for (auto& a : arcs) emit(a.label);
    flush();
    for (auto& s : states) emit(s.weight);
    flush();
    for (auto& s : states) emit(s.backoff);
    flush();
    for (auto& s : states) emit(s.final);
    flush();
    for (auto& s : states) {
        emit(s.begin);
        emit(s.end);
    }
    flush();
    file.close();
    nlohmann::json config = {{"vocab_size", vocab},
                             {"max_order", max_order},
                             {"num_states", ns},
                             {"num_arcs", na},
                             {"separate_bos_state", true}};
    std::ofstream meta(output / "lm.json");
    meta.exceptions(std::ios::badbit | std::ios::failbit);
    meta << config.dump() << '\n';
}
} // namespace

int main(int argc, char** argv) {
    try {
        if (argc != 4)
            throw std::runtime_error(
                "usage: surogate-stt-lm INPUT.arpa OUTPUT_DIRECTORY VOCAB_SIZE");
        convert(argv[1], argv[2], std::stoi(argv[3]));
    } catch (const std::exception& e) {
        std::cerr << "speech language model: " << e.what() << '\n';
        return 1;
    }
}
