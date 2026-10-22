// Trainer vs serving engine: the n-gram memory's row selection on fixed ids.
//
// The trainer's ngram_memory::rows (kernels/ngram_memory.cu, linked from libsurogate-common) and the engine's
// rows_kernel (csrc/src/serve/ops/ngram_ple/ngram_ple.cu, extracted verbatim by extract_engine_rows.py into
// engine_rows_extract.cuh) on the same deterministic token stream: N tokens, ids uniform over the 248,320-token
// vocabulary, EOS (the hash's eos) at 2 %, conversation starts at random (position 0) and at a chunk start.
//   trainer: one batch row, sequence chunks of 4096 columns with the two-token carry, positions per conversation;
//   engine : the same stream served in segments of 1, 3 or up to 700 columns, every segment reading its two
//            predecessors from slot history (a fresh conversation's history is EOS), as the engine serves it.
// Usage: ngram_rows_parity HASH_TXT N SEED OUT_DIR
//   HASH_TXT: "ngram heads eos m0 m1 m2" then heads offsets then heads vocab sizes (whitespace separated)
// Writes OUT_DIR/{ids,pos,rows_trainer,rows_engine}.i32; prints the mismatch count; exit 0 iff identical.
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "engine_rows_extract.cuh"
#include "kernels/ngram_memory.h"

#define CK(x)                                                                              \
    do {                                                                                   \
        cudaError_t e_ = (x);                                                              \
        if (e_ != cudaSuccess) {                                                           \
            std::fprintf(stderr, "CUDA %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); \
            return 2;                                                                      \
        }                                                                                  \
    } while (0)

static std::uint64_t rng_state = 0;
static std::uint64_t next_u64() {  // splitmix64
    std::uint64_t z = (rng_state += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}
static double uniform() { return static_cast<double>(next_u64() >> 11) * (1.0 / 9007199254740992.0); }

static void write_i32(const std::string& path, const std::vector<int>& v) {
    std::ofstream f(path, std::ios::binary);
    f.write(reinterpret_cast<const char*>(v.data()), static_cast<std::streamsize>(v.size() * sizeof(int)));
}

int main(int argc, char** argv) {
    if (argc != 5) {
        std::fprintf(stderr, "usage: %s HASH_TXT N SEED OUT_DIR\n", argv[0]);
        return 2;
    }
    std::ifstream hf(argv[1]);
    ngram_memory::Hash th{};
    engine_rows::NgramPleHash eh{};
    long long ngram = 0, heads = 0, eos = 0;
    hf >> ngram >> heads >> eos;
    if (!hf || ngram < 2 || ngram > 3 || heads <= 0 || heads > 16) {
        std::fprintf(stderr, "bad hash file\n");
        return 2;
    }
    th.ngram = eh.ngram = static_cast<int>(ngram);
    th.heads = eh.heads = static_cast<int>(heads);
    th.eos_token = eh.eos_token = static_cast<int>(eos);
    for (int i = 0; i < 3; ++i) {
        unsigned long long m = 0;
        hf >> m;
        th.multipliers[i] = eh.multipliers[i] = m;
    }
    for (int h = 0; h < heads; ++h) {
        long long o = 0;
        hf >> o;
        th.head_offsets[h] = eh.head_offsets[h] = static_cast<int>(o);
    }
    for (int h = 0; h < heads; ++h) {
        long long s = 0;
        hf >> s;
        th.head_vocab_sizes[h] = eh.head_vocab_sizes[h] = static_cast<int>(s);
    }
    if (!hf) {
        std::fprintf(stderr, "bad hash file (offsets / sizes)\n");
        return 2;
    }
    const long N = std::stol(argv[2]);
    rng_state = std::stoull(argv[3]);
    const std::string out = argv[4];
    constexpr int kChunk = 4096;
    const long n = (N + kChunk - 1) / kChunk * kChunk;  // whole chunks

    // The stream: ids, conversation starts (a start at position 0 of the stream, at random, and at chunk 3's start)
    std::vector<int> ids(n), pos(n);
    int p = 0;
    for (long t = 0; t < n; ++t) {
        const bool start = t == 0 || uniform() < 1.0 / 900.0 || t == 3L * kChunk;
        p = start ? 0 : p + 1;
        pos[t] = p;
        ids[t] = uniform() < 0.02 ? th.eos_token : static_cast<int>(next_u64() % 248320ULL);
    }

    // Trainer: chunk by chunk with the carry (the two tokens before the chunk, oldest first)
    std::vector<int> rows_t(static_cast<size_t>(n) * heads), rows_e(static_cast<size_t>(n) * heads);
    int *d_ids = nullptr, *d_pos = nullptr, *d_carry = nullptr, *d_rows = nullptr;
    CK(cudaMalloc(&d_ids, n * sizeof(int)));
    CK(cudaMalloc(&d_pos, n * sizeof(int)));
    CK(cudaMalloc(&d_carry, 2 * sizeof(int)));
    CK(cudaMalloc(&d_rows, static_cast<size_t>(n) * heads * sizeof(int)));
    CK(cudaMemcpy(d_ids, ids.data(), n * sizeof(int), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_pos, pos.data(), n * sizeof(int), cudaMemcpyHostToDevice));
    for (long c = 0; c < n / kChunk; ++c) {
        const int* carry = nullptr;
        if (c > 0) {
            const int h2[2] = {ids[c * kChunk - 2], ids[c * kChunk - 1]};
            CK(cudaMemcpy(d_carry, h2, sizeof(h2), cudaMemcpyHostToDevice));
            carry = d_carry;
        }
        ngram_memory::rows(d_rows + c * kChunk * heads, d_ids + c * kChunk, d_pos + c * kChunk, carry, 1, kChunk, th,
                           nullptr);
        CK(cudaDeviceSynchronize());
    }
    CK(cudaMemcpy(rows_t.data(), d_rows, rows_t.size() * sizeof(int), cudaMemcpyDeviceToHost));

    // Engine: segments of 1, 3 or up to 700 columns inside each conversation; every segment its own slot whose
    // history holds the conversation's two tokens before it (EOS where the conversation has none: a fresh slot)
    std::vector<int> seg_begin(n), slot(n), history;
    int slots = 0;
    for (long t = 0; t < n;) {
        long conv_end = t + 1;
        while (conv_end < n && pos[conv_end] != 0) ++conv_end;
        for (long s = t; s < conv_end;) {
            const double u = uniform();
            long len = u < 0.2 ? 1 : (u < 0.4 ? 3 : 1 + static_cast<long>(next_u64() % 700));
            len = std::min(len, conv_end - s);
            for (int j = 0; j < 2; ++j) {  // oldest first: the columns s-2, s-1 of this conversation
                const long src = s - 2 + j;
                history.push_back(src >= t ? ids[src] : th.eos_token);
            }
            for (long k = s; k < s + len; ++k) {
                seg_begin[k] = static_cast<int>(s);
                slot[k] = slots;
            }
            ++slots;
            s += len;
        }
        t = conv_end;
    }
    int *d_begin = nullptr, *d_slot = nullptr, *d_hist = nullptr;
    CK(cudaMalloc(&d_begin, n * sizeof(int)));
    CK(cudaMalloc(&d_slot, n * sizeof(int)));
    CK(cudaMalloc(&d_hist, history.size() * sizeof(int)));
    CK(cudaMemcpy(d_begin, seg_begin.data(), n * sizeof(int), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_slot, slot.data(), n * sizeof(int), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_hist, history.data(), history.size() * sizeof(int), cudaMemcpyHostToDevice));
    CK(cudaMemset(d_rows, 0xFF, static_cast<size_t>(n) * heads * sizeof(int)));
    engine_rows::rows_kernel<<<static_cast<unsigned>(n), 32>>>(d_ids, d_begin, d_slot, d_hist, eh, 2, slots, d_rows);
    CK(cudaGetLastError());
    CK(cudaDeviceSynchronize());
    CK(cudaMemcpy(rows_e.data(), d_rows, rows_e.size() * sizeof(int), cudaMemcpyDeviceToHost));

    long mismatches = 0;
    for (size_t i = 0; i < rows_t.size(); ++i) {
        if (rows_t[i] != rows_e[i]) {
            if (mismatches < 5) {
                std::fprintf(stderr, "mismatch at token %zu head %zu: trainer %d engine %d\n", i / heads, i % heads,
                             rows_t[i], rows_e[i]);
            }
            ++mismatches;
        }
    }
    write_i32(out + "/ids.i32", ids);
    write_i32(out + "/pos.i32", pos);
    write_i32(out + "/rows_trainer.i32", rows_t);
    write_i32(out + "/rows_engine.i32", rows_e);
    std::printf("tokens %ld heads %lld segments %d conversations-starts %ld mismatches %ld\n", n, heads, slots,
                static_cast<long>(std::count(pos.begin(), pos.end(), 0)), mismatches);
    return mismatches == 0 ? 0 : 1;
}
