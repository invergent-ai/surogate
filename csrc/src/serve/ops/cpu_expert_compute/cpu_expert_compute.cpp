// CPU expert compute over the planar W8G32 host bank (api/ops/cpu_expert_compute.h).
//
// Layout of one expert: gate/up = 2*intermediate rows (gate rows first) of `hidden` int8 codes,
// scales fp16 per 32-group; down = hidden rows of `intermediate` codes. Per job:
//   xq   = quantise(x)                      (int8 per group, float scale per group)
//   g,u  = dot(gate_row, xq), dot(up_row, xq)
//   h    = silu(g) * u                       (float, intermediate wide)
//   hq   = quantise(h)
//   y    = dot(down_row, hq)                 (hidden wide)
//   out += weight * y
// The AVX-512 path (compiled with -mavx512bw for this file) widens 32 int8 to int16, multiplies
// pairwise into int32 with vpmaddwd and accumulates in float per group; the scalar path is the
// same arithmetic in plain C++ and is what the unit test compares against a double reference.

#include "api/ops/cpu_expert_compute.h"

#include <atomic>
#include <cmath>
#include <condition_variable>
#include <cstring>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <vector>

#if defined(__x86_64__)
#include <immintrin.h>
#include <sched.h>
#endif

namespace ninfer::ops {
namespace {

constexpr int kGroup = 32;

inline float bf16_to_float(std::uint16_t bits) {
    std::uint32_t word = static_cast<std::uint32_t>(bits) << 16;
    float value;
    std::memcpy(&value, &word, sizeof(value));
    return value;
}

inline float fp16_to_float(std::uint16_t h) {
    const std::uint32_t sign = (h >> 15) & 1U;
    const std::uint32_t exp  = (h >> 10) & 0x1FU;
    const std::uint32_t mant = h & 0x3FFU;
    std::uint32_t word;
    if (exp == 0) {
        if (mant == 0) {
            word = sign << 31;
        } else { // subnormal
            int e = -1;
            std::uint32_t m = mant;
            do {
                ++e;
                m <<= 1;
            } while ((m & 0x400U) == 0);
            word = (sign << 31) | static_cast<std::uint32_t>(127 - 15 - e) << 23 | ((m & 0x3FFU) << 13);
        }
    } else if (exp == 31) {
        word = (sign << 31) | 0x7F800000U | (mant << 13);
    } else {
        word = (sign << 31) | ((exp + 127 - 15) << 23) | (mant << 13);
    }
    float value;
    std::memcpy(&value, &word, sizeof(value));
    return value;
}

inline float silu(float v) { return v / (1.0F + std::exp(-v)); }

/// Quantises `count` floats (a multiple of 32) into int8 groups with one float scale each.
void quantise_groups(const float* values, int count, std::int8_t* q, float* scales) {
    for (int g = 0; g < count / kGroup; ++g) {
        float amax = 0.0F;
        for (int i = 0; i < kGroup; ++i) { amax = std::fmax(amax, std::fabs(values[g * kGroup + i])); }
        const float scale = amax / 127.0F;
        const float inv   = scale > 0.0F ? 1.0F / scale : 0.0F;
        scales[g]         = scale;
        for (int i = 0; i < kGroup; ++i) {
            const float r = std::nearbyint(values[g * kGroup + i] * inv);
            q[g * kGroup + i] = static_cast<std::int8_t>(r > 127.0F ? 127.0F : (r < -127.0F ? -127.0F : r));
        }
    }
}

/// Scalar dot of one W8 row against a quantised activation: sum_g (int32 dot) * ws_g * xs_g.
float dot_row_scalar(const std::int8_t* codes, const std::uint16_t* scales, const std::int8_t* xq,
                     const float* xs, int k) {
    float acc = 0.0F;
    for (int g = 0; g < k / kGroup; ++g) {
        std::int32_t dot = 0;
        for (int i = 0; i < kGroup; ++i) {
            dot += static_cast<std::int32_t>(codes[g * kGroup + i]) * static_cast<std::int32_t>(xq[g * kGroup + i]);
        }
        acc += static_cast<float>(dot) * fp16_to_float(scales[g]) * xs[g];
    }
    return acc;
}

#if defined(__x86_64__) && defined(__AVX512BW__)
constexpr bool kAvx512Compiled = true;

bool detect_avx512() {
    return __builtin_cpu_supports("avx512f") && __builtin_cpu_supports("avx512bw") &&
           __builtin_cpu_supports("avx512vl") && __builtin_cpu_supports("avx512dq");
}

/// AVX-512 dot: per 32-group, widen int8→int16 (32 lanes), vpmaddwd → 16 int32 pairs, reduce.
__attribute__((target("avx512f,avx512bw,avx512vl,avx512dq,f16c,fma")))
float dot_row_avx512(const std::int8_t* codes, const std::uint16_t* scales, const std::int8_t* xq,
                     const float* xs, int k) {
    const int groups = k / kGroup;
    __m512 acc       = _mm512_setzero_ps();
    int g            = 0;
    // Two groups per iteration to amortise the horizontal reduction: 16 int32 lanes each.
    for (; g + 2 <= groups; g += 2) {
        const __m512i w0 = _mm512_cvtepi8_epi16(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(codes + g * kGroup)));
        const __m512i x0 = _mm512_cvtepi8_epi16(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(xq + g * kGroup)));
        const __m512i w1 = _mm512_cvtepi8_epi16(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(codes + (g + 1) * kGroup)));
        const __m512i x1 = _mm512_cvtepi8_epi16(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(xq + (g + 1) * kGroup)));
        const __m512i p0 = _mm512_madd_epi16(w0, x0); // 16 x int32, each a sum of 2 products
        const __m512i p1 = _mm512_madd_epi16(w1, x1);
        const float s0   = fp16_to_float(scales[g]) * xs[g];
        const float s1   = fp16_to_float(scales[g + 1]) * xs[g + 1];
        acc = _mm512_fmadd_ps(_mm512_cvtepi32_ps(p0), _mm512_set1_ps(s0), acc);
        acc = _mm512_fmadd_ps(_mm512_cvtepi32_ps(p1), _mm512_set1_ps(s1), acc);
    }
    for (; g < groups; ++g) {
        const __m512i w0 = _mm512_cvtepi8_epi16(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(codes + g * kGroup)));
        const __m512i x0 = _mm512_cvtepi8_epi16(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(xq + g * kGroup)));
        const __m512i p0 = _mm512_madd_epi16(w0, x0);
        const float s0   = fp16_to_float(scales[g]) * xs[g];
        acc = _mm512_fmadd_ps(_mm512_cvtepi32_ps(p0), _mm512_set1_ps(s0), acc);
    }
    return _mm512_reduce_add_ps(acc);
}
#else
constexpr bool kAvx512Compiled = false;
bool detect_avx512() { return false; }
float dot_row_avx512(const std::int8_t*, const std::uint16_t*, const std::int8_t*, const float*, int) {
    return 0.0F;
}
#endif

const bool kUseAvx512 = kAvx512Compiled && detect_avx512();

inline float dot_row(const std::int8_t* codes, const std::uint16_t* scales, const std::int8_t* xq,
                     const float* xs, int k) {
    return kUseAvx512 ? dot_row_avx512(codes, scales, xq, xs, k) : dot_row_scalar(codes, scales, xq, xs, k);
}

struct Scratch {
    float* x_float;
    std::int8_t* xq;
    float* xs;
    float* h;
    std::int8_t* hq;
    float* hs;
};

Scratch carve_scratch(const SparseMoeGeometry& geometry, std::byte* base) {
    auto align = [](std::size_t v) { return (v + 63) / 64 * 64; };
    Scratch s{};
    std::size_t offset = 0;
    s.x_float = reinterpret_cast<float*>(base + offset); offset += align(sizeof(float) * geometry.hidden);
    s.xq      = reinterpret_cast<std::int8_t*>(base + offset); offset += align(geometry.hidden);
    s.xs      = reinterpret_cast<float*>(base + offset); offset += align(sizeof(float) * (geometry.hidden / kGroup));
    s.h       = reinterpret_cast<float*>(base + offset); offset += align(sizeof(float) * geometry.intermediate);
    s.hq      = reinterpret_cast<std::int8_t*>(base + offset); offset += align(geometry.intermediate);
    s.hs      = reinterpret_cast<float*>(base + offset);
    return s;
}

void require_geometry(const SparseMoeGeometry& geometry) {
    if (geometry.hidden <= 0 || geometry.hidden % kGroup != 0 || geometry.intermediate <= 0 ||
        geometry.intermediate % kGroup != 0 || geometry.experts <= 0) {
        throw std::invalid_argument("cpu_expert_compute: hidden and intermediate must be positive multiples of 32");
    }
}

} // namespace

bool cpu_expert_compute_has_avx512() noexcept { return kUseAvx512; }

std::size_t cpu_expert_scratch_bytes(const SparseMoeGeometry& geometry) {
    require_geometry(geometry);
    auto align = [](std::size_t v) { return (v + 63) / 64 * 64; };
    return align(sizeof(float) * geometry.hidden) + align(geometry.hidden) +
           align(sizeof(float) * (geometry.hidden / kGroup)) + align(sizeof(float) * geometry.intermediate) +
           align(geometry.intermediate) + align(sizeof(float) * (geometry.intermediate / kGroup)) + 64;
}

void cpu_expert_compute_job(const SparseMoeGeometry& geometry, const CpuExpertBank& bank,
                            const CpuExpertJob& job, const std::uint16_t* x_column, float* out_column,
                            std::byte* scratch) {
    require_geometry(geometry);
    if (job.expert < 0 || job.expert >= geometry.experts) {
        throw std::invalid_argument("cpu_expert_compute: expert out of range");
    }
    const int hidden       = geometry.hidden;
    const int intermediate = geometry.intermediate;
    const Scratch s        = carve_scratch(geometry, scratch);
    for (int i = 0; i < hidden; ++i) { s.x_float[i] = bf16_to_float(x_column[i]); }
    quantise_groups(s.x_float, hidden, s.xq, s.xs);

    const std::size_t gate_rows       = static_cast<std::size_t>(2) * intermediate;
    const std::size_t gate_row_codes  = static_cast<std::size_t>(hidden);
    const std::size_t gate_row_scales = static_cast<std::size_t>(hidden / kGroup);
    const auto* gate_codes            = reinterpret_cast<const std::int8_t*>(bank.gate_up_codes) +
                           static_cast<std::size_t>(job.expert) * gate_rows * gate_row_codes;
    const auto* gate_scales = reinterpret_cast<const std::uint16_t*>(bank.gate_up_scales) +
                              static_cast<std::size_t>(job.expert) * gate_rows * gate_row_scales;
    for (int j = 0; j < intermediate; ++j) {
        const float g = dot_row(gate_codes + j * gate_row_codes, gate_scales + j * gate_row_scales, s.xq, s.xs, hidden);
        const float u = dot_row(gate_codes + (intermediate + j) * gate_row_codes,
                                gate_scales + (intermediate + j) * gate_row_scales, s.xq, s.xs, hidden);
        s.h[j] = silu(g) * u;
    }
    quantise_groups(s.h, intermediate, s.hq, s.hs);

    const std::size_t down_row_codes  = static_cast<std::size_t>(intermediate);
    const std::size_t down_row_scales = static_cast<std::size_t>(intermediate / kGroup);
    const auto* down_codes            = reinterpret_cast<const std::int8_t*>(bank.down_codes) +
                           static_cast<std::size_t>(job.expert) * hidden * down_row_codes;
    const auto* down_scales = reinterpret_cast<const std::uint16_t*>(bank.down_scales) +
                              static_cast<std::size_t>(job.expert) * hidden * down_row_scales;
    for (int r = 0; r < hidden; ++r) {
        out_column[r] += job.weight * dot_row(down_codes + r * down_row_codes, down_scales + r * down_row_scales,
                                              s.hq, s.hs, intermediate);
    }
}

// ---------------------------------------------------------------------------------------------
// Pool
// ---------------------------------------------------------------------------------------------

struct CpuExpertPool::Impl {
    SparseMoeGeometry geometry;
    std::uint32_t threads = 0;
    std::vector<std::thread> workers;
    std::vector<std::vector<std::byte>> scratch;
    std::mutex mutex;
    std::condition_variable wake;
    std::condition_variable done;
    bool stop = false;
    // Current round
    std::uint64_t generation = 0;
    const CpuExpertBank* bank = nullptr;
    const CpuExpertRound* round = nullptr;
    std::atomic<std::int64_t> next_job{0};
    std::atomic<std::uint32_t> finished{0};
    std::vector<std::unique_ptr<std::mutex>> token_locks;

    void worker(std::uint32_t index, bool pin) {
#if defined(__x86_64__)
        if (pin) {
            cpu_set_t set;
            CPU_ZERO(&set);
            CPU_SET(static_cast<int>(index), &set);
            sched_setaffinity(0, sizeof(set), &set);
        }
#else
        (void)pin;
#endif
        std::uint64_t seen = 0;
        for (;;) {
            {
                std::unique_lock<std::mutex> lock(mutex);
                wake.wait(lock, [&] { return stop || generation != seen; });
                if (stop) { return; }
                seen = generation;
            }
            std::byte* local = scratch[index].data();
            for (;;) {
                const std::int64_t j = next_job.fetch_add(1);
                if (j >= static_cast<std::int64_t>(round->jobs.size())) { break; }
                const CpuExpertJob& job = round->jobs[static_cast<std::size_t>(j)];
                const std::uint16_t* x  = round->x + static_cast<std::size_t>(job.token) * geometry.hidden;
                float* out              = round->out + static_cast<std::size_t>(job.token) * geometry.hidden;
                std::lock_guard<std::mutex> guard(*token_locks[static_cast<std::size_t>(job.token)]);
                cpu_expert_compute_job(geometry, *bank, job, x, out, local);
            }
            if (finished.fetch_add(1) + 1 == threads) {
                std::lock_guard<std::mutex> lock(mutex);
                done.notify_all();
            }
        }
    }
};

CpuExpertPool::CpuExpertPool(const SparseMoeGeometry& geometry, Options options)
    : impl_(std::make_unique<Impl>()) {
    require_geometry(geometry);
    impl_->geometry = geometry;
    std::uint32_t threads = options.threads;
    if (threads == 0) {
        const unsigned hw = std::thread::hardware_concurrency();
        threads           = hw > 1 ? hw / 2 : 1; // one per physical core on SMT-2 parts
    }
    impl_->threads = threads;
    impl_->scratch.resize(threads);
    for (auto& s : impl_->scratch) { s.resize(cpu_expert_scratch_bytes(geometry) + 64); }
    for (std::uint32_t i = 0; i < threads; ++i) {
        impl_->workers.emplace_back([impl = impl_.get(), i, pin = options.pin_threads] { impl->worker(i, pin); });
    }
}

CpuExpertPool::~CpuExpertPool() {
    {
        std::lock_guard<std::mutex> lock(impl_->mutex);
        impl_->stop = true;
    }
    impl_->wake.notify_all();
    for (auto& t : impl_->workers) { t.join(); }
}

std::uint32_t CpuExpertPool::threads() const noexcept { return impl_->threads; }

void CpuExpertPool::run(const CpuExpertBank& bank, const CpuExpertRound& round) {
    if (round.x == nullptr || round.out == nullptr || round.tokens <= 0) {
        throw std::invalid_argument("cpu_expert_compute: round needs activations, output and tokens");
    }
    for (const CpuExpertJob& job : round.jobs) {
        if (job.token < 0 || job.token >= round.tokens) {
            throw std::invalid_argument("cpu_expert_compute: job token out of range");
        }
    }
    if (round.jobs.empty()) { return; }
    Impl& impl = *impl_;
    if (impl.token_locks.size() < static_cast<std::size_t>(round.tokens)) {
        impl.token_locks.reserve(static_cast<std::size_t>(round.tokens));
        while (impl.token_locks.size() < static_cast<std::size_t>(round.tokens)) {
            impl.token_locks.emplace_back(std::make_unique<std::mutex>());
        }
    }
    std::unique_lock<std::mutex> lock(impl.mutex);
    impl.bank  = &bank;
    impl.round = &round;
    impl.next_job.store(0);
    impl.finished.store(0);
    ++impl.generation;
    impl.wake.notify_all();
    impl.done.wait(lock, [&] { return impl.finished.load() == impl.threads; });
    impl.bank  = nullptr;
    impl.round = nullptr;
}

} // namespace ninfer::ops
