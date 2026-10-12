#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h> // cudaStream_t
#include <cstdint>
#include <array>

namespace sinfer::ops {

// Interleaved three-axis text RoPE. Sections count temporal, height and width pairs.
// Height occupies 1:3*sections[1]:3, width 2:3*sections[2]:3; remaining pairs use time.
// The sections sum to rotary_dim/2. Storage and arithmetic follow rope below.
void rope_interleaved(const Tensor& positions, int rotary_dim, float theta,
                      std::array<int, 3> sections, Tensor& q, Tensor& k, cudaStream_t stream);
void rope_interleaved(const Tensor& positions, int rotary_dim, float theta,
                      std::array<int, 3> sections, Tensor& x, cudaStream_t stream);

/**
 * Applies split-half NeoX RoPE in place. For pair i in [0,rotary_dim/2), angle phi(i,t), and
 * each head:
 *
 *   ideal[i]              = x[i] * cos(phi) - x[i+R/2] * sin(phi)
 *   ideal[i+rotary_dim/2] = x[i+R/2] * cos(phi) + x[i] * sin(phi).
 *
 * Dimensions [rotary_dim,head_dim) are unchanged. Supported modes are:
 *
 * - Text 1-D: positions I32 [T], either head_dim=256 with even 0<rotary_dim<=256, or the
 *   DFlash full-head domain head_dim=rotary_dim=128; phi=positions[t]*theta^(-2*i/rotary_dim).
 * - Text MRoPE: positions I32 [T,3], head_dim=256, rotary_dim=64; pair i uses axis i%3 with
 *   the same frequency as Text 1-D.
 * - Vision 2-D: positions I32 [T,2], head_dim=rotary_dim=D in {64,72,96}; pairs 0..D/4-1 use axis 0 and
 *   pairs D/4..D/2-1 use axis 1, each with local frequency theta^(-2*(i%(D/4))/(D/2)).
 *
 * positions is contiguous and theta is positive and finite. Q/K tensors are BF16
 * [head_dim,heads,T] with positive head counts, contiguous head features and heads, and an optional
 * padded token stride. The registered optimized domains are D256/R64 Text Q/K head geometries
 * 24/4 and 16/2, D128/R128 1-D Text geometry 32/8, plus Vision geometry 16/16. q and k must not
 * overlap one another or positions. The Op mutates only dimensions [0,rotary_dim) of the supplied
 * Q/K tensor storage. The oracle evaluates the rotated dimensions naively in FP64 from the
 * represented inputs. The updated BF16 values are promoted and compared directly with that result;
 * output storage rounding belongs to the Op's numerical criterion, not the oracle. Unrotated
 * dimensions remain bit-exact. Private kernel arithmetic is implementation-defined. The Op uses no
 * workspace or persistent state.
 */
void rope(const Tensor& positions, int rotary_dim, float theta, Tensor& q, Tensor& k,
          cudaStream_t stream);

/**
 * The same, with only the first `active_pairs` pairs rotated and the rest left exactly as they
 * are. `active_pairs == rotary_dim / 2` is the form above.
 *
 * This is a *partial* rotation over the whole head, which is not the same thing as a narrower
 * one. Gemma 4's global layers rotate 64 pairs of a 512-wide head: the pairs are still
 * (i, i + 256), so the rotated and unrotated channels interleave at stride 256. A rotary_dim
 * of 128 would pair i with i+64 -- different partners, and wrong in a way that still produces
 * plausible text.
 *
 * The inert pairs are an exact identity, not an approximation: cos 1 and sin 0 leave both
 * halves bit-for-bit unchanged.
 * `frequency_scale` multiplies the phase; 1/factor implements linear RoPE scaling.
 */
void rope(const Tensor& positions, int rotary_dim, int active_pairs, float theta, Tensor& q,
          Tensor& k, cudaStream_t stream, float frequency_scale = 1.0F);

/**
 * The generic kernel's coefficients, derived once for every position a caller will rotate:
 * `table` is F32 [2, rotary_dim / 2, positions], contiguous -- (cos, sin) per pair, a position's
 * pairs adjacent -- for a one-dimensional rotation of `head_dim`-wide heads with these
 * rotary_dim, active_pairs, theta and frequency_scale. Each entry is the pair the generic kernel
 * computes for that position, bit for bit.
 */
void rope_table(int head_dim, int rotary_dim, int active_pairs, float theta,
                float frequency_scale, Tensor& table, cudaStream_t stream);

/**
 * ops::rope over I32 [T] positions with its coefficients read from a rope_table `table` instead of
 * derived per token. The output is bit for bit what ops::rope writes when it takes the generic
 * kernel -- every geometry but the fixed modes listed above -- for the table's own parameters.
 * Positions must lie in [0, positions of the table); q and k as for ops::rope, head dim
 * table.ne[1] * 2 or wider. Deriving the angles costs double-precision pow and sincos per pair per
 * token, which a GPU with little FP64 pays on every column of a long prompt; the table pays it once
 * per position.
 */
void rope_from_table(const Tensor& positions, const Tensor& table, Tensor& q, Tensor& k,
                     cudaStream_t stream);

// Single-tensor form with the same formula and storage contract. The head count comes directly
// from x; Q versus K role does not change the transformation.
void rope(const Tensor& positions, int rotary_dim, float theta, Tensor& x, cudaStream_t stream);
void rope(const Tensor& positions, int rotary_dim, int active_pairs, float theta, Tensor& x,
          cudaStream_t stream);

/**
 * YaRN context extension (Hugging Face `rope_type: "yarn"`) for a rotary dim of 64 with [T] or
 * [T,3] positions. Pair i turns at `inv_frequency[i]` instead of theta^(-2i/64), and both cos and
 * sin are multiplied by `attention_factor` -- on the rotated channels only, as transformers' and
 * vLLM's rotary embeddings scale them. The angle is taken in double and reduced before sincosf,
 * so it stays exact to the float's precision out to a million positions.
 *
 * For [T,3] positions pair i turns on the height axis when i % 3 == 1 and i / 3 < height_pairs,
 * on the width axis when i % 3 == 2 and i / 3 < width_pairs, and on the temporal axis otherwise:
 * interleaved MRoPE over the model's sections, as ops::rope_interleaved assigns them. -1 (or
 * sections of 11/11/10 and wider) is every pair by i % 3, which is what the fixed kernels compute.
 */
struct RopeYarn {
    static constexpr int kPairs = 32;
    float inv_frequency[kPairs]{};
    float attention_factor     = 1.0F;
    std::int32_t height_pairs  = -1;
    std::int32_t width_pairs   = -1;
};

/// True when `yarn`'s sections put every pair on axis i % 3, the assignment the fixed kernels
/// hard-wire; other sections take the generic kernel.
[[nodiscard]] inline bool rope_yarn_cyclic(const RopeYarn& yarn) {
    return (yarn.height_pairs < 0 || yarn.height_pairs >= (RopeYarn::kPairs + 1) / 3) &&
           (yarn.width_pairs < 0 || yarn.width_pairs >= RopeYarn::kPairs / 3);
}

/**
 * YaRN's coefficients the way transformers' _compute_yarn_parameters and vLLM's
 * YaRNScalingRotaryEmbedding derive them: theta^(-2i/rotary_dim) kept (extrapolated) for the
 * pairs that turn more than `beta_fast` times over `original_context` positions, divided by
 * `factor` (interpolated) for those that turn fewer than `beta_slow` times, and blended linearly
 * between the two correction dims (floor and ceil); attention_factor 0.1 * ln(factor) + 1 unless
 * one is given (> 0). rotary_dim must be 64; factor > 1.
 */
[[nodiscard]] RopeYarn rope_yarn(int rotary_dim, float theta, float factor, int original_context,
                                 float beta_fast = 32.0F, float beta_slow = 1.0F,
                                 float attention_factor = 0.0F);

/// ops::rope with YaRN's coefficients, Q and K or a single tensor, rotary dim 64 within any even
/// head width. The fixed text geometries (head dim 256; Q/K heads 24/4 or 16/2; single tensors
/// of 24, 16, 4 or 2 heads) take their own kernels; any other -- Flash-Next's 24/2 attention,
/// its 128-wide QSA indexer heads -- or non-cyclic sections take the generic kernel, which
/// derives the same coefficients the same way.
void rope(const Tensor& positions, int rotary_dim, float theta, const RopeYarn& yarn, Tensor& q,
          Tensor& k, cudaStream_t stream);
void rope(const Tensor& positions, int rotary_dim, float theta, const RopeYarn& yarn, Tensor& x,
          cudaStream_t stream);

} // namespace sinfer::ops
