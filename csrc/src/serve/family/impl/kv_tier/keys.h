// Keys of the KV tier (api/kv_tier.h): a chained SHA-256 over 1,024-token chunks.
//
//   root = SHA256("surogate-kv-v1\0" || namespace)
//   d_i  = SHA256(d_{i-1} || the chunk's tokens as little-endian u32)
//   chunk key    = first 16 bytes of d_i, or of SHA256(d_i || "next" || token) when the chunk also
//                  holds a draft head's KV, whose last column was paired with the next token
//   boundary key = first 16 bytes of SHA256(d_n || tail tokens || kind || frontier)
//
// This is the chain SGLang and Mooncake's conductor use ("sha256_raw"), so an external index can
// read our keys without a profile of its own. The namespace is everything that makes two programs'
// KV bytes different for the same tokens: the weights, the KV layout and precision, the draft
// head, the pipeline stage, the adapter's content, and a tenant salt.
#pragma once

#include "api/kv_tier.h"

#include <array>
#include <cstdint>
#include <optional>
#include <span>
#include <string_view>
#include <vector>

namespace sinfer::family::detail::kvkeys {

using Digest = std::array<std::uint8_t, 32>;
using AdapterDigest = std::array<std::uint8_t, 16>;

enum class BoundaryKind : std::uint8_t { AppendFrontier = 0, RewriteCheckpoint = 1, SharedPrefix = 2 };

/// The root of every chain in one namespace. `model` is the program's fingerprint of weights,
/// layout and precision; `adapter` is zero for the base model.
[[nodiscard]] Digest namespace_root(std::span<const std::uint8_t> model, const AdapterDigest& adapter,
                                    std::string_view tenant = {});

/// d_1 .. d_n over the first `chunks` whole chunks of `tokens`.
[[nodiscard]] std::vector<Digest> chunk_chain(const Digest& root, std::span<const std::int32_t> tokens,
                                              std::size_t chunks);

[[nodiscard]] kvtier::Key chunk_key(const Digest& chain, std::optional<std::int32_t> next_token);

/// `chain` is d_n of the last whole chunk before the boundary (the root when there is none).
[[nodiscard]] kvtier::Key boundary_key(const Digest& chain, std::span<const std::int32_t> tail, BoundaryKind kind,
                                       std::uint32_t frontier);

/// The adapter digest the namespace takes: the first 16 bytes of SHA-256 over the bytes given.
[[nodiscard]] AdapterDigest adapter_digest(std::span<const std::uint8_t> content);

} // namespace sinfer::family::detail::kvkeys
