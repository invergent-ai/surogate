#include "family/impl/kv_tier/keys.h"

#include "family/impl/frontend/digest.h"

#include <cstring>

namespace sinfer::family::detail::kvkeys {
namespace {

void append_u32(std::vector<std::uint8_t>& out, std::uint32_t value) {
    for (int i = 0; i < 4; ++i) { out.push_back(static_cast<std::uint8_t>(value >> (8 * i))); }
}

Digest sha(const std::vector<std::uint8_t>& bytes) { return family::frontend_internal::sha256(bytes); }

kvtier::Key truncate(const Digest& digest) {
    kvtier::Key key;
    std::memcpy(key.bytes.data(), digest.data(), key.bytes.size());
    return key;
}

} // namespace

Digest namespace_root(std::span<const std::uint8_t> model, const AdapterDigest& adapter, std::string_view tenant) {
    static constexpr char kTag[] = "surogate-kv-v1";
    std::vector<std::uint8_t> bytes(kTag, kTag + sizeof(kTag)); // with its terminating zero
    append_u32(bytes, static_cast<std::uint32_t>(model.size()));
    bytes.insert(bytes.end(), model.begin(), model.end());
    bytes.insert(bytes.end(), adapter.begin(), adapter.end());
    append_u32(bytes, static_cast<std::uint32_t>(tenant.size()));
    bytes.insert(bytes.end(), tenant.begin(), tenant.end());
    return sha(bytes);
}

std::vector<Digest> chunk_chain(const Digest& root, std::span<const std::int32_t> tokens, std::size_t chunks) {
    std::vector<Digest> chain;
    chain.reserve(chunks);
    std::vector<std::uint8_t> bytes;
    bytes.reserve(root.size() + 4 * kvtier::kChunkTokens);
    Digest previous = root;
    for (std::size_t c = 0; c < chunks && (c + 1) * kvtier::kChunkTokens <= tokens.size(); ++c) {
        bytes.assign(previous.begin(), previous.end());
        for (std::size_t i = c * kvtier::kChunkTokens; i < (c + 1) * kvtier::kChunkTokens; ++i) {
            append_u32(bytes, static_cast<std::uint32_t>(tokens[i]));
        }
        previous = sha(bytes);
        chain.push_back(previous);
    }
    return chain;
}

kvtier::Key chunk_key(const Digest& chain, std::optional<std::int32_t> next_token) {
    if (!next_token) { return truncate(chain); }
    std::vector<std::uint8_t> bytes(chain.begin(), chain.end());
    for (const char c : std::string_view("next")) { bytes.push_back(static_cast<std::uint8_t>(c)); }
    append_u32(bytes, static_cast<std::uint32_t>(*next_token));
    return truncate(sha(bytes));
}

kvtier::Key boundary_key(const Digest& chain, std::span<const std::int32_t> tail, BoundaryKind kind,
                         std::uint32_t frontier) {
    std::vector<std::uint8_t> bytes(chain.begin(), chain.end());
    for (const char c : std::string_view("boundary")) { bytes.push_back(static_cast<std::uint8_t>(c)); }
    for (const std::int32_t token : tail) { append_u32(bytes, static_cast<std::uint32_t>(token)); }
    bytes.push_back(static_cast<std::uint8_t>(kind));
    append_u32(bytes, frontier);
    return truncate(sha(bytes));
}

AdapterDigest adapter_digest(std::span<const std::uint8_t> content) {
    const Digest digest = family::frontend_internal::sha256(content);
    AdapterDigest out{};
    std::memcpy(out.data(), digest.data(), out.size());
    return out;
}

} // namespace sinfer::family::detail::kvkeys
