// KV tier back ends: where conversation KV and recurrent state go when they leave the GPU.
//
// The engine keeps a finished conversation it can no longer hold on the GPU as objects of two
// kinds, keyed by a chained SHA-256 over its tokens:
//
//   Chunk     the KV of 1,024 tokens (16 pages of 64), every plane of the attention pool and,
//             under MTP, of the draft head's pool. A chunk's key covers every token before its
//             end (and, with a draft head, the token after it, which the head's last column was
//             paired with), so conversations that share a prefix share its chunks.
//   Boundary  the state at one of the engine's own frontiers -- the end of a finished turn or a
//             rewrite checkpoint: the Gated DeltaNet recurrent and convolution state of every
//             linear-attention layer, the tail hidden the next token is sampled from, PLE and
//             draft state where the model has them, and the KV pages past the last whole chunk.
//
// A conversation resumes at the longest boundary whose own object and every chunk before it
// exist (a hybrid model can only resume where its recurrent state was saved; nothing is ever
// snapshotted per chunk).
//
// The engine's own host RAM tier (HostKvPool, --prefix-cache-host-gb) is the first back end.
// External stores -- Mooncake, a file tier -- implement the C table below in a shared library of
// their own, so their dependencies stay out of the engine; `CBackend` adapts one to the C++
// interface the engine calls.
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <span>

extern "C" {

#define SINFER_KVTIER_ABI_VERSION 1u

/// 128-bit prefix of a chained SHA-256 (see `sinfer::kvtier::chunk_keys`).
typedef struct sinfer_kvtier_key {
    uint8_t bytes[16];
} sinfer_kvtier_key;

enum { SINFER_KVTIER_CHUNK = 0, SINFER_KVTIER_BOUNDARY = 1 };

typedef struct sinfer_kvtier_object {
    sinfer_kvtier_key key;
    uint8_t kind;          ///< SINFER_KVTIER_CHUNK or SINFER_KVTIER_BOUNDARY
    uint32_t first_token;  ///< chunk: its first token; boundary: the first token past the last whole chunk
    uint32_t tokens;       ///< chunk: tokens it covers; boundary: the frontier minus `first_token`
    uint64_t bytes;
} sinfer_kvtier_object;

typedef struct sinfer_kvtier_span {
    void* data;
    size_t bytes;
} sinfer_kvtier_span;

typedef struct sinfer_kvtier_caps {
    uint8_t persistent;           ///< objects survive the process
    uint8_t cross_process;        ///< other processes see the same objects
    uint8_t zero_copy_registered; ///< put/get read and write registered host memory directly
    uint8_t async;                ///< put/get return before the bytes have moved
    uint32_t chunk_tokens;        ///< 1024
} sinfer_kvtier_caps;

/// Results of `poll`.
enum { SINFER_KVTIER_PENDING = 0, SINFER_KVTIER_DONE = 1, SINFER_KVTIER_FAILED = -1 };

/// A back end, as a shared library exports it (`sinfer_kvtier_v1_open`). Every call may come from
/// any engine thread; none may block on the network. `ctx` is the back end's own.
typedef struct sinfer_kvtier_v1 {
    uint32_t abi_version; ///< SINFER_KVTIER_ABI_VERSION
    void* ctx;
    sinfer_kvtier_caps (*caps)(void* ctx);
    /// The engine's page-locked host memory, once, before any put or get names it.
    int (*register_host)(void* ctx, void* base, size_t bytes);
    /// Which objects the back end holds; may be stale by the time the engine acts on it.
    void (*contains)(void* ctx, const sinfer_kvtier_object* objects, size_t count, uint8_t* present);
    /// Stores `count` objects from the spans (registered memory, one span per object). Returns a
    /// ticket for `poll`; the spans must stay valid until it is done.
    uint64_t (*put)(void* ctx, const sinfer_kvtier_object* objects, const sinfer_kvtier_span* spans, size_t count);
    /// Fills the spans (registered memory) with the objects. A ticket, as for `put`.
    uint64_t (*get)(void* ctx, const sinfer_kvtier_object* objects, const sinfer_kvtier_span* spans, size_t count);
    /// SINFER_KVTIER_PENDING, _DONE or _FAILED; never blocks. A done or failed ticket is forgotten.
    int (*poll)(void* ctx, uint64_t ticket);
    void (*erase)(void* ctx, const sinfer_kvtier_key* keys, size_t count);
    /// `stored_or_removed(listener_ctx, objects, n, stored)` for objects the back end stored or
    /// dropped by itself: feeds the engine's index and metrics.
    void (*set_listener)(void* ctx,
                         void (*stored_or_removed)(void* listener_ctx, const sinfer_kvtier_object* objects,
                                                   size_t count, int stored),
                         void* listener_ctx);
    void (*close)(void* ctx);
} sinfer_kvtier_v1;

/// What a back-end library exports. `config` is the back end's own (a path, a URL).
typedef const sinfer_kvtier_v1* (*sinfer_kvtier_v1_open_fn)(const char* config);

} // extern "C"

namespace sinfer::kvtier {

inline constexpr std::uint32_t kChunkTokens = 1024;

struct Key {
    std::array<std::uint8_t, 16> bytes{};
    friend bool operator==(const Key&, const Key&) = default;
};

struct KeyHash {
    std::size_t operator()(const Key& key) const noexcept {
        std::size_t h = 0;
        std::memcpy(&h, key.bytes.data(), sizeof(h));
        return h;
    }
};

enum class Kind : std::uint8_t { Chunk = SINFER_KVTIER_CHUNK, Boundary = SINFER_KVTIER_BOUNDARY };

struct ObjectRef {
    Key key;
    Kind kind                = Kind::Chunk;
    std::uint32_t first_token = 0;
    std::uint32_t tokens      = 0;
    std::uint64_t bytes       = 0;
};

using Caps   = sinfer_kvtier_caps;
using Span   = sinfer_kvtier_span;
using Ticket = std::uint64_t;
using Listener = std::function<void(std::span<const ObjectRef> objects, bool stored)>;

/// The interface the engine calls. Implementations: HostKvPool (in process), CBackend (a library).
class Backend {
public:
    virtual ~Backend() = default;
    [[nodiscard]] virtual Caps caps() const = 0;
    virtual bool register_host(void* base, std::size_t bytes) = 0;
    virtual void contains(std::span<const ObjectRef> objects, std::span<std::uint8_t> present) = 0;
    [[nodiscard]] virtual Ticket put(std::span<const ObjectRef> objects, std::span<const Span> spans) = 0;
    [[nodiscard]] virtual Ticket get(std::span<const ObjectRef> objects, std::span<const Span> spans) = 0;
    [[nodiscard]] virtual int poll(Ticket ticket) = 0;
    virtual void erase(std::span<const Key> keys) = 0;
    virtual void set_listener(Listener listener) = 0;
};

[[nodiscard]] inline sinfer_kvtier_object to_c(const ObjectRef& ref) noexcept {
    sinfer_kvtier_object out{};
    std::memcpy(out.key.bytes, ref.key.bytes.data(), 16);
    out.kind        = static_cast<std::uint8_t>(ref.kind);
    out.first_token = ref.first_token;
    out.tokens      = ref.tokens;
    out.bytes       = ref.bytes;
    return out;
}

[[nodiscard]] inline ObjectRef from_c(const sinfer_kvtier_object& object) noexcept {
    ObjectRef out;
    std::memcpy(out.key.bytes.data(), object.key.bytes, 16);
    out.kind        = static_cast<Kind>(object.kind);
    out.first_token = object.first_token;
    out.tokens      = object.tokens;
    out.bytes       = object.bytes;
    return out;
}

} // namespace sinfer::kvtier
