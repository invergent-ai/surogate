# Host KV tier (`--prefix-cache-host-gb`)

## Problem

An agent conversation waits on its GPU lane between turns. Under KV pressure the executor takes
retained lanes back (`find_admission_lane` → evict, `release_idle_kv`, elastic overcommit), and
`evict_retained_lane` dropped the conversation with no host copy. It also cleared the whole host
archive, so even the conversations archived when another request took their lane were lost at the
next eviction. The archive held 512 MiB per GPU, less than one 100k-token conversation of a
Qwen3.5/3.6 hybrid (~1.0 GB of FP8 attention KV plus the MTP head's KV and ~31 MiB of Gated
DeltaNet state per frontier). The next turn then prefilled its whole context again.

## Design

### Objects and keys (`api/kv_tier.h`, `family/impl/kv_tier/keys.*`)

A conversation leaves the GPU as objects of two kinds:

- **Chunk**: the KV of 1,024 tokens, which is 16 pages of 64. It holds every plane of the
  attention pool and, under MTP, of the draft head's pool, in one pool block.
- **Boundary**: the state at one of the engine's own frontiers. These are the end of a finished
  turn (the append frontier) and a rewrite checkpoint. The state is:
  - the recurrent and convolution state of every linear-attention layer;
  - the tail hidden;
  - PLE and draft state where the model has them;
  - the KV pages past the last whole chunk.

Keys are a chained SHA-256:

- `root = SHA256("surogate-kv-v1\0" || namespace)`;
- `d_i = SHA256(d_{i-1} || chunk tokens as LE u32)`;
- a chunk key is the first 16 bytes of `d_i`;
- a boundary key also covers the tail tokens, the frontier kind and the position.

This is SGLang's and Mooncake Conductor's `sha256_raw` chain. A chunk that carries draft-head KV
also names the token after it, because the head's last column was paired with that token.

The namespace is everything that makes two programs' bytes differ for the same tokens:

- the weights (artifact path, size and mtime);
- the KV layout and precision (planes, page bytes, dtype);
- the draft head;
- the pipeline stage;
- the adapter's content digest (SHA-256 over its tensors, set on the slot's `LoraStore` at load),
  not the slot number, which names another adapter after a swap;
- a tenant salt (empty today).

A conversation resumes at the longest boundary whose own object and every chunk before it exist.
Nothing is snapshotted per chunk: a hybrid model can only resume where its recurrent state was
saved.

### Host pool (`family/impl/kv_tier/host_pool.*`)

There is one page-locked pool per process and block size. Every data-parallel replica and pipeline
stage of a model adds its `--prefix-cache-host-gb` to it (mmap, parallel first touch, then
`cudaHostRegister` with the portable flag).

- **Blocks** are fixed-size, one chunk each, so there is no fragmentation. A boundary takes as
  many blocks as it needs, with no piece straddling two.
- **Entries.** A program's snapshot of one conversation is an *entry*: its chunks, its boundary
  and its checkpoint boundary. Chunks are shared by key: conversations with the same opening (the
  system prompt and tools), and the turns of one conversation, store them once.
- **Eviction.** The pool drops whole entries, least recently used first, across all programs.
  Pinned entries go last. An object goes when no entry holds it. The owner of a dropped entry
  learns of it through `take_dropped` and unindexes it before it next plans.
- **Leases.**
  - A plan that will restore an entry leases it, and the restore keeps the lease until its copies
    have run.
  - A writer holds its new entry until its copies are done.
  - Another program may lease the entry only once every object in it is ready.
- **Abandoned writes.** An object whose writer gives up before its bytes are complete is
  poisoned, so no entry that found it meanwhile can restore it.

### Engine side (`family/impl/runtime/host_tier_impl.h`)

- **Background copy.** At every round boundary, with the copy stream free, the oldest retained
  lane without a current snapshot is copied to the host. Only the chunks the pool lacks are
  copied, then the boundary objects. A later eviction of that lane copies nothing.
- **Spill under pressure.** The executor's three pressure paths call `spill_retained_lane`:
  - if the lane is already on the host, the eviction is free;
  - otherwise it is copied at that moment;
  - if it does not fit, it is dropped.
  A lane another request takes is archived the same way.
- **Copy stream.** Every copy runs on the tier's copy stream, ordered after the compute stream's
  work by an event.
  - A background copy holds nothing up. The compute stream waits for it only when the lane it
    reads is reused or evicted first (`fence_lane_copies`).
  - A restore's copies are waited for before the lane's next GPU work, with no host
    synchronisation. The legacy restore ended in `device.synchronize()`.
- **Supersede.** A longer snapshot of a conversation replaces its shorter ones. A lane's next
  snapshot also replaces the one the lane continued, once it has moved past that snapshot's
  rewrite checkpoint (the client resent its own rendering of the answer). Parallel attempts that
  share a prompt keep their own.
- **In-place first.** Every lane may plan a host restore. The plan says it is one
  (`RequestPlanSummary::host_restore`), and at equal reuse the executor resumes a conversation on
  the lane that still holds it. An earlier version kept the host copy away from other lanes while
  the conversation's own lane held it. That brought back the turns the base engine admits on
  another lane with only a shared GPU prefix: 40 of 950 long calls with 10 agents.
- **Invalidation.** An adapter update or idle shrink calls `evict_archived_prefixes`, which drops
  the program's entries at once.
- **Default unchanged.** Without the flag, the 512 MiB archive behaves as before, except that an
  eviction no longer wipes it.

### Back-end interface (for Mooncake, a file tier)

`api/kv_tier.h` defines a C table, `sinfer_kvtier_v1`, that a shared library exports through
`sinfer_kvtier_v1_open(config)`. The table has these calls:

| Call | Meaning |
|---|---|
| `register_host(base, bytes)` | The engine's page-locked memory, once. |
| `contains(objects, n, present)` | Batch existence check. The answer may be stale. |
| `put(objects, spans, n)` → ticket | Store from registered memory, asynchronously. |
| `get(objects, spans, n)` → ticket | Fill registered memory, asynchronously. |
| `poll(ticket)` | `PENDING`, `DONE` or `FAILED`. Never blocks. |
| `erase(keys, n)` | Drop objects. |
| `set_listener(fn, ctx)` | Report objects the back end stored or dropped by itself. |
| `close()` | Release the back end. |

`CBackend` adapts a C table to the C++ `kvtier::Backend` interface. `HostKvPool` implements the
same interface and is the first back end; `host_pool_c_table` exposes it through the C ABI, and
the unit test drives it that way.

The engine calls the pool's zero-copy API: it acquires blocks, its copies write straight into them,
and it then marks the objects ready. An external back end would sit behind the pool as a second
level:

- a pool eviction puts the entry's objects there instead of dropping them;
- a plan that misses the pool asks `contains`, and on a hit `get`s the objects into fresh blocks.

That needs an executor state to wait in, because a remote get is not immediate, so it is not built
yet. The rules the interface implies:

- **Restorable length:** as above.
- **Lifetime:** pages are freed only after the copy that reads them has run.
- **Adapter swaps:** a swap changes the namespace, so stale objects become unreachable.

## Not built

- **Cross-replica lookup.** The pool is shared, but each program indexes only its own snapshots.
  The router keeps conversations on their replica, so a moved conversation misses.
- **Routing by estimated TTFT** (restore cost against queue wait).
- **A layer-wise restore** overlapped with the first chunk.
- **An NVMe or Mooncake back end.**
- **Boundaries at the shared-prefix capture point.**
