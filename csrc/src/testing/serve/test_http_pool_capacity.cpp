// The HTTP connection backlog is not bounded by the serving capacity. Model-free and GPU-free.
//
// The pool used to take one number for both jobs: `max_concurrency + max_pending_requests` became
// the worker count AND the queue bound. cpp-httplib does not reject a connection past that bound,
// it calls shutdown_socket + close_socket with no response (httplib.h, the `!task_queue->enqueue`
// branch of the accept loop), so a caller reads a closed socket and the server logs nothing.
//
// GRPO met this at the end of a run: a final eval fanned out to 64 concurrent rollouts against a
// pool sized for `--max-num-seqs 8`, which is 24 queued and 25 workers, so 49 connections. The
// orchestrator's weight-update POST was one of the 15 over the line, and a run that had trained
// and written its adapter exited 1. Reproduced without a GPU at exactly 49.
//
// How many connections arrive is the caller's decision, not ours: the rollout client is
// configured for up to 8192, and a colocated judge adds its own on the same base URL. So the
// queue is unbounded, as speech/server.cpp and encoder/embedding_server.cpp already leave it. The
// worker count stays derived from the serving capacity, because a worker is a thread.
#include "serve/http_server.h"

#include <cassert>
#include <cstdio>

using sinfer::serve::HttpPoolSizes;
using sinfer::serve::http_pool_sizes;
using sinfer::serve::extra_model_options;
using sinfer::serve::ServeOptions;

namespace {

ServeOptions options_for(std::uint32_t max_concurrency, std::uint32_t max_pending_requests) {
    ServeOptions options;
    options.max_concurrency = max_concurrency;
    options.max_pending_requests = max_pending_requests;
    return options;
}

} // namespace

int main() {
    // The configuration that lost the weight update. No connection count may be refused now, and
    // the worker count is unchanged from what it always was.
    {
        const HttpPoolSizes sizes = http_pool_sizes(options_for(8, 16));
        assert(sizes.queued == 0 && "0 is cpp-httplib's unbounded; any bound reinstates the drop");
        assert(sizes.workers == 25 && "a worker is a thread, so it stays derived");
    }

    // A worker per connection would be the other way to never drop one, and it is the trap: this
    // must stay a handful of threads, not one per caller.
    {
        const HttpPoolSizes sizes = http_pool_sizes(options_for(1, 1));
        assert(sizes.workers == 3);
        assert(sizes.queued == 0);
    }

    // Extra models each add their own serving capacity, so they add workers.
    {
        ServeOptions options = options_for(8, 16);
        options.extra_models.emplace_back();
        options.extra_models.back().max_num_seqs = 600;
        const HttpPoolSizes sizes = http_pool_sizes(options);
        assert(sizes.workers == 8 + 16 + 600 + 16 + 1 && "an extra model adds its own capacity");
        assert(sizes.queued == 0);
    }

    // `max_num_seqs == 0` on an extra model means "same as the primary" (serve_options.h), and the
    // worker count has to honour that rather than add nothing.
    {
        ServeOptions options = options_for(8, 16);
        options.extra_models.emplace_back(); // max_num_seqs defaults to 0
        const HttpPoolSizes sizes = http_pool_sizes(options);
        assert(sizes.workers == 8 + 16 + 8 + 16 + 1 && "0 inherits the primary's max_num_seqs");
    }

    // The worker count restates `extra_model_options()`'s inheritance rule rather than calling it,
    // because it needs the sum and not the options. So pin the two against each other: if someone
    // changes how an extra model inherits pending capacity, the executor's `max_outstanding_` and
    // the worker count drift apart with nothing to say so, and that equality is the whole reason
    // an unbounded queue cannot grow without limit.
    {
        for (const std::uint32_t inherited : {std::uint32_t{0}, std::uint32_t{4}, std::uint32_t{600}}) {
            ServeOptions options = options_for(8, 16);
            options.extra_models.emplace_back();
            options.extra_models.back().max_num_seqs = inherited;
            const ServeOptions derived = extra_model_options(options, options.extra_models.back());
            const std::size_t outstanding = static_cast<std::size_t>(options.max_concurrency) +
                                            options.max_pending_requests +
                                            static_cast<std::size_t>(derived.max_concurrency) +
                                            derived.max_pending_requests;
            assert(http_pool_sizes(options).workers == outstanding + 1 &&
                   "a worker per admissible request, plus one, or the queue can outgrow the pool");
        }
    }

    std::printf("ok: the backlog is unbounded, the worker count is not\n");
    return 0;
}
