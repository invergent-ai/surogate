// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0

#include "runtime/attention/attention_backend.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <sstream>
#include <stdexcept>

namespace dsl {

AttentionBackendRegistry& AttentionBackendRegistry::instance() {
    static AttentionBackendRegistry singleton;
    return singleton;
}

void AttentionBackendRegistry::add(std::unique_ptr<AttentionBackend> backend) {
    if (!backend) {
        throw std::invalid_argument("AttentionBackendRegistry::add: null backend");
    }
    // Insert in priority-descending order so select() can return on first match.
    const int prio = backend->priority();
    auto pos = std::upper_bound(mBackends.begin(),
                                mBackends.end(),
                                prio,
                                [](int p, const std::unique_ptr<AttentionBackend>& b) { return p > b->priority(); });
    mBackends.insert(pos, std::move(backend));
}

AttentionBackend& AttentionBackendRegistry::select(const AttentionParams& p) const {
    for (const auto& b : mBackends) {
        if (b->supports(p)) {
            if (const char* env = std::getenv("SUROGATE_DEBUG_ATTN_SELECT")) {
                if (env[0] == '1') {
                    static int select_count = 0;
                    if (select_count++ < 80) {
                        std::fprintf(stderr,
                                     "[attn-select] backend=%s prio=%d T=%d Hq=%d Hs=%d window=%d varlen=%s\n",
                                     b->name(),
                                     b->priority(),
                                     p.T,
                                     p.Hq,
                                     p.Hs,
                                     p.window_size,
                                     p.cu_seqlens ? "yes" : "no");
                    }
                }
            }
            return *b;
        }
    }
    std::ostringstream msg;
    msg << "No attention backend supports: "
        << "B=" << p.B << " T=" << p.T << " Hq=" << p.Hq << " Hkv=" << p.Hkv << " Hs=" << p.Hs
        << " window=" << p.window_size << " causal=" << (p.causal ? "yes" : "no")
        << " varlen=" << (p.cu_seqlens ? "yes" : "no") << " dtype=" << static_cast<int>(p.dtype);
    if (!mBackends.empty()) {
        msg << ". Registered backends:";
        for (const auto& b : mBackends) {
            msg << " " << b->name() << "(prio=" << b->priority() << ")";
        }
    } else {
        msg << ". No backends registered.";
    }
    throw std::runtime_error(msg.str());
}

size_t AttentionBackendRegistry::max_workspace_bytes(int B,
                                                     int T,
                                                     int Hq,
                                                     int Hkv,
                                                     int max_hs,
                                                     cudnnHandle_t cudnn_handle,
                                                     cublasHandle_t cublas_handle) const {
    // Probe each backend with a representative params at (max_hs, window=0,
    // no varlen) — the full-attention non-packed case, which is what the
    // persistent cuDNN workspace is sized for. Backends that only match
    // under varlen / sliding-window never consume the shared workspace,
    // so probing at the full-attention shape is sufficient.
    AttentionParams probe;
    probe.B = B;
    probe.T = T;
    probe.Hq = Hq;
    probe.Hkv = Hkv;
    probe.Hs = max_hs;
    probe.window_size = 0;
    probe.causal = true;
    probe.dtype = ETensorDType::BF16;
    probe.cudnn_handle = cudnn_handle;
    probe.cublas_handle = cublas_handle;

    size_t max_bytes = 0;
    for (const auto& b : mBackends) {
        if (!b->supports(probe)) {
            continue;
        }
        max_bytes = std::max(max_bytes, b->workspace_bytes(probe));
    }
    return max_bytes;
}

}  // namespace dsl
