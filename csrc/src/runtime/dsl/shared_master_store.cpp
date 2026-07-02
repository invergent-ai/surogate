// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0

#include "runtime/dsl/shared_master_store.h"

#include <cstdlib>
#include <iostream>

#include <cuda_runtime.h>

namespace dsl {

void* SharedMasterStore::reserve(const std::string& name, std::size_t bytes) {
    std::lock_guard<std::mutex> lk(mMutex);
    auto it = mEntries.find(name);
    if (it != mEntries.end()) {
        return it->second.data;
    }
    void* p = std::malloc(bytes);  // pageable; faulted in by the read, pinned after
    if (!p) {
        throw std::runtime_error("SharedMasterStore: malloc failed for '" + name + "'");
    }
    mEntries.emplace(name, Entry{p, bytes, /*claimed=*/false, /*populated=*/false});
    return p;
}

bool SharedMasterStore::try_claim(const std::string& name) {
    std::lock_guard<std::mutex> lk(mMutex);
    auto& e = mEntries.at(name);
    if (e.claimed || e.populated) return false;
    e.claimed = true;
    return true;
}

void SharedMasterStore::register_and_finish(const std::string& name) {
    void* data;
    std::size_t bytes;
    {
        std::lock_guard<std::mutex> lk(mMutex);
        auto& e = mEntries.at(name);
        data = e.data;
        bytes = e.bytes;
    }
    // Page-lock the now-resident buffer so per-block streaming can cudaMemcpyAsync it.
    const cudaError_t st = cudaHostRegister(data, bytes, cudaHostRegisterDefault);
    if (st != cudaSuccess && st != cudaErrorHostMemoryAlreadyRegistered) {
        throw std::runtime_error("SharedMasterStore: cudaHostRegister failed for '" + name +
                                 "': " + cudaGetErrorString(st));
    }
    {
        std::lock_guard<std::mutex> lk(mMutex);
        mEntries.at(name).populated = true;
    }
    mCond.notify_all();
}

void SharedMasterStore::wait_populated(const std::string& name) {
    std::unique_lock<std::mutex> lk(mMutex);
    mCond.wait(lk, [&] { return mEntries.at(name).populated; });
}

bool SharedMasterStore::try_claim_fp8(const std::string& name) {
    std::lock_guard<std::mutex> lk(mMutex);
    auto& e = mEntries.at(name);
    if (e.fp8_claimed || e.fp8_populated) return false;
    e.fp8_claimed = true;
    return true;
}

void SharedMasterStore::finish_fp8(const std::string& name) {
    {
        std::lock_guard<std::mutex> lk(mMutex);
        mEntries.at(name).fp8_populated = true;
    }
    mCond.notify_all();
}

void SharedMasterStore::wait_fp8(const std::string& name) {
    std::unique_lock<std::mutex> lk(mMutex);
    mCond.wait(lk, [&] { return mEntries.at(name).fp8_populated; });
}

bool SharedMasterStore::has(const std::string& name) const {
    std::lock_guard<std::mutex> lk(mMutex);
    return mEntries.find(name) != mEntries.end();
}

void SharedMasterStore::clear() {
    std::lock_guard<std::mutex> lk(mMutex);
    for (auto& [name, e] : mEntries) {
        if (e.data) {
            if (e.populated) cudaHostUnregister(e.data);
            std::free(e.data);
        }
    }
    mEntries.clear();
}

SharedMasterStore& shared_master_store() {
    static SharedMasterStore instance;
    return instance;
}

}  // namespace dsl
