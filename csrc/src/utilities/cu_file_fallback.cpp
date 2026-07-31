// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// Copyright (c) 2025, IST Austria, developed by Erik Schultheis
// SPDX-License-Identifier: Apache-2.0
//
// This file implements a fallback path, to be used in cases in which cuFile is not available
// at build time (WSL, AMD, toolkits without the cuFile component). The read implementation
// itself lives in cu_file_common.cpp and is shared with the cuFile-enabled build, which
// degrades to it at runtime when GPUDirect Storage is unusable.
//

#include <cerrno>
#include <cstring>

#include <fcntl.h>
#include <unistd.h>
#include <fmt/core.h>

#include "cu_file.h"
#include "cu_file_common.h"

/**
 * @brief Open a file for reading via the POSIX path (no cuFile).
 *
 * @param file_name Path to the file to open.
 * @return A cuFileRef with a POSIX file descriptor and a null cuFile handle.
 *
 * @throws std::runtime_error If the file cannot be opened (POSIX open() error).
 */
cuFileRef open_cufile(std::string file_name) {
    int fd = open(file_name.c_str(), O_RDONLY | O_CLOEXEC);
    if (fd < 0) {
        throw std::runtime_error(
            fmt::format("posix open error ({}) for file {}: {}", errno, file_name, strerror(errno)));
    }

    return {nullptr, fd, std::move(file_name)};
}

/**
 * @brief Destructor; closes the underlying POSIX file descriptor if open.
 *
 * Never throws.
 */
cuFileRef::~cuFileRef() noexcept {
    if (mFileDescriptor >= 0) {
        close(mFileDescriptor);
        mFileDescriptor = -1;
    }
}

/**
 * @brief Read a byte range from the file into device memory.
 *
 * Range semantics: [begin, end) in bytes.
 *
 * @param target Destination pointer in device memory; must be valid for (end - begin) bytes.
 * @param begin Starting byte offset within the file (inclusive).
 * @param end Ending byte offset within the file (exclusive).
 *
 * @throws std::logic_error If @p end < @p begin.
 * @throws std::runtime_error On POSIX/CUDA errors or short reads.
 */
void cuFileRef::read_bytes(std::byte* target, std::ptrdiff_t begin, std::ptrdiff_t end) {
    posix_read_bytes(mFileDescriptor, target, begin, end, mFileName);
}
