// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// Copyright (c) 2025, IST Austria, developed by Erik Schultheis
// SPDX-License-Identifier: Apache-2.0
//

#include <cstdio>
#include <cstring>
#include <algorithm>  // std::min

#include <cufile.h>
#include <fcntl.h>
#include <unistd.h>
#include <fmt/core.h>

#include "cu_file.h"
#include "cu_file_common.h"

#include "kernels/kernels.h"
#include "utilities/dtype.h"
#include "utilities/utils.h"

namespace {

std::string cufile_error_str(CUfileError_t status) {
    return fmt::format("{} (err={}, cu_err={})",
                       CUFILE_ERRSTR(status.err),
                       static_cast<int>(status.err),
                       static_cast<int>(status.cu_err));
}

/**
 * @brief Whether GPUDirect Storage can be used in this process, evaluated once.
 *
 * cuFile needs the nvidia-fs kernel module and a filesystem that supports it. Neither
 * holds on network- or FUSE-backed storage (Modal volumes, WSL, many container storage
 * drivers). Any of these is a reason to read the checkpoint differently, not to fail the
 * run, so we latch the answer and let open_cufile() adapt.
 *
 * The nvidia-fs probe below must happen before any libcufile entry point. Without the
 * module libcufile does not fail: it logs "running in compatible mode" and, on container
 * filesystems, then spins inside the driver-open path instead of returning — hanging the
 * load with no error. Checking the same file it checks keeps us out of that path
 * entirely. The driver-open and properties checks that follow are belt-and-braces for
 * hosts where nvidia-fs is present but GDS still is not usable.
 */
bool gds_available() {
    static const bool available = [] {
        if (cufile_disabled_by_env()) {
            fprintf(stderr, "cuFile: disabled via SUROGATE_DISABLE_CUFILE, using buffered reads\n");
            return false;
        }
        if (::access("/proc/driver/nvidia-fs/devcount", F_OK) != 0) {
            fprintf(stderr, "cuFile: nvidia-fs kernel module not loaded; using buffered reads\n");
            return false;
        }
        CUfileError_t status = cuFileDriverOpen();
        if (status.err != CU_FILE_SUCCESS) {
            fprintf(stderr,
                    "cuFile: GPUDirect Storage unavailable (%s); falling back to buffered reads\n",
                    cufile_error_str(status).c_str());
            return false;
        }

        // cuFileDriverOpen() succeeding is not enough. With no nvidia-fs kernel module,
        // libcufile logs "running in compatible mode" and keeps going, reporting nvfs
        // version 0. Compat mode is POSIX I/O behind a bounce buffer, so it buys nothing
        // over the path below — and on 9p/FUSE mounts (Modal volumes) registering a handle
        // there spins instead of returning, hanging the load outright. Refuse it.
        CUfileDrvProps_t props;
        std::memset(&props, 0, sizeof(props));
        CUfileError_t props_status = cuFileDriverGetProperties(&props);
        if (props_status.err != CU_FILE_SUCCESS) {
            fprintf(stderr,
                    "cuFile: could not query driver properties (%s); falling back to buffered reads\n",
                    cufile_error_str(props_status).c_str());
            return false;
        }
        if (props.nvfs.major_version == 0) {
            fprintf(stderr,
                    "cuFile: nvidia-fs absent, driver is in compatibility mode; using buffered reads\n");
            return false;
        }
        return true;
    }();
    return available;
}

//! Emit a per-process warning the first time a file drops off the GDS path.
void warn_gds_fallback_once(std::string_view reason, std::string_view file_name) {
    static bool warned = false;
    if (warned) return;
    warned = true;
    fprintf(stderr,
            "cuFile: %.*s for %.*s; falling back to buffered reads for this and subsequent files\n",
            static_cast<int>(reason.size()),
            reason.data(),
            static_cast<int>(file_name.size()),
            file_name.data());
}

//! Open \p file_name without O_DIRECT, for reads through posix_read_bytes().
//! \throws std::runtime_error If the file cannot be opened.
int open_buffered_fd(const std::string& file_name) {
    int fd = open(file_name.c_str(), O_RDONLY | O_CLOEXEC);
    if (fd < 0) {
        throw std::runtime_error(
            fmt::format("posix open error ({}) for file {}: {}", errno, file_name, strerror(errno)));
    }
    return fd;
}

//! Build a ref on the buffered path; a null handle marks it as non-GDS.
cuFileRef open_buffered(std::string file_name) {
    int fd = open_buffered_fd(file_name);
    return {nullptr, fd, std::move(file_name)};
}

}  // namespace

/**
 * @brief Open a file for tensor reads, registering it with cuFile when GDS is usable.
 *
 * Tries the GDS path — O_DIRECT open plus cuFileHandleRegister() — and falls back to a
 * buffered descriptor if the driver is missing, the filesystem rejects O_DIRECT, or the
 * handle cannot be registered. Only a failure to open the file at all is fatal.
 *
 * @param file_name Path to the file to open.
 * @return A cuFileRef on either the GDS or the POSIX path; see cuFileRef::uses_gds().
 * @throws std::runtime_error If the file cannot be opened by either path.
 */
cuFileRef open_cufile(std::string file_name) {
    if (!gds_available()) {
        return open_buffered(std::move(file_name));
    }

    int fd = open(file_name.c_str(), O_RDONLY | O_DIRECT | O_CLOEXEC);
    if (fd < 0) {
        // Typically EINVAL: the backing filesystem has no O_DIRECT support.
        warn_gds_fallback_once(fmt::format("O_DIRECT open failed ({}: {})", errno, strerror(errno)), file_name);
        return open_buffered(std::move(file_name));
    }

    CUfileDescr_t descr;
    CUfileHandle_t handle;
    std::memset(&descr, 0, sizeof(CUfileDescr_t));
    descr.handle.fd = fd;
    descr.type = CU_FILE_HANDLE_TYPE_OPAQUE_FD;
    CUfileError_t status = cuFileHandleRegister(&handle, &descr);
    if (status.err != CU_FILE_SUCCESS) {
        close(fd);
        warn_gds_fallback_once(fmt::format("handle registration failed ({})", cufile_error_str(status)), file_name);
        return open_buffered(std::move(file_name));
    }

    return {handle, fd, std::move(file_name)};
}

/**
 * @brief Read a byte range from a registered cuFile handle into device memory.
 *
 * The interval is treated as [begin, end) in file offsets (bytes). The target is
 * expected to be a device-accessible pointer suitable for cuFileRead.
 *
 * @param handle Registered cuFile handle.
 * @param target Destination buffer (device pointer) receiving (end - begin) bytes.
 * @param begin Start offset in the file (inclusive), in bytes.
 * @param end End offset in the file (exclusive), in bytes. Must be >= begin.
 * @param file_name File name used for diagnostics only.
 * @throws std::logic_error if end < begin.
 * @throws std::runtime_error on cuFileRead errors or short reads.
 */
void cufile_read_bytes(CUfileHandle_t handle,
                       std::byte* target,
                       std::ptrdiff_t begin,
                       std::ptrdiff_t end,
                       std::string_view file_name) {
    if (end < begin) {
        throw std::logic_error(fmt::format("Invalid range {} - {} in cufile_read_bytes for {}", begin, end, file_name));
    }
    ssize_t ret = cuFileRead(handle, target, end - begin, begin, 0);
    if (ret < 0) {
        if (ret == -1) {
            throw std::runtime_error(fmt::format("cufile read error ({}) for file {}, range {} - {}: {}",
                                                 errno,
                                                 file_name,
                                                 begin,
                                                 end,
                                                 strerror(errno)));
        } else {
            throw std::runtime_error(fmt::format("cufile read error ({}: {}) for file {}, range {} - {}",
                                                 -ret,
                                                 CUFILE_ERRSTR(-ret),
                                                 file_name,
                                                 begin,
                                                 end));
        }
    } else if (ret != end - begin) {
        throw std::runtime_error(
            fmt::format("cufile read error for file {}: expected {} bytes, got {}", file_name, end - begin, ret));
    }
}

cuFileRef::~cuFileRef() noexcept {
    // Deregister before closing: cuFile holds the descriptor for the lifetime of the handle.
    if (mHandle != nullptr) {
        cuFileHandleDeregister(mHandle);
        mHandle = nullptr;
    }
    if (mFileDescriptor >= 0) {
        close(mFileDescriptor);
        mFileDescriptor = -1;
    }
}

/**
 * @brief Read a byte range from the underlying file into device memory.
 *
 * The interval is treated as [begin, end) in file offsets (bytes). Dispatches to
 * GPUDirect Storage or the buffered POSIX path depending on how the file was opened.
 *
 * @param target Destination buffer (device pointer) receiving (end - begin) bytes.
 * @param begin Start offset in the file (inclusive), in bytes.
 * @param end End offset in the file (exclusive), in bytes. Must be >= begin.
 * @throws std::logic_error / std::runtime_error as propagated from the selected read path.
 */
void cuFileRef::read_bytes(std::byte* target, std::ptrdiff_t begin, std::ptrdiff_t end) {
    if (uses_gds()) {
        try {
            cufile_read_bytes(mHandle, target, begin, end, mFileName);
            return;
        } catch (const std::runtime_error& e) {
            // Some filesystems accept the handle registration and only fail once DMA is
            // actually attempted. A partial read may have landed in `target`; the POSIX
            // retry below rewrites the whole range, so that is harmless.
            warn_gds_fallback_once(fmt::format("read failed ({})", e.what()), mFileName);
            degrade_to_buffered();
        }
    }
    posix_read_bytes(mFileDescriptor, target, begin, end, mFileName);
}

/**
 * @brief Deregister the cuFile handle and reopen the file with buffered I/O.
 *
 * Reopening is required rather than reusing the descriptor: it was opened O_DIRECT, which
 * constrains every read to aligned offsets and sizes that the POSIX path does not honour.
 *
 * @throws std::runtime_error If the file cannot be reopened.
 */
void cuFileRef::degrade_to_buffered() {
    if (mHandle != nullptr) {
        cuFileHandleDeregister(mHandle);
        mHandle = nullptr;
    }
    if (mFileDescriptor >= 0) {
        close(mFileDescriptor);
        mFileDescriptor = -1;
    }
    mFileDescriptor = open_buffered_fd(mFileName);
}
