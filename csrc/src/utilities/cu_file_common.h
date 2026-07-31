// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//
// Pieces shared by the cuFile (GPUDirect Storage) and POSIX read paths. Both are
// reachable in a cuFile-enabled build: GDS is unavailable on network/FUSE-backed
// filesystems (Modal volumes, WSL, most container storage drivers), so open_cufile()
// degrades to the POSIX path at runtime rather than failing the load.
//

#ifndef SUROGATE_SRC_UTILITIES_CU_FILE_COMMON_H
#define SUROGATE_SRC_UTILITIES_CU_FILE_COMMON_H

#include <cstddef>
#include <string_view>

#include "utilities/dtype.h"

//! \brief Read `[begin, end)` from `fd` into device memory through a pinned host staging buffer.
//! \param fd POSIX file descriptor opened for reading (buffered, i.e. without O_DIRECT).
//! \param d_target Destination device pointer, valid for `end - begin` bytes.
//! \param begin Start offset in the file (inclusive), in bytes.
//! \param end End offset in the file (exclusive), in bytes.
//! \param file_name Name used for diagnostics only.
//! \throws std::logic_error If `end < begin`.
//! \throws std::runtime_error On pread() failure, short reads, or CUDA copy failures.
void posix_read_bytes(int fd,
                      std::byte* d_target,
                      std::ptrdiff_t begin,
                      std::ptrdiff_t end,
                      std::string_view file_name);

//! \brief Convert `size` elements from `s_type` to `t_type`; both buffers are device pointers.
//! \throws std::runtime_error If the conversion pair is unsupported.
void convert_tensor_dispatch(std::byte* target,
                             const std::byte* source,
                             std::size_t size,
                             ETensorDType t_type,
                             ETensorDType s_type);

//! \brief True when GDS must not be used: `SUROGATE_DISABLE_CUFILE` set to anything but `0`.
bool cufile_disabled_by_env();

#endif  //SUROGATE_SRC_UTILITIES_CU_FILE_COMMON_H
