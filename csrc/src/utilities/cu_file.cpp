// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// Copyright (c) 2025, IST Austria, developed by Erik Schultheis
// SPDX-License-Identifier: Apache-2.0
//

#include <cstring>
#include <algorithm> // std::min

#include <cufile.h>
#include <fcntl.h>
#include <unistd.h>
#include <fmt/core.h>

#include "cu_file.h"

#include "kernels/kernels.h"
#include "utilities/dtype.h"
#include "utilities/utils.h"

/**
 * @brief Open a file and register it with cuFile (GPUDirect Storage).
 *
 * Calls cuFileDriverOpen(), opens the file with O_DIRECT for direct I/O, and
 * registers the resulting file descriptor as a cuFile handle.
 *
 * @param file_name Path to the file to open.
 * @return A cuFileRef containing the registered CUfile handle, file descriptor, and stored file name.
 * @throws std::runtime_error on driver open failure, file open failure, or handle registration failure.
 */
cuFileRef open_cufile(std::string file_name) {
    auto status = cuFileDriverOpen();
    if (status.err != CU_FILE_SUCCESS) {
        throw std::runtime_error("cufile driver open error");
    }

    int fd = open(file_name.c_str(), O_RDONLY | O_DIRECT);
    if (fd < 0) {
        throw std::runtime_error(fmt::format("cufile file open error ({}) for file {}: {}", errno, file_name, strerror(errno)));
    }

    // create cufile handle
    CUfileDescr_t descr;
    CUfileHandle_t handle;
    std::memset(&descr, 0, sizeof(CUfileDescr_t));
    descr.handle.fd = fd;
    descr.type = CU_FILE_HANDLE_TYPE_OPAQUE_FD;
    status = cuFileHandleRegister(&handle, &descr);
    if (status.err != CU_FILE_SUCCESS) {
        throw std::runtime_error("cufile register error");
    }
    return {handle, fd, std::move(file_name)};
}

/**
 * @brief Read a byte range from a cuFile handle into device memory.
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
void cufile_read_bytes(CUfileHandle_t handle, std::byte* target, std::ptrdiff_t begin, std::ptrdiff_t end, std::string_view file_name) {
    if(end < begin) {
        throw std::logic_error(fmt::format("Invalid range {} - {} in cufile_read_bytes for {}", begin, end, file_name));
    }
    ssize_t ret = cuFileRead(handle, target, end - begin, begin, 0);
    if (ret < 0) {
        if (ret == -1) {
            throw std::runtime_error(
                    fmt::format("cufile read error ({}) for file {}, range {} - {}: {}", errno, file_name,
                                begin, end, strerror(errno)));
        } else {
            throw std::runtime_error(
                    fmt::format("cufile read error ({}) for file {}, range {} - {}", -ret, file_name, begin,
                                end));
        }
    } else if (ret != end - begin) {
        throw std::runtime_error(fmt::format("cufile read error for file {}: expected {} bytes, got {}", file_name, end-begin, ret));
    }
}

/**
 * @brief Convert a tensor buffer between supported element dtypes.
 *
 * Converts @p size elements from @p source to @p target. Both buffers are assumed
 * to be device-accessible pointers. Only specific dtype pairs are supported.
 *
 * @param target Destination buffer (device pointer), interpreted according to @p t_type.
 * @param source Source buffer (device pointer), interpreted according to @p s_type.
 * @param size Number of elements (not bytes) to convert.
 * @param t_type Target tensor element dtype.
 * @param s_type Source tensor element dtype.
 * @throws std::runtime_error if the conversion pair is not supported.
 */
void convert_tensor_dispatch(std::byte* target, const std::byte* source, std::size_t size, ETensorDType t_type, ETensorDType s_type) {
    if(t_type == ETensorDType::FP32 && s_type == ETensorDType::BF16) {
        convert_dtype(reinterpret_cast<float*>(target), reinterpret_cast<const nv_bfloat16*>(source), size);
    } else if(t_type == ETensorDType::BF16 && s_type == ETensorDType::FP32) {
        convert_dtype(reinterpret_cast<nv_bfloat16*>(target), reinterpret_cast<const float*>(source), size);
    } else if(t_type == ETensorDType::BF16 && s_type == ETensorDType::FP16) {
        convert_dtype(reinterpret_cast<nv_bfloat16*>(target), reinterpret_cast<const half*>(source), size);
    } else if ((t_type == ETensorDType::BYTE || s_type == ETensorDType::BYTE) &&
               get_dtype_size(t_type) == get_dtype_size(s_type)) {
        // BYTE is a raw storage type — identity copy when partner has same byte width.
        // Handles FP4_E2M1 <-> BYTE, FP8_E4M3 <-> BYTE, INT8 <-> BYTE, etc.
        CUDA_CHECK(cudaMemcpyAsync(target, source, size * get_dtype_size(t_type), cudaMemcpyDefault));
    } else {
        throw std::runtime_error(fmt::format("Unsupported conversion: {} -> {}", dtype_to_str(s_type), dtype_to_str(t_type)));
    }
}

/**
 * @brief Read a byte range from file in chunks and convert tensor elements on-device.
 *
 * Reads [begin, end) bytes (file offsets) into a temporary device buffer @p d_buffer
 * in chunks of @p buffer_size, then converts into @p target according to the dtype
 * conversion (@p s_type -> @p t_type).
 *
 * Note: The per-chunk conversion computes the target offset using dtype sizes; callers
 * should ensure the byte range aligns to whole elements of @p s_type.
 *
 * @param handle Registered cuFile handle.
 * @param target Destination tensor buffer (device pointer) in dtype @p t_type.
 * @param begin Start offset in the file (inclusive), in bytes.
 * @param end End offset in the file (exclusive), in bytes.
 * @param file_name File name used for diagnostics only.
 * @param t_type Target tensor element dtype.
 * @param s_type Source tensor element dtype (as stored in the file).
 * @param d_buffer Temporary device buffer used for staged reads (dtype-agnostic bytes).
 * @param buffer_size Size of @p d_buffer in bytes; also the maximum chunk size read per iteration.
 * @throws std::logic_error / std::runtime_error as propagated from reads/conversion.
 */
void cufile_convert_tensor(CUfileHandle_t handle, std::byte* target, std::ptrdiff_t begin, std::ptrdiff_t end,
                           std::string_view file_name, ETensorDType t_type, ETensorDType s_type,
                           std::byte* d_buffer, std::size_t buffer_size) {
    for(std::ptrdiff_t p = 0; p < end - begin; p += buffer_size) {
        std::ptrdiff_t amount = std::min(end - begin - p, (std::ptrdiff_t)buffer_size);
        cufile_read_bytes(handle, d_buffer, begin + p, begin + p + amount, file_name);
        convert_tensor_dispatch(target + p * get_dtype_size(t_type) / get_dtype_size(s_type),
                                d_buffer,
                                amount / get_dtype_size(s_type),
                                t_type, s_type);
        CUDA_CHECK(cudaDeviceSynchronize());
    }
}


cuFileRef::cuFileRef(std::string file_name) : cuFileRef(open_cufile(std::move(file_name)))
{

}

cuFileRef::~cuFileRef() noexcept
{
    close(mFileDescriptor);
    cuFileHandleDeregister(mHandle);
}

/**
 * @brief Read a byte range from the underlying file into device memory.
 *
 * The interval is treated as [begin, end) in file offsets (bytes).
 *
 * @param target Destination buffer (device pointer) receiving (end - begin) bytes.
 * @param begin Start offset in the file (inclusive), in bytes.
 * @param end End offset in the file (exclusive), in bytes. Must be >= begin.
 * @throws std::logic_error / std::runtime_error as propagated from cufile_read_bytes().
 */
void cuFileRef::read_bytes(std::byte* target, std::ptrdiff_t begin, std::ptrdiff_t end)
{
    cufile_read_bytes(mHandle, target, begin, end, mFileName);
}

/**
 * @brief Read a byte range from the underlying file and convert tensor elements on-device.
 *
 * Reads [begin, end) bytes into a temporary device buffer and converts into @p target
 * from dtype @p s_type to dtype @p t_type.
 *
 * @param target Destination tensor buffer (device pointer) in dtype @p t_type.
 * @param begin Start offset in the file (inclusive), in bytes.
 * @param end End offset in the file (exclusive), in bytes.
 * @param file_name File name used for diagnostics only.
 * @param t_type Target tensor element dtype.
 * @param s_type Source tensor element dtype (as stored in the file).
 * @param d_buffer Temporary device buffer used for staged reads.
 * @param buffer_size Size of @p d_buffer in bytes; also the maximum chunk size read per iteration.
 * @throws std::logic_error / std::runtime_error as propagated from cufile_convert_tensor().
 */
void cuFileRef::read_and_convert(std::byte* target, std::ptrdiff_t begin, std::ptrdiff_t end,
        std::string_view file_name, ETensorDType t_type, ETensorDType s_type,
        std::byte* d_buffer, std::size_t buffer_size)
{
    cufile_convert_tensor(mHandle, target, begin, end, file_name, t_type, s_type, d_buffer, buffer_size);
}
