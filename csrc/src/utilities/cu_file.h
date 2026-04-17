// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// Copyright (c) 2025, IST Austria, developed by Erik Schultheis
// SPDX-License-Identifier: Apache-2.0
//

#ifndef SUROGATE_SRC_MODELS_CUFILE_H
#define SUROGATE_SRC_MODELS_CUFILE_H

#include <string>

#include "utilities/dtype.h"

// forward declarations
typedef void* CUfileHandle_t;

class cuFileRef {
public:
    explicit cuFileRef(std::string file_name);
    cuFileRef(CUfileHandle_t h, int fd, std::string name)
        : mHandle(h),
          mFileDescriptor(fd),
          mFileName(std::move(name)) {
    }
    ~cuFileRef() noexcept;
    CUfileHandle_t& handle() {
        return mHandle;
    }

    //! \brief Read raw bytes from the range `[begin, end)`
    //! \param target Pointer to the target buffer
    //! \param begin Offset into the file for the beginning of the read range (inclusive)
    //! \param end Offset into the file for the end of the read range (exclusive)
    //! \throws std::runtime_error If the range cannot be read
    //! \throws std::logic_error If `[begin, end)` does not form a valid range
    void read_bytes(std::byte* target, std::ptrdiff_t begin, std::ptrdiff_t end);

    void read_and_convert(std::byte* target,
                          std::ptrdiff_t begin,
                          std::ptrdiff_t end,
                          std::string_view file_name,
                          ETensorDType t_type,
                          ETensorDType s_type,
                          std::byte* d_buffer,
                          std::size_t buffer_size);

private:
    CUfileHandle_t mHandle;
    int mFileDescriptor;
    std::string mFileName;
};

#endif  //SUROGATE_SRC_MODELS_CUFILE_H
