// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// Copyright (c) 2025, IST Austria, developed by Erik Schultheis
// SPDX-License-Identifier: Apache-2.0
//

#ifndef SUROGATE_SRC_UTILS_SAFETENSORS_H
#define SUROGATE_SRC_UTILS_SAFETENSORS_H

#include <string>
#include <unordered_map>
#include <vector>

#include "tensor_container.h"

class Tensor;
class TensorAllocator;
class cuFileRef;
class NCCLCommunicator;
enum class ETensorDType : int;

class SafeTensorsReader;

class SafeTensorEntry {
public:
    //! Get the name of the tensor
    const std::string& name() const { return mName; }
    //! Get the shape of the tensor.
    const std::vector<long>& shape() const { return mShape; }
    //! Get the dtype of the tensor. Note that safetensors has very limited
    //! dtype support, so this information may not be accurate.
    ETensorDType dtype() const { return mDType; }

    //! Read part of the tensor data into another tensor
    //! ignoring all shape information.
    //! \param target The tensor to read data into
    //! \param offset Offset in elements from the beginning of this tensor's data
    //! \param elements Number of elements to read
    //! \param allow_cast Whether to allow dtype conversion during read
    void read_raw(Tensor& target, std::ptrdiff_t offset,
                  std::ptrdiff_t elements, bool allow_cast = false) const;

    //! Reads then entire Tensor into target.
    //! \param target The tensor to read into (must match shape and optionally dtype)
    //! \param allow_cast Whether to allow dtype conversion
    void read_tensor(Tensor& target, bool allow_cast = false) const;

private:
    friend class SafeTensorsReader;

    SafeTensorEntry(const std::string& name, const std::vector<long>& shape, ETensorDType dtype,
                    std::string file_name, std::shared_ptr<cuFileRef> handle, SafeTensorsReader* reader,
                    std::ptrdiff_t data_begin, std::ptrdiff_t data_end);

    std::string mName;
    std::vector<long> mShape;
    ETensorDType mDType;
    std::string mFileName;
    std::shared_ptr<cuFileRef> mHandle;
    SafeTensorsReader* mReader;

    // in bytes
    std::ptrdiff_t mDataBegin;
    std::ptrdiff_t mDataEnd;
};


class SafeTensorsReader {
public:
    explicit SafeTensorsReader(const std::string& file_name);
    ~SafeTensorsReader();

    //! Get all tensor entries
    const std::vector<SafeTensorEntry>& entries() const { return mEntries; }

    //! Load specific tensors into a container
    void load_tensors(ITensorContainer& container, bool allow_cast = false) const;

    //! Linear(!) search for a tensor by name
    const SafeTensorEntry& find_entry(std::string_view name) const;
private:
    friend class SafeTensorEntry;
    void parse_single_file(const std::string& file_path);
    void parse_index_file(const std::string& index_file);

    std::vector<SafeTensorEntry> mEntries;

    long mConversionBufferSize = 0;
    std::byte* mConversionBuffer = nullptr;
};

/*!
 * \brief Safetensors need to be written in two phases. First, for every Tensor intended
 * to be saved, `register_tensor` must be called. At this point, only the metadata of
 * the tensor, its shape, dtype, and name, are recorded.
 * Then, a call to prepare_metadata must be made. This will open the actual file for writing
 * and write the metadata, as well as reserve space for the full tensors.
 * Next, for every tensor, `write_tensor` must be called exactly once. This will write the tensor data
 * into the file.
 * Finally, `finalize` must be called. This will close the file and finalize the metadata.
 */
class SafeTensorWriter {
public:
    SafeTensorWriter(std::string file_name);
    ~SafeTensorWriter();
    void register_tensor(const std::string& name, const TensorShard& tensor);
    void prepare_metadata(NCCLCommunicator* comm);
    void write_tensor(const std::string& name, const TensorShard& tensor, NCCLCommunicator* comm);
    void write_raw(const std::string& name, std::ptrdiff_t offset,
                  std::ptrdiff_t elements, const TensorShard& tensor);
    void mark_done(const std::string& name);
    void finalize(NCCLCommunicator* comm);
private:
    struct sTensorInfo {
        ETensorDType DType;
        std::vector<long> Shape;
        long Begin = -1;
        long Size = -1;
        bool Done = false;
    };
    std::unordered_map<std::string, sTensorInfo> mRegisteredTensors;
    bool mMetaFinalized = false;
    long mHeaderSize = -1;
    long mTotalSize = -1;

    std::string mFileName;
    int mFileDescriptor = -1;
    std::byte* mMappedFile = nullptr;
};

void load_safetensors(const std::string& file_name, ITensorContainer& tensors, bool allow_cast);
void write_safetensors(const std::string& file_name, ITensorContainer& tensors);
void write_safetensors(const std::string& file_name, ITensorContainer& tensors, NCCLCommunicator& comm);

//! Gets the path to the HF cache
std::string get_hf_hub();
//! Gets the base path to a hf model of the given name (may or may not exist)
std::string get_hf_model_path(std::string model_name);
//! Gets the path to the files of a specific HF model.
std::string get_hf_model_files(std::string model_name, std::string revision = {});

#endif //SUROGATE_SRC_UTILS_SAFETENSORS_H
