// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// Copyright (c) 2025, IST Austria, developed by Erik Schultheis
// SPDX-License-Identifier: Apache-2.0
//

#include "safetensors.h"

#include <bit>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <unordered_set>

#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

#include <fmt/core.h>
#include <nlohmann/json.hpp>

#include "allocator.h"
#include "comm.h"
#include "cu_file.h"
#include "tensor.h"

/**
 * @brief Parsed SafeTensors header data.
 *
 * The SafeTensors file starts with an 8-byte little-endian unsigned integer
 * indicating the JSON header size in bytes, followed by the JSON header.
 */
struct sSafeTensorsHeader {
    /** @brief Size of the JSON header (bytes), not including this 8-byte length field. */
    std::uint64_t HeaderSize;
    /** @brief Parsed JSON metadata for all tensor entries and optional "__metadata__". */
    nlohmann::json MetaData;
};

/**
 * @brief Read and parse the SafeTensors JSON header from a file.
 *
 * @param file_name Path to the `.safetensors` file.
 * @return A struct containing the header size (bytes) and parsed JSON metadata.
 *
 * @throws std::runtime_error If the file cannot be read or the header is invalid.
 */
sSafeTensorsHeader read_safetensors_header(const std::string& file_name) {
    std::uint64_t header_size = -1;
    std::ifstream file(file_name, std::ios_base::binary);
    file.read(reinterpret_cast<char*>(&header_size), sizeof(header_size));
    if (!file) {
        // read error
        throw std::runtime_error("Error opening safetensors file '" + file_name + "'");
    }

    std::vector<char> header(header_size, '\0');
    file.read(header.data(), (long)header_size);
    auto parsed = nlohmann::json::parse(header.begin(), header.end());
    return {header_size, std::move(parsed)};
}

// SafeTensorEntry implementation

/**
 * @brief Construct a tensor entry view into a SafeTensors file.
 *
 * @param name Tensor name (as stored in SafeTensors JSON).
 * @param shape Tensor shape (global) as a list of dimension sizes.
 * @param dtype Element data type stored in the file.
 * @param file_name File path backing this entry (kept for conversion path / diagnostics).
 * @param handle Shared cuFileRef used for reading from the backing file.
 * @param reader Owning reader instance (used for shared conversion buffer).
 * @param data_begin Absolute byte offset in file where tensor data begins.
 * @param data_end Absolute byte offset in file where tensor data ends (exclusive).
 */
SafeTensorEntry::SafeTensorEntry(const std::string& name, const std::vector<long>& shape, ETensorDType dtype,
                                 std::string file_name, std::shared_ptr<cuFileRef> handle, SafeTensorsReader* reader,
                                 std::ptrdiff_t data_begin, std::ptrdiff_t data_end)
    : mName(name), mShape(shape), mDType(dtype), mFileName(std::move(file_name)), mHandle(handle), mReader(reader),
      mDataBegin(data_begin), mDataEnd(data_end) {
}

/**
 * @brief Read a contiguous range of elements from this entry into a target tensor.
 *
 * Bounds are validated against the entry size, and the target tensor byte size
 * must match the requested element count at the target dtype.
 *
 * @param target Destination tensor (device memory) to fill.
 * @param offset Element offset (not bytes) from the start of this entry.
 * @param elements Number of elements to read.
 * @param allow_cast If true, allow dtype conversion from file dtype to target dtype.
 *
 * @throws std::runtime_error On invalid range, size mismatch, or dtype mismatch (if allow_cast is false).
 */
void SafeTensorEntry::read_raw(Tensor& target, std::ptrdiff_t offset,
                               std::ptrdiff_t elements, bool allow_cast) const {
    long nelem = (mDataEnd - mDataBegin) / get_dtype_size(mDType);
    if (offset < 0 || offset + elements > nelem)
        throw std::runtime_error(fmt::format("Invalid read range: offset={}, elements={}, size={}",
                                             offset, elements, nelem));

    // Check if target has enough space (in bytes)
    if (target.bytes() != elements * get_dtype_size(target.DType))
        throw std::runtime_error(fmt::format("Target tensor size mismatch for `{}`: has {} bytes, needs {} elements of {} bytes",
                                                 mName, target.bytes(), elements, get_dtype_size(target.DType)));

    std::ptrdiff_t start = mDataBegin + offset * get_dtype_size(mDType);
    std::ptrdiff_t end = start + elements * get_dtype_size(mDType);

    // Validate dtype
    if (mDType != target.DType && !allow_cast)
        throw std::runtime_error(fmt::format("DType mismatch: tensor has {}, file has {}",
                                             dtype_to_str(target.DType), dtype_to_str(mDType)));

    if (mDType == target.DType) {
        mHandle->read_bytes(target.Data, start, end);
    } else {
        // Need conversion buffer
        if (mReader->mConversionBufferSize < end - start) {
            CUDA_CHECK(cudaFree(mReader->mConversionBuffer));
            mReader->mConversionBufferSize = std::min(end - start, 256 * 1024 * 1024L);
            CUDA_CHECK(cudaMalloc((void**)&mReader->mConversionBuffer, mReader->mConversionBufferSize));
        }
        mHandle->read_and_convert(target.Data, start, end, mFileName,
                                  target.DType, mDType,
                                  mReader->mConversionBuffer,
                                  mReader->mConversionBufferSize);
    }
}

/**
 * @brief Read the full tensor for this entry into @p target, validating rank and shape.
 *
 * @param target Destination tensor whose rank and sizes must match this entry.
 * @param allow_cast If true, allow dtype conversion from file dtype to target dtype.
 *
 * @throws std::runtime_error On rank/shape mismatch or dtype mismatch (if allow_cast is false).
 */
void SafeTensorEntry::read_tensor(Tensor& target, bool allow_cast) const {
    if (target.Rank != static_cast<int>(mShape.size()))
        throw std::runtime_error(fmt::format("Rank mismatch for tensor `{}`: expected {}, got {}",
                                             mName, mShape.size(), target.Rank));

    for (int i = 0; i < target.Rank; ++i)
        if (mShape[i] != target.Sizes[i])
            throw std::runtime_error(fmt::format("Shape mismatch for tensor `{}` at dim {}: expected {}, got {}",
                                                 mName, i, mShape[i], target.Sizes[i]));
    read_raw(target, 0, target.nelem(), allow_cast);
}

// SafeTensorsReader implementation

/**
 * @brief Construct a reader for a SafeTensors file or an HF-style `.index.json`.
 *
 * If @p file_name ends with `.index.json`, the index is parsed and all referenced
 * shard files are added; otherwise, the single SafeTensors file is parsed.
 *
 * @param file_name Path to a `.safetensors` file or a `.safetensors.index.json`.
 *
 * @throws std::runtime_error / nlohmann::json exceptions on I/O or parse failures.
 */
SafeTensorsReader::SafeTensorsReader(const std::string& file_name) {
    namespace fs = std::filesystem;

    if (file_name.ends_with(".index.json")) {
        parse_index_file(file_name);
    } else if (fs::is_directory(file_name)) {
        // Handle directory paths by looking for index or single safetensors file
        fs::path dir(file_name);
        fs::path index_file = dir / "model.safetensors.index.json";
        fs::path single_file = dir / "model.safetensors";

        if (fs::exists(index_file)) {
            parse_index_file(index_file.string());
        } else if (fs::exists(single_file)) {
            parse_single_file(single_file.string());
        } else {
            throw std::runtime_error("No safetensors files found in directory '" + file_name + "'");
        }
    } else {
        parse_single_file(file_name);
    }
}

/**
 * @brief Parse a single `.safetensors` file and append all tensor entries to this reader.
 *
 * This reads the JSON header, computes the absolute data offsets (including the
 * header length field and JSON header), and stores entries referencing the file.
 *
 * @param file_path Path to a `.safetensors` file.
 *
 * @throws std::runtime_error / nlohmann::json exceptions on I/O or parse failures.
 */
void SafeTensorsReader::parse_single_file(const std::string& file_path) {
    auto [HeaderSize, MetaData] = read_safetensors_header(file_path);
    ptrdiff_t offset = HeaderSize + sizeof(HeaderSize);
    std::shared_ptr<cuFileRef> cu_file = std::make_shared<cuFileRef>(file_path);
    for (const auto& el : MetaData.items()) {
        const std::string& name = el.key();
        if (name == "__metadata__") {
            // TODO extract metadata?
            continue;
        }

        ETensorDType dtype = dtype_from_str(el.value()["dtype"].get<std::string_view>());
        auto shape = el.value()["shape"].get<std::vector<long>>();
        auto begin = el.value()["data_offsets"][0].get<std::ptrdiff_t>();
        auto end = el.value()["data_offsets"][1].get<std::ptrdiff_t>();

        mEntries.emplace_back(SafeTensorEntry{name, shape, dtype, file_path, cu_file, this,
                                              begin + offset, end + offset});
    }
}

/**
 * @brief Parse a HuggingFace SafeTensors index file and load all referenced shard files.
 *
 * The index file maps tensor names to shard filenames in `weight_map`. Each shard
 * is processed once and parsed via parse_single_file().
 *
 * @param index_file Path to a `.index.json` file.
 *
 * @throws std::runtime_error / nlohmann::json exceptions on I/O or parse failures.
 */
void SafeTensorsReader::parse_index_file(const std::string& index_file) {
    std::ifstream file(index_file);
    auto parsed = nlohmann::json::parse(file);
    auto weight_map = parsed["weight_map"];

    std::unordered_set<std::string> processed_files;
    std::filesystem::path index_path(index_file);

    for (const auto& el : weight_map.items()) {
        auto f_name = el.value().get<std::string>();
        if (processed_files.contains(f_name))
            continue;
        processed_files.insert(f_name);

        std::filesystem::path full_path = index_path.parent_path() / f_name;
        parse_single_file(full_path.native());
    }
}

/**
 * @brief Load all tensors present in both the reader and the container.
 *
 * The container is enumerated once and matched by name against the entries parsed
 * from the file(s). Only matching names are read.
 *
 * @param container Tensor container providing named tensors to populate.
 * @param allow_cast If true, allow dtype conversion into the container's tensors.
 *
 * @throws std::runtime_error On shape/rank/dtype mismatches (depending on allow_cast).
 */
void SafeTensorsReader::load_tensors(ITensorContainer& container, bool allow_cast) const {
    std::unordered_map<std::string, Tensor> named_tensors;
    container.iterate_tensors([&named_tensors](std::string name, const Tensor& tensor) {
        named_tensors.emplace(std::move(name), tensor);
    });

    for (const auto& entry : mEntries)
        if (auto found = named_tensors.find(entry.name()); found != named_tensors.end())
            entry.read_tensor(found->second, allow_cast);
}

/**
 * @brief Destroy the reader and release any shared conversion buffer.
 *
 * Frees CUDA device memory used for temporary conversion, if allocated.
 */
SafeTensorsReader::~SafeTensorsReader() {
    if (mConversionBuffer) {
        auto err = cudaFree(mConversionBuffer);
        if (err != cudaSuccess) {
            // Never throw from a destructor — it causes std::terminate during stack unwinding.
            fprintf(stderr, "[SafeTensorsReader] WARNING: cudaFree(mConversionBuffer) failed: %s\n",
                    cudaGetErrorString(err));
            cudaGetLastError();  // Clear the error
        }
    }
}

/**
 * @brief Find an entry by tensor name.
 *
 * @param name Tensor name to look up.
 * @return Reference to the matching entry.
 *
 * @throws std::out_of_range If no entry with this name exists.
 */
const SafeTensorEntry& SafeTensorsReader::find_entry(std::string_view name) const {
    for (auto& entry : mEntries)
        if (entry.name() == name)
            return entry;
    throw std::out_of_range(fmt::format("Entry not found: {}", name));
}

/**
 * @brief Convenience function to load SafeTensors into an ITensorContainer.
 *
 * @param file_name Path to `.safetensors` or `.index.json`.
 * @param tensors Container to populate.
 * @param allow_cast If true, allow dtype conversion into container tensors.
 */
void load_safetensors(const std::string& file_name, ITensorContainer& tensors, bool allow_cast) {
    try {
        SafeTensorsReader reader(file_name);
        reader.load_tensors(tensors, allow_cast);
    } catch (std::exception& e) {
        throw std::runtime_error(fmt::format("Error loading safetensors file '{}': {}", file_name, e.what()));
    }
}

/**
 * @brief Construct a SafeTensors writer targeting @p file_name.
 *
 * The writer uses a temporary file (`<file>.tmp`) and renames it to the final name on finalize().
 *
 * @param file_name Output file path for the final `.safetensors`.
 */
SafeTensorWriter::SafeTensorWriter(std::string file_name) : mFileName(file_name) {
}

/**
 * @brief Destroy the writer and clean up any temporary resources.
 *
 * If a temporary file is still open/mapped, it is unmapped/closed and removed.
 * (Note: throwing from a destructor is generally unsafe; this matches current behavior.)
 */
SafeTensorWriter::~SafeTensorWriter() {
    if (mFileDescriptor > 0) {
        std::string temp_name = mFileName + ".tmp";
        if (mMappedFile) {
            if (munmap(mMappedFile, mTotalSize) != 0)
                throw std::system_error(errno, std::system_category(), "Error unmapping file " + temp_name);
        }
        close(mFileDescriptor);
        unlink(temp_name.c_str());
    }
}

/**
 * @brief Register a tensor for later writing (metadata + offsets are derived from registrations).
 *
 * Must be called before prepare_metadata().
 *
 * @param name Tensor name to store in the SafeTensors header.
 * @param tensor Tensor shard describing dtype and global shape/size.
 *
 * @throws std::logic_error If metadata has already been finalized.
 */
void SafeTensorWriter::register_tensor(const std::string& name, const TensorShard& tensor) {
    if (mMetaFinalized)
        throw std::logic_error("Cannot register tensor after metadata has been finalized");
    mRegisteredTensors.insert({name, {tensor.DType, std::vector<long>(tensor.GlobalShape.begin(), tensor.GlobalShape.begin() + tensor.Rank), 0, (long)tensor.global_nelem() * get_dtype_size(tensor.DType)}});
}

/**
 * @brief Finalize and broadcast metadata, allocate/map output file, and write the header.
 *
 * Builds the JSON header with dtype/shape/data_offsets for every registered tensor.
 * If @p comm is provided, rank 0 creates and mmaps the temporary file and shares the
 * mapped pointer with peers via host_gather(); then all ranks synchronize.
 *
 * @param comm Optional NCCL communicator used for multi-process coordination; may be nullptr.
 *
 * @throws std::system_error On file open/truncate/mmap failures.
 */
void SafeTensorWriter::prepare_metadata(NCCLCommunicator* comm) {
    nlohmann::json meta_data;
    meta_data["__metadata__"] = nlohmann::json::object({{"format", "pt"},
                                                        {"writer", "surogate"}});

    long offset = 0;
    for (auto& [name, tensor] : mRegisteredTensors) {
        meta_data[name]["dtype"] = dtype_to_str(tensor.DType);
        meta_data[name]["shape"] = tensor.Shape;
        tensor.Begin = offset;
        meta_data[name]["data_offsets"] = std::vector<long>{offset, offset + tensor.Size};
        offset += tensor.Size;
    }

    std::string header = meta_data.dump();
    std::uint64_t header_size = header.size();
    mHeaderSize = header_size + sizeof(header_size);

    std::vector<SafeTensorWriter*> peers;
    if (comm) {
        if (comm->num_nodes() <= 1) {
            // Single-node: host_gather shares the mmap pointer with all local threads.
            // All pointers are in the same process — safe to dereference.
            peers = comm->host_gather(this);
        } else {
            // Multi-node: host_gather gathers raw pointers across processes via NCCL.
            // Cross-process pointers are invalid — only keep local peers.
            auto all_peers = comm->host_gather(this);
            if (comm->rank() == 0) {
                int local_gpus = comm->num_local_gpus();
                peers.assign(all_peers.begin(), all_peers.begin() + local_gpus);
            }
        }
    }

    if (!comm || comm->rank() == 0) {
        std::string temp_name = mFileName + ".tmp";

        mFileDescriptor = open(temp_name.c_str(), O_RDWR | O_CREAT, S_IRUSR | S_IWUSR | S_IRGRP);
        if (mFileDescriptor == -1)
            throw std::system_error(errno, std::system_category(), "Error opening file '" + temp_name + "' for writing");
        mTotalSize = sizeof(header_size) + header_size + offset;
        if (ftruncate(mFileDescriptor, mTotalSize) < 0)
            throw std::system_error(errno, std::system_category(), "Error truncating file " + temp_name);

        std::byte* host_ptr = (std::byte*)mmap(nullptr, mTotalSize, PROT_WRITE,
                                               MAP_SHARED, mFileDescriptor, 0);
        if (host_ptr == MAP_FAILED)
            throw std::system_error(errno, std::system_category(), "Error memory-mapping file " + temp_name);

        // write the header
        std::memcpy(host_ptr, &header_size, sizeof(header_size));
        std::memcpy(host_ptr + sizeof(header_size), header.data(), header_size);

        for (auto& peer : peers) {
            peer->mMappedFile = host_ptr;
            peer->mMetaFinalized = true;
        }

        mMappedFile = host_ptr;
        mMetaFinalized = true;
    }

    // Multi-node: non-root-node ranks don't get the mmap pointer, but must
    // be marked as finalized so write_tensor() can mark tensors as Done
    // (skipping actual writes) without throwing.
    if (comm && !mMetaFinalized) {
        mMetaFinalized = true;
    }

    if (comm)
        comm->barrier();
}

/**
 * @brief Write an entire tensor shard into the output file.
 *
 * Requires prepare_metadata() to have been called. For sharded tensors, each rank writes
 * its shard (based on tensor.ShardIndex). For replicated tensors, only the root rank writes
 * (indicated by all calls having ShardIndex == 0).
 *
 * @param name Registered tensor name to write.
 * @param tensor Tensor shard containing device pointer and shard metadata.
 * @param comm Optional communicator; required when tensor.NumShards > 1.
 *
 * @throws std::logic_error If metadata is not finalized, tensor already written, or sharded write lacks comm.
 * @throws std::out_of_range If @p name was not registered.
 */
void SafeTensorWriter::write_tensor(const std::string& name, const TensorShard& tensor, NCCLCommunicator* comm) {
    if (!mMetaFinalized)
        throw std::logic_error("Cannot write tensor before metadata has been finalized");

    auto found = mRegisteredTensors.find(name);
    if (found == mRegisteredTensors.end())
        throw std::out_of_range("Invalid tensor " + name);

    if (found->second.Done)
        throw std::logic_error("Tensor " + name + " has already been written");

    if (!comm && tensor.NumShards > 1)
        throw std::logic_error("Cannot write tensor " + name + " with multiple shards without a communicator");

    if (comm && tensor.ShardIndex != comm->rank()) {
        // When a tensor is replicated instead of sharded, all calls with have ShardIdx == 0,
        // so only the root rank writes the tensor.
        found->second.Done = true;
        return;
    }

    // Multi-node: non-root-node ranks don't have a mapped file.
    // Mark as done without writing (rank 0's node writes the file).
    if (!mMappedFile) {
        found->second.Done = true;
        return;
    }

    long shard_begin = found->second.Begin + mHeaderSize + tensor.ShardIndex * tensor.bytes();
    std::vector<std::byte> data(tensor.bytes());
    CUDA_CHECK(cudaMemcpy(data.data(), tensor.Data, tensor.bytes(), cudaMemcpyDeviceToHost));
    std::memcpy(mMappedFile + shard_begin, data.data(), tensor.bytes());
    found->second.Done = true;
}

/**
 * @brief Write a contiguous element range for a registered tensor.
 *
 * The data is copied from device to host and written into the mmapped file at the correct byte offset.
 * This is useful for partial writes (e.g., streaming) while keeping SafeTensors offsets fixed.
 *
 * @param name Registered tensor name to write into.
 * @param offset Element offset (not bytes) from the beginning of the registered tensor.
 * @param elements Number of elements to write.
 * @param tensor Source shard providing device memory and dtype.
 *
 * @throws std::logic_error If metadata is not finalized, range is invalid, or dtype mismatches registration.
 * @throws std::out_of_range If @p name was not registered.
 */
void SafeTensorWriter::write_raw(const std::string& name, std::ptrdiff_t offset,
                                 std::ptrdiff_t elements, const TensorShard& tensor) {
    if (!mMetaFinalized)
        throw std::logic_error("Cannot write tensor before metadata has been finalized");

    auto found = mRegisteredTensors.find(name);
    if (found == mRegisteredTensors.end())
        throw std::out_of_range("Invalid tensor " + name);

    ETensorDType dtype = tensor.DType;
    long nelem = found->second.Size / get_dtype_size(dtype);
    if (offset < 0 || offset + elements > nelem)
        throw std::logic_error(fmt::format("Invalid write range for tensor `{}`: offset={}, elements={}, size={}",
                                           name, offset, elements, nelem));

    if (found->second.DType != dtype)
        throw std::logic_error(fmt::format("DType mismatch for tensor `{}`: registered as {}, writing as {}",
                                           name, dtype_to_str(found->second.DType), dtype_to_str(dtype)));

    std::ptrdiff_t write_start = found->second.Begin + mHeaderSize + offset * get_dtype_size(dtype);
    std::ptrdiff_t write_size = elements * get_dtype_size(dtype);

    std::vector<std::byte> host_data(write_size);
    CUDA_CHECK(cudaMemcpy(host_data.data(), tensor.Data, write_size, cudaMemcpyDeviceToHost));
    std::memcpy(mMappedFile + write_start, host_data.data(), write_size);
}

/**
 * @brief Mark a registered tensor as written without writing any data.
 *
 * Intended for cases where another mechanism wrote the tensor bytes, or when a rank
 * is not responsible for writing (coordination handled externally).
 *
 * @param name Registered tensor name.
 *
 * @throws std::out_of_range If @p name was not registered.
 */
void SafeTensorWriter::mark_done(const std::string& name) {
    auto found = mRegisteredTensors.find(name);
    if (found == mRegisteredTensors.end())
        throw std::out_of_range("Invalid tensor " + name);
    found->second.Done = true;
}

/**
 * @brief Finalize the file: verify all tensors written, unmap/close, and rename temp file.
 *
 * If @p comm is provided, barriers are used to ensure all ranks have finished writing
 * and validation is performed consistently. Rank 0 performs the filesystem rename.
 *
 * @param comm Optional NCCL communicator used for synchronization; may be nullptr.
 *
 * @throws std::logic_error If any registered tensor has not been written/marked done.
 * @throws std::system_error On unmap failures.
 */
void SafeTensorWriter::finalize(NCCLCommunicator* comm) {
    if (comm)
        comm->barrier();

    for (auto& [name, tensor] : mRegisteredTensors)
        if (!tensor.Done)
            throw std::logic_error("Tensor " + name + " has not been written");

    if (comm)
        comm->barrier();

    if (!comm || comm->rank() == 0) {
        if (mMappedFile) {
            if (munmap(mMappedFile, mTotalSize) != 0)
                throw std::system_error(errno, std::system_category(), "Error unmapping file " + mFileName);
            mMappedFile = nullptr;
        }
        if (mFileDescriptor >= 0) {
            close(mFileDescriptor);
            mFileDescriptor = -1;
            std::string temp_name = mFileName + ".tmp";
            std::filesystem::rename(temp_name, mFileName);
        }
    }
}

/**
 * @brief Convenience function to write all tensors from a container into a SafeTensors file.
 *
 * Registers all tensors, writes metadata, writes each tensor in full, then finalizes.
 *
 * @param file_name Output `.safetensors` path.
 * @param tensors Tensor container providing named tensors to serialize.
 */
void write_safetensors(const std::string& file_name, ITensorContainer& tensors) {
    SafeTensorWriter writer(file_name);
    tensors.iterate_tensors([&writer](std::string name, const Tensor& tensor) {
        writer.register_tensor(name, tensor);
    });
    writer.prepare_metadata(nullptr);
    tensors.iterate_tensors([&writer](std::string name, const Tensor& tensor) {
        writer.write_tensor(name, tensor, nullptr);
    });
    writer.finalize(nullptr);
}

/**
 * @brief Convenience function to write all tensors from a container into a SafeTensors file (multi-GPU).
 *
 * Registers all tensors, writes metadata, writes each tensor in full, then finalizes.
 * Uses the provided communicator for synchronization - only rank 0 performs the actual write.
 *
 * @param file_name Output `.safetensors` path.
 * @param tensors Tensor container providing named tensors to serialize.
 * @param comm NCCL communicator for multi-GPU synchronization.
 */
void write_safetensors(const std::string& file_name, ITensorContainer& tensors, NCCLCommunicator& comm) {
    SafeTensorWriter writer(file_name);
    tensors.iterate_tensors([&writer](std::string name, const Tensor& tensor) {
        writer.register_tensor(name, tensor);
    });
    writer.prepare_metadata(&comm);
    tensors.iterate_tensors([&writer, &comm](std::string name, const Tensor& tensor) {
        writer.write_tensor(name, tensor, &comm);
    });
    writer.finalize(&comm);
}

/**
 * @brief Determine the HuggingFace hub cache directory.
 *
 * Resolution order:
 * - If HF_HOME is set: `${HF_HOME}/hub`
 * - Else if XDG_CACHE_HOME is set: `${XDG_CACHE_HOME}/huggingface/hub`
 * - Else: `${HOME}/.cache/huggingface/hub`
 *
 * @return Filesystem path to the HF hub cache directory.
 */
std::string get_hf_hub() {
    const char* hf_home_env = std::getenv("HF_HOME");
    if (hf_home_env == nullptr) {
        const char* xdg_cache_home = std::getenv("XDG_CACHE_HOME");
        if (xdg_cache_home == nullptr)
            return std::string(std::getenv("HOME")) + "/.cache/huggingface/hub";
        else
            return std::string(xdg_cache_home) + "/huggingface/hub";
    }
    return std::string(hf_home_env) + "/hub";
}

/**
 * @brief Convert a HuggingFace model id (`org/name`) to its local hub cache directory.
 *
 * This matches the HF hub convention `models--org--name` under the hub cache root.
 *
 * @param model_name Model id of the form `org/name`.
 * @return Full path to the model cache directory (not a snapshot).
 *
 * @throws std::runtime_error If @p model_name is not of the form `org/name`.
 */
std::string get_hf_model_path(std::string model_name) {
    auto slash = model_name.find_last_of('/');
    if(slash == std::string::npos) {
        throw std::runtime_error("HF model name must be of the form org/name");
    }
    model_name = "/models--" + model_name.replace(slash, 1, "--");
    return get_hf_hub() + model_name;
}

/**
 * @brief Resolve the local snapshot directory for a cached HuggingFace model.
 *
 * If @p revision is empty, attempts to pick the only available snapshot under
 * `<model>/snapshots/`. If multiple snapshots exist, throws.
 *
 * @param model_name Model id of the form `org/name`.
 * @param revision Snapshot revision directory name; if empty, auto-detect if unambiguous.
 * @return Full path to the resolved snapshot directory, or empty string if not found locally.
 *
 * @throws std::runtime_error If multiple snapshots exist and @p revision is empty.
 */
std::string get_hf_model_files(std::string model_name, std::string revision) {
    // First check if model_name is already a valid directory with config.json
    if (std::filesystem::exists(model_name) && std::filesystem::is_directory(model_name)) {
        std::string config_path = model_name + "/config.json";
        if (std::filesystem::exists(config_path)) {
            return model_name;
        }
    }

    // Otherwise, try to resolve as a HuggingFace model name
    auto base_path = get_hf_model_path(model_name);
    if (!std::filesystem::exists(base_path))
        return "";

    if (revision.empty()) {
        std::string snapshot_path = base_path + "/snapshots/";
        for (auto& p : std::filesystem::directory_iterator(snapshot_path)) {
            if (!revision.empty())
                throw std::runtime_error("Found multiple snapshots, please specify a revision");
            if (p.is_directory())
                revision = p.path().filename();
        }
    }

    std::string revision_path = base_path + "/snapshots/" + revision;
    if (!std::filesystem::exists(revision_path))
        return "";
    return revision_path;
}
