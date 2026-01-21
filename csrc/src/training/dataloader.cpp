// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// Copyright (c) 2025, IST Austria, developed by Erik Schultheis
// SPDX-License-Identifier: Apache-2.0
//

#include "dataloader.h"

#include <glob.h>

#include <algorithm>
#include <cassert>
#include <cstring>
#include <filesystem>
#include <random>
#include <ranges>

#include <fmt/core.h>

#include "utilities/tensor.h"
#include "utilities/philox.h"

/**
 * @brief Construct a DataLoader from a glob pattern.
 *
 * Matches files using @p file_pattern, then delegates to the file-list constructor.
 *
 * @param file_pattern Glob pattern for token files (e.g. "/path/*.bin").
 * @param seq_len Sequence length (number of tokens) per training sample.
 * @param rank Local process rank in [0, world_size).
 * @param world_size Total number of participating ranks.
 * @param seed Global seed used for deterministic shuffling across epochs.
 */
DataLoader::DataLoader(const std::string& file_pattern, int seq_len, int rank, int world_size, unsigned long seed) :
       DataLoader(match_files(file_pattern), seq_len, rank, world_size, seed) {

}

/**
 * @brief Construct a DataLoader from an explicit list of token files.
 *
 * Parses headers of all files, validates vocabulary consistency, computes total token/chunk counts,
 * and advances into epoch 0 (via an initial call to advance_epoch()).
 *
 * @param file_list List of token file paths.
 * @param seq_len Sequence length (number of tokens) per training sample.
 * @param rank Local process rank in [0, world_size).
 * @param world_size Total number of participating ranks.
 * @param seed Global seed used for deterministic shuffling across epochs.
 *
 * @throws std::runtime_error If @p file_list is empty, files cannot be opened/parsed, or vocab sizes mismatch.
 */
DataLoader::DataLoader(const std::vector<std::string>& file_list, int seq_len, int rank, int world_size, unsigned long seed) :
        mSeqLen(seq_len), mSeed(seed), mRank(rank), mWorldSize(world_size), mChunkIndex(rank) {
    if (file_list.empty()) {
        throw std::runtime_error("Empty list of token files provided");
    }

    for(const auto& file_name: file_list) {
        mFileInfos.push_back(parse_token_file_header(file_name));
    }

    mVocabSize = mFileInfos[0].VocabSize;
    for (auto& info: mFileInfos) {
        if (info.VocabSize != mVocabSize) {
            throw std::runtime_error(fmt::format("Inconsistent vocabulary sizes. Expected {}, got {} in {}.", mVocabSize, info.VocabSize, info.FileName));
        }
    }

    std::int64_t total_chunks = 0;
    std::int64_t total_tokens = 0;
    for (auto& info: mFileInfos) {
        total_tokens += info.NumTokens;
        // NonOverlapping: each chunk is exactly seq_len tokens, no overlap
        // Overlapping: chunks overlap by seq_len-1 tokens (standard for training)
        if (info.NonOverlapping) {
            total_chunks += info.NumTokens / mSeqLen;
        } else {
            total_chunks += (info.NumTokens - 1) / mSeqLen;
        }
    }

    mTotalChunks = total_chunks;
    mTotalTokens = total_tokens;

    // this ensures that the first call to advance_epoch ends up in epoch 0
    mEpoch = -1;
    advance_epoch();
}

/**
 * @brief Expand a glob pattern into a sorted list of file paths.
 *
 * Uses POSIX glob() with tilde and brace expansion and returns a lexicographically sorted list.
 *
 * @param pattern Glob pattern to match.
 * @return Sorted list of matched file paths.
 *
 * @throws std::runtime_error If glob fails or no files match.
 */
std::vector<std::string> DataLoader::match_files(const std::string& pattern) {
    std::vector<std::string> files;

    glob_t glob_result;
    int ret = glob(pattern.c_str(), GLOB_TILDE | GLOB_BRACE, nullptr, &glob_result);

    if (ret == 0 || ret == GLOB_NOMATCH) {
        for (size_t i = 0; i < glob_result.gl_pathc; ++i) {
            files.emplace_back(glob_result.gl_pathv[i]);
        }
        std::ranges::sort(files);
    } else {
        globfree(&glob_result);
        throw std::runtime_error(fmt::format("Failed to match files with pattern '{}': {}", pattern, glob_result.gl_pathv[0]));
    }

    globfree(&glob_result);

    if (files.empty()) {
        throw std::runtime_error(fmt::format("No files found with pattern '{}'", pattern));
    }

    return files;
}

/**
 * @brief Parse and validate the header of a token file.
 *
 * Reads the fixed-size header and extracts metadata such as version, vocabulary size, token count,
 * and whether masks are present.
 *
 * @param file_name Path to the token file.
 * @return Parsed TokenFileInfo for @p file_name.
 *
 * @throws std::runtime_error If the file cannot be opened, the magic/version is invalid, or token size is unsupported.
 */
DataLoader::TokenFileInfo DataLoader::parse_token_file_header(const std::string& file_name) {
    std::ifstream token_file(file_name, std::ios::binary);
    if (!token_file.is_open() || !token_file.good()) {
        throw std::runtime_error("Could not open token file: " + file_name);
    }
    token_file.exceptions(std::ifstream::failbit);

    TokenFileInfo info{.FileName = file_name};

    // read the header
    int header[256];
    token_file.read((char*)header, sizeof(header));
    constexpr char MAGIC[] = {'B', 'I', 'N', '.', 'T', 'O', 'K', '\n'};
    if(std::memcmp(header, MAGIC, sizeof(MAGIC)) != 0) {
        throw std::runtime_error(fmt::format("Invalid token file: '{}'", std::string_view((char*)header, sizeof(MAGIC))));
    }

    int version = header[2];
    if(version == 2 || version == 3) {
        info.VocabSize = header[5];
    } else if(version != 1) {
        throw std::runtime_error(fmt::format("Unsupported token file version: {}", version));
    }

    int bytes_per_token = header[3];
    if(bytes_per_token != 4) {
        throw std::runtime_error(fmt::format("Unsupported bytes per token: {}", bytes_per_token));
    }

    info.Version = version;
    info.BytesPerToken = bytes_per_token;
    info.NumTokens = header[4];
    info.HasMasks = header[6] == 1;
    info.NonOverlapping = header[7] == 1;
    return info;
}

/**
 * @brief Compute loader progress through the current epoch as a percentage.
 *
 * Progress is measured as (tokens consumed so far in the epoch) / (total tokens across all files).
 * Note: this reflects the logical traversal order given current shuffling and rank-local chunk index.
 *
 * @return Progress in [0, 100] (floating point).
 */
float DataLoader::progress() const {
    std::int64_t epoch_tokens = 0;
    for (int i = 0; i < mFileIndex; ++i) {
        epoch_tokens += mShuffledFiles.at(i)->NumTokens;
    }
    epoch_tokens += mChunkIndex * mSeqLen;

    return 100.f * ((double)epoch_tokens / (double)mTotalTokens);
}

/**
 * @brief Shuffle file order for the current epoch deterministically.
 *
 * The shuffle is derived from (mSeed, mEpoch) using Philox, so it is reproducible across runs.
 */
void DataLoader::shuffle_files() {
    mShuffledFiles.clear();
    for (auto& file : mFileInfos) {
        mShuffledFiles.push_back(&file);
    }
    Philox4x32 rng{mSeed};
    auto shuffle_seed = rng.generate(mEpoch, 0x73653753);
    std::ranges::shuffle(mShuffledFiles, std::default_random_engine{shuffle_seed[0]});
}

/**
 * @brief Shuffle chunk order within the current file deterministically.
 *
 * Chunk IDs are [0, num_chunks) where num_chunks = (num_tokens - 1) / mSeqLen.
 * The shuffle is derived from (mSeed, mEpoch, mFileIndex) using Philox.
 */
void DataLoader::shuffle_chunks() {
    const auto* file_info = mShuffledFiles.at(mFileIndex);
    int num_tokens = file_info->NumTokens;
    int num_chunks = file_info->NonOverlapping
        ? (num_tokens / mSeqLen)
        : ((num_tokens - 1) / mSeqLen);
    std::ranges::iota_view ids(0, num_chunks);
    mChunkOffsets.assign(std::begin(ids), std::end(ids));

    Philox4x32 rng{mSeed};
    auto shuffle_seed = rng.generate(mEpoch, mFileIndex);
    std::ranges::shuffle(mChunkOffsets, std::default_random_engine{shuffle_seed[0]});
}

/**
 * @brief Advance to the next file in the epoch and prepare chunk shuffling.
 *
 * Opens the next file (binary), shuffles its chunks, and resets the rank-local chunk index to @c mRank.
 *
 * @return True if a new file was opened; false if there are no more files in the epoch.
 *
 * @throws std::runtime_error If the next file cannot be opened.
 */
bool DataLoader::advance_file() {
    ++mFileIndex;
    if (mFileIndex >= mShuffledFiles.size()) {
        return false;
    }

    // open the next file
    std::string file_name = mShuffledFiles.at(mFileIndex)->FileName;
    mTokenFile = std::ifstream(file_name, std::ios::binary);
    if (!mTokenFile.is_open() || !mTokenFile.good()) {
        throw std::runtime_error("Could not open token file: " + file_name);
    }
    mTokenFile.exceptions(std::ifstream::failbit);

    shuffle_chunks();

    // reset read position
    mChunkIndex = mRank;

    return true;
}

/**
 * @brief Advance to the next epoch and initialize iteration state.
 *
 * Increments @c mEpoch, shuffles file order, resets file index, and opens the first file.
 */
void DataLoader::advance_epoch() {
    ++mEpoch;
    shuffle_files();
    mFileIndex = -1;
    advance_file();
}

/**
 * @brief Check whether at least @p n more sequences can be loaded for this rank.
 *
 * This accounts for distributed striding by @c mWorldSize and the rank offset.
 *
 * @param n Number of upcoming sequences to test availability for.
 * @return True if at least @p n sequences remain; false otherwise.
 */
bool DataLoader::has_next(int n) const {
    if (mFileIndex != mShuffledFiles.size() - 1) {
        return true;
    }
    return mChunkIndex + n * mWorldSize - mRank < mChunkOffsets.size();
}

/**
 * @brief Get the current chunk index within the current file, normalized to rank-local indexing.
 *
 * Internally, @c mChunkIndex includes the rank offset; this returns @c (mChunkIndex - mRank).
 *
 * @return Rank-local chunk index.
 */
std::int32_t DataLoader::chunk_index() const {
    return mChunkIndex - mRank;
}

/**
 * @brief Load a single sequence (inputs and targets) from the current file/chunk position.
 *
 * Reads @c mSeqLen tokens into @p inputs and the subsequent @c mSeqLen tokens into @p targets
 * (targets are offset by +1 token). If masks are present, masked target positions are set to -100.
 *
 * @param inputs CPU tensor (Device == -1) with exactly @c mSeqLen elements.
 * @param targets CPU tensor (Device == -1) with exactly @c mSeqLen elements.
 *
 * @throws std::runtime_error On size/device mismatch, end-of-data, incomplete reads, or underlying I/O errors.
 */
void DataLoader::load_seq(Tensor& inputs, Tensor& targets, Tensor* position_ids) {
    assert(inputs.Device == -1);
    assert(targets.Device == -1);
    if(inputs.nelem() != mSeqLen) {
        throw std::runtime_error(fmt::format("Expected inputs tensor of {} elements, got {}", mSeqLen, inputs.nelem()));
    }
    if(targets.nelem() != mSeqLen) {
        throw std::runtime_error(fmt::format("Expected targets tensor of {} elements, got {}", mSeqLen, targets.nelem()));
    }

    const long header_offset = 1024;

    if (mChunkIndex + mWorldSize - mRank >= mChunkOffsets.size()) {
        if (!advance_file()) {
            throw std::runtime_error("No more files to load");
        }
    }

    try {
        const auto& file_info = mShuffledFiles.at(mFileIndex);
        const long input_bytes = inputs.bytes();
        const long target_bytes = targets.bytes();
        const long element_size = file_info->BytesPerToken;
        const int chunk_pos = mSeqLen * mChunkOffsets[mChunkIndex];
        const long input_offset = element_size * chunk_pos + header_offset;
        const long target_offset = input_offset + element_size;

        // Seek and read input data
        mTokenFile.seekg(input_offset, std::ios::beg);
        mTokenFile.read(reinterpret_cast<char*>(inputs.Data), input_bytes);

        // Verify we read the expected number of bytes
        if (mTokenFile.gcount() != static_cast<std::streamsize>(input_bytes)) {
            throw std::runtime_error("Incomplete read of input data: expected " +
                                     std::to_string(input_bytes) + " bytes, got " +
                                     std::to_string(mTokenFile.gcount()));
        }

        // Seek and read target data
        mTokenFile.seekg(target_offset, std::ios::beg);
        mTokenFile.read(reinterpret_cast<char*>(targets.Data), target_bytes);

        // Verify we read the expected number of bytes
        if (mTokenFile.gcount() != static_cast<std::streamsize>(target_bytes)) {
            throw std::runtime_error("Incomplete read of target data: expected " +
                                     std::to_string(target_bytes) + " bytes, got " +
                                     std::to_string(mTokenFile.gcount()));
        }

        if (position_ids != nullptr) {
            assert(position_ids->Device == -1);
            if (position_ids->nelem() != mSeqLen) {
                throw std::runtime_error(fmt::format("Expected position_ids tensor of {} elements, got {}", mSeqLen, position_ids->nelem()));
            }

            if (file_info->Version >= 3) {
                const long pos_ids_start = element_size * file_info->NumTokens + header_offset;
                const long read_offset = pos_ids_start + element_size * chunk_pos;

                mTokenFile.seekg(read_offset, std::ios::beg);
                mTokenFile.read(reinterpret_cast<char*>(position_ids->Data), position_ids->bytes());

                if (mTokenFile.gcount() != static_cast<std::streamsize>(position_ids->bytes())) {
                    throw std::runtime_error("Incomplete read of position_ids data");
                }
            } else {
                int* pos_ptr = position_ids->get<int>();
                for (int i = 0; i < mSeqLen; ++i) {
                    pos_ptr[i] = i;
                }
            }
        }

        if(file_info->HasMasks) {
            long masks_start = element_size * file_info->NumTokens + header_offset;
            if (file_info->Version >= 3) {
                // Skip over PositionIDs block
                masks_start += element_size * file_info->NumTokens;
            }
            const long mask_start = masks_start + chunk_pos / 8;
            const long mask_end = masks_start + (chunk_pos + mSeqLen + 7) / 8;
            mTokenFile.seekg(mask_start, std::ios::beg);
            mMaskBuffer.resize(mask_end - mask_start);
            mTokenFile.read(reinterpret_cast<char*>(mMaskBuffer.data()), mask_end - mask_start);
            int start = chunk_pos % 8;
            int end = start + mSeqLen;
            int* target_tokens = targets.get<int>();
            for(int i = start; i < end; ++i) {
                int byte_id = i / 8;
                int bit_id = i % 8;
                bool mask_bit = (mMaskBuffer[byte_id] >> bit_id) & 1;
                if(!mask_bit) {
                    target_tokens[i - start] = -100;
                }
            }
        }

        // Update position only after successful reads
        mChunkIndex += mWorldSize;

    } catch (const std::ios_base::failure& e) {
        throw std::runtime_error("File I/O error: " + std::string(e.what()));
    }
}

/**
 * @brief Load a batch of sequences into contiguous input/target tensors.
 *
 * Splits @p inputs and @p targets into @c batch_size shards along the first dimension and calls load_seq() per shard.
 *
 * @param inputs CPU tensor whose total element count is divisible by @c mSeqLen.
 * @param targets CPU tensor whose total element count is divisible by @c mSeqLen.
 *
 * @throws std::runtime_error If sharding or underlying sequence loading fails.
 */
void DataLoader::load_batch(Tensor& inputs, Tensor& targets, Tensor* position_ids) {
    int batch_size = div_exact((int)inputs.nelem(), mSeqLen);
    for (int i = 0; i < batch_size; ++i) {
        Tensor bi = shard_view(inputs, i, batch_size);
        Tensor bt = shard_view(targets, i, batch_size);
        if (position_ids) {
            Tensor bp = shard_view(*position_ids, i, batch_size);
            load_seq(bi, bt, &bp);
        } else {
            load_seq(bi, bt, nullptr);
        }
    }
}

/**
 * @brief Restore loader state for deterministic resumption.
 *
 * Sets seed/epoch/file index and restores the next chunk position. The provided @p chunk_index is expected
 * to be rank-local (i.e., what chunk_index() returns); internally the loader stores @c chunk_index + mRank.
 *
 * @param seed Seed to use for shuffling.
 * @param epoch Epoch number to restore.
 * @param file_index Index into the shuffled file list to restore.
 * @param chunk_index Rank-local chunk index within the current file (will be offset by @c mRank internally).
 */
void DataLoader::set_state(std::uint64_t seed, std::int32_t epoch, std::int32_t file_index, std::int32_t chunk_index) {
    mSeed = seed;
    mEpoch = epoch;
    mFileIndex = file_index;
    mChunkIndex = chunk_index + mRank;
}
