// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// Copyright (c) 2025, IST Austria, developed by Erik Schultheis
// SPDX-License-Identifier: Apache-2.0
//

#ifndef SUROGATE_TRAINING_DATALOADER_H
#define SUROGATE_TRAINING_DATALOADER_H

#include <fstream>
#include <string>
#include <vector>

#include <nlohmann/json_fwd.hpp>

class Tensor;

/*!
 * \brief The DataLoader handles loading pre-tokenized training/eval inputs from a given list of files.
 * \details The DataLoader is responsible for handling streaming loads (i.e., never loading the full file into memory)
 * and random shuffling, as well as matching input tokens and target tokens.
 *
 * Random shuffling works on two levels. First, the order of the shard files is randomly shuffled. Then, within each
 * file, the order of chunks is randomly shuffled. Once a full pass over all files is completed, both file order and
 * chunk order are reshuffled. Each shuffle is generated from a seed, which is in turn generated from a counter-based
 * random number generator. Thus, the full state of the distributed data loader is characterized by just four numbers:
 * The seed, the current epoch, the current file `file_index()`, and the current chunk `chunk_index()`.
 *
 * There are a few corner cases that need to be specified:
 *  1) Number of tokens in a file is not a multiple of the chunk size.
 *     Currently, this is handled by just dropping the remaining tokens.
 *  2) In a distributed setting, `load_batch()` does not need to load a single chunk, but one chunk for all ranks.
 *     Thus, it can happen that some ranks would run out of data while others can still serve more chunks. In this case,
 *     the affected workers could jump to the next file, but that would make tracking state very hard, as now workers are
 *     not synchronized anymore. Thus, instead, we also drop these chunks at the end of a shuffle.
 *     Note that upon reshuffling, a different set of chunks would be dropped, so contrary to the partial chunk problem
 *     this does not hide training data.
 *
 */
class DataLoader {
public:
    DataLoader(const std::string& file_pattern, int seq_len, int rank, int world_size, unsigned long seed = 42);
    DataLoader(const std::vector<std::string>& file_list, int seq_len, int rank, int world_size, unsigned long seed = 42);

    static std::vector<std::string> match_files(const std::string& pattern);

    //! Fills `inputs` and `targets` with the next chunk of token indices, where targets is shifted left by one position.
    void load_seq(Tensor& inputs, Tensor& targets, Tensor* position_ids = nullptr);
    //! Fills `inputs` and `targets` with a batch of sequences
    void load_batch(Tensor& inputs, Tensor& targets, Tensor* position_ids = nullptr);

    //! Increment the epoch counter, reset the iterators, and re-shuffle the files and chunks.
    void advance_epoch();

    //! Check if there is enough data left in the current epoch for `n` calls to `load_batch`
    bool has_next(int n) const;

    const std::string& file_name(int i) const { return mFileInfos.at(i).FileName; }
    std::int32_t file_tokens(int i) const { return mFileInfos.at(i).NumTokens; }
    int seq_len() const { return mSeqLen; }

    std::int32_t file_index() const { return mFileIndex; }
    std::int32_t chunk_index() const;
    std::int32_t epoch() const { return mEpoch; }
    std::uint64_t seed() const { return mSeed; }

    std::size_t num_chunks_in_file() const { return mChunkOffsets.size(); }

    std::size_t num_files() const { return mShuffledFiles.size(); }
    std::int64_t num_chunks() const { return mTotalChunks; }
    std::int64_t num_tokens() const { return mTotalTokens; }

    std::int32_t vocab_size() const { return mVocabSize; }
    float progress() const;

    void set_state(std::uint64_t seed, std::int32_t epoch, std::int32_t file_index, std::int32_t chunk_index);

private:
    void shuffle_files();
    void shuffle_chunks();
    bool advance_file();

    struct TokenFileInfo {
        std::string FileName;
        std::int32_t NumTokens;
        std::int32_t Version;
        std::int32_t VocabSize;
        std::int32_t BytesPerToken;
        bool HasMasks;
        bool NonOverlapping;  // If true, chunks don't overlap (validation/eval data)
    };

    static TokenFileInfo parse_token_file_header(const std::string& file_name);

    // immutable config
    std::int32_t mVocabSize = -1;
    std::int32_t mSeqLen;
    std::vector<TokenFileInfo> mFileInfos;

    int mRank;
    int mWorldSize;
    std::int64_t mTotalChunks;
    std::int64_t mTotalTokens;

    // state
    std::vector<const TokenFileInfo*> mShuffledFiles;
    std::vector<std::int32_t> mChunkOffsets;
    std::ifstream mTokenFile;
    std::int32_t mChunkIndex = 0;
    std::int32_t mFileIndex = 0;

    std::int32_t mEpoch = 0;

    unsigned long mSeed;

    // buffers
    std::vector<std::uint8_t> mMaskBuffer;
};

#endif //SUROGATE_TRAINING_DATALOADER_H
