// Copyright (c) 2026 Invergent SA. SPDX-License-Identifier: Apache-2.0
#pragma once
#include <string>
#include <vector>

namespace sinfer::speech {
// Decode an uploaded file in memory; external protocols and nested playlists
// are disabled. Output is mono 16 kHz float PCM, capped at 10 minutes.
std::vector<float> decode_audio(const std::string& bytes);
} // namespace sinfer::speech
