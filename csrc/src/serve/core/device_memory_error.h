#pragma once

#include <stdexcept>
#include <string>

namespace sinfer {

/// The device had no memory for an allocation the engine needed in the middle of a round.
///
/// Unlike every other CUDA failure this one leaves the context usable: nothing the round needed
/// was written, so the requests that needed the memory can be failed, their pages returned, and
/// the engine can go on with the next round. The executor catches this type to do exactly that;
/// anything else out of a round is still fatal to the worker.
class DeviceOutOfMemory final : public std::runtime_error {
public:
    explicit DeviceOutOfMemory(const std::string& what) : std::runtime_error(what) {}
};

} // namespace sinfer
