#pragma once

// An Inkling-2 decision adapter directory (surogate/serve/convert/inkling2/decision.py): `decision_adapter.json`
// (geometry, calibration, provenance) + `decision.safetensors` (the objects). This reader is host-only: it parses the
// two files, lists the objects and reads one object's bytes; the engine side uploads what it serves.

#include "targets/inkling2/impl/decision/heads.h"

#include <nlohmann/json.hpp>

#include <cstdint>
#include <filesystem>
#include <map>
#include <string>
#include <vector>

namespace sinfer::targets::inkling2::detail::decision {

struct AdapterObject {
    std::string dtype; ///< safetensors dtype ("BF16", "F32", ...)
    std::vector<std::int64_t> shape;
    std::uint64_t offset = 0; ///< from the start of the file
    std::uint64_t bytes  = 0;
};

class AdapterFile {
public:
    /// Opens `dir` and checks it is an inkling2-decision adapter of a version this build reads.
    explicit AdapterFile(const std::filesystem::path& dir);

    [[nodiscard]] const nlohmann::json& meta() const { return meta_; }
    [[nodiscard]] const std::map<std::string, AdapterObject>& objects() const { return objects_; }
    [[nodiscard]] bool has(const std::string& name) const { return objects_.count(name) != 0; }
    [[nodiscard]] const AdapterObject& object(const std::string& name) const;
    /// The object's bytes as stored.
    [[nodiscard]] std::vector<std::byte> read(const std::string& name) const;
    /// An F32 object as a matrix: [rows, cols] for 2-D, [1, n] for 1-D, rows = the product of the leading dims.
    [[nodiscard]] Mat read_f32(const std::string& name) const;
    /// Every `heads/...` object (F32).
    [[nodiscard]] HeadTensors heads() const;
    [[nodiscard]] const std::filesystem::path& dir() const { return dir_; }

private:
    std::filesystem::path dir_, weights_;
    nlohmann::json meta_;
    std::map<std::string, AdapterObject> objects_;
};

} // namespace sinfer::targets::inkling2::detail::decision
