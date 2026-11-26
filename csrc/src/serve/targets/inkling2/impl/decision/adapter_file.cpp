#include "targets/inkling2/impl/decision/adapter_file.h"

#include <cstring>
#include <fstream>
#include <stdexcept>

namespace sinfer::targets::inkling2::detail::decision {

namespace {

[[noreturn]] void bad(const std::filesystem::path& dir, const std::string& why) {
    throw std::invalid_argument("decision adapter " + dir.string() + ": " + why);
}

std::uint64_t element_bytes(const std::string& dtype) {
    if (dtype == "F32" || dtype == "I32") { return 4; }
    if (dtype == "BF16" || dtype == "F16") { return 2; }
    if (dtype == "U8" || dtype == "I8" || dtype == "F8_E4M3") { return 1; }
    return 0;
}

} // namespace

AdapterFile::AdapterFile(const std::filesystem::path& dir) : dir_(dir), weights_(dir / "decision.safetensors") {
    std::ifstream meta_file(dir / "decision_adapter.json");
    if (!meta_file) { bad(dir, "no decision_adapter.json"); }
    meta_ = nlohmann::json::parse(meta_file, nullptr, true);
    if (meta_.value("kind", std::string()) != "inkling2-decision") { bad(dir, "kind is not inkling2-decision"); }
    if (meta_.value("version", 0) != 1) { bad(dir, "unsupported version " + std::to_string(meta_.value("version", 0))); }

    std::ifstream f(weights_, std::ios::binary);
    if (!f) { bad(dir, "no decision.safetensors"); }
    std::uint64_t header_bytes = 0;
    f.read(reinterpret_cast<char*>(&header_bytes), sizeof(header_bytes));
    if (!f || header_bytes == 0 || header_bytes > (std::uint64_t{1} << 30)) { bad(dir, "bad safetensors header"); }
    std::string header(header_bytes, '\0');
    f.read(header.data(), static_cast<std::streamsize>(header_bytes));
    const nlohmann::json h = nlohmann::json::parse(header);
    const std::uint64_t payload = sizeof(header_bytes) + header_bytes;
    const auto file_bytes = std::filesystem::file_size(weights_);
    for (const auto& [name, entry] : h.items()) {
        if (name == "__metadata__") { continue; }
        AdapterObject o;
        o.dtype = entry.at("dtype").get<std::string>();
        for (const auto& d : entry.at("shape")) { o.shape.push_back(d.get<std::int64_t>()); }
        const auto begin = entry.at("data_offsets").at(0).get<std::uint64_t>();
        const auto end   = entry.at("data_offsets").at(1).get<std::uint64_t>();
        std::uint64_t n  = 1;
        for (auto d : o.shape) { n *= static_cast<std::uint64_t>(d); }
        const std::uint64_t eb = element_bytes(o.dtype);
        if (eb == 0 || end < begin || end - begin != n * eb || payload + end > file_bytes) {
            bad(dir, "object " + name + " has a bad dtype, shape or extent");
        }
        o.offset = payload + begin;
        o.bytes  = end - begin;
        objects_.emplace(name, std::move(o));
    }
}

const AdapterObject& AdapterFile::object(const std::string& name) const {
    const auto it = objects_.find(name);
    if (it == objects_.end()) { bad(dir_, "no object " + name); }
    return it->second;
}

std::vector<std::byte> AdapterFile::read(const std::string& name) const {
    const AdapterObject& o = object(name);
    std::vector<std::byte> out(o.bytes);
    std::ifstream f(weights_, std::ios::binary);
    f.seekg(static_cast<std::streamoff>(o.offset));
    f.read(reinterpret_cast<char*>(out.data()), static_cast<std::streamsize>(o.bytes));
    if (!f) { bad(dir_, "short read of " + name); }
    return out;
}

Mat AdapterFile::read_f32(const std::string& name) const {
    const AdapterObject& o = object(name);
    if (o.dtype != "F32") { bad(dir_, name + " is " + o.dtype + ", expected F32"); }
    Mat m;
    if (o.shape.empty()) {
        m.rows = m.cols = 1;
    } else {
        m.cols = o.shape.back();
        m.rows = 1;
        for (std::size_t i = 0; i + 1 < o.shape.size(); ++i) { m.rows *= o.shape[i]; }
    }
    const std::vector<std::byte> raw = read(name);
    m.v.resize(raw.size() / sizeof(float));
    std::memcpy(m.v.data(), raw.data(), raw.size());
    return m;
}

HeadTensors AdapterFile::heads() const {
    HeadTensors out;
    for (const auto& [name, o] : objects_) {
        if (name.rfind("heads/", 0) == 0) { out.emplace(name, read_f32(name)); }
    }
    return out;
}

} // namespace sinfer::targets::inkling2::detail::decision
