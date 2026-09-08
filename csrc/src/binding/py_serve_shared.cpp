// In-process GRPO serving, retaining the owners of borrowed trainer tensors.
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include "serve/http_server.h"
#include "core/device.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <thread>

namespace nb = nanobind;
namespace {

using Array = nb::ndarray<>;

Array bf16_array(nb::handle value, int device, bool allow_fp32 = false) {
    Array array = nb::cast<Array>(value);
    const bool bf16 = array.dtype().code == static_cast<std::uint8_t>(nb::dlpack::dtype_code::Bfloat) &&
                      array.dtype().bits == 16;
    const bool fp32 = array.dtype().code == static_cast<std::uint8_t>(nb::dlpack::dtype_code::Float) &&
                      array.dtype().bits == 32;
    if (array.device_type() != nb::device::cuda::value || array.device_id() != device ||
        !(bf16 || (allow_fp32 && fp32))) {
        throw std::invalid_argument("shared tensors must use the expected floating dtype on the serving CUDA device");
    }
    std::int64_t stride = 1;
    for (std::size_t i = array.ndim(); i-- > 0;) {
        if (array.stride(i) != stride) { throw std::invalid_argument("shared tensors must be contiguous"); }
        stride *= array.shape(i);
    }
    return array;
}

class SharedServer {
public:
    SharedServer(const std::string& artifact, nb::dict weights, nb::dict settings)
        : weights_owner_(std::move(weights)) {
        sinfer::serve::ServeOptions options;
        options.artifact_path = artifact;
        options.host = nb::cast<std::string>(settings["host"]);
        options.port = nb::cast<int>(settings["port"]);
        options.device = nb::cast<int>(settings["device"]);
        options.model_id_override = nb::cast<std::string>(settings["model"]);
        options.max_context = nb::cast<std::uint32_t>(settings["max_context"]);
        options.prefill_chunk = nb::cast<std::uint32_t>(settings["prefill_chunk"]);
        options.max_concurrency = nb::cast<std::uint32_t>(settings["max_concurrency"]);
        options.max_pending_requests = std::max(16U, options.max_concurrency);
        options.pending_timeout_ms = 300000;
        options.kv_capacity = sinfer::KvCapacityPolicy::explicit_capacity(
            nb::cast<std::uint32_t>(settings["kv_capacity"]));
        options.use_cuda_graph = nb::cast<bool>(settings["use_cuda_graph"]);
        options.enable_sleep_mode = true;
        options.enable_lora = true;
        options.max_loras = 1;
        options.max_lora_rank = nb::cast<std::uint32_t>(settings["rank"]);
        options.log_stats_interval_ms = 0;
        device_ = options.device;
        std::vector<std::pair<std::uintptr_t, std::uintptr_t>> ranges;
        for (auto [key, value] : weights_owner_) {
            const auto array = bf16_array(value, device_, true);
            sinfer::BorrowedTensor tensor;
            tensor.name = nb::cast<std::string>(key);
            tensor.data = array.data();
            tensor.device = device_;
            tensor.bytes = array.size() * (array.dtype().bits / 8);
            tensor.dtype = array.dtype().bits == 32 ? sinfer::SharedWeightDType::FP32
                                                    : sinfer::SharedWeightDType::BF16;
            for (std::size_t i = 0; i < array.ndim(); ++i) { tensor.shape.push_back(array.shape(i)); }
            options.borrowed_weights.push_back(tensor);
            const auto begin = reinterpret_cast<std::uintptr_t>(tensor.data);
            ranges.emplace_back(begin, begin + tensor.bytes);
        }
        if (ranges.empty()) { throw std::invalid_argument("shared weights must not be empty"); }
        std::sort(ranges.begin(), ranges.end());
        std::uintptr_t end = 0;
        for (auto [begin, next] : ranges) {
            if (next > end) { shared_bytes_ += next - std::max(begin, end); end = next; }
        }
        nb::gil_scoped_release release;
        const sinfer::ScopedDevice on_device(device_);
        // Validate and build before binding: cpp-httplib does not close an
        // unstarted listener when its Server is destroyed after a load failure.
        service_ = std::make_unique<sinfer::serve::GenerationService>(options);
        service_->begin_shared_training(); // publish the initial adapter before accepting rollouts
        http_ = std::make_unique<sinfer::serve::HttpServer>(std::move(options));
        http_->attach(*service_);
        if (!http_->bind()) { throw std::runtime_error("could not bind shared serving address"); }
        thread_ = std::thread([this] {
            try { http_->listen(); }
            catch (...) { listen_error_ = std::current_exception(); }
            listen_finished_ = true;
        });
        // close() must not race listen startup: httplib's stop() only closes a
        // running listener. Surface startup errors on the Python caller too.
        while (!http_->is_running() && !listen_finished_) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        if (listen_finished_) {
            thread_.join();
            if (listen_error_) { std::rethrow_exception(listen_error_); }
            throw std::runtime_error("shared HTTP listener failed to start");
        }
    }

    ~SharedServer() { close(); }

    void begin_training() {
        std::lock_guard lock(mutex_);
        require_open();
        service_->begin_shared_training();
    }

    void publish(const std::string& name, nb::list modules, std::uint64_t version) {
        std::vector<sinfer::DeviceAdapterModule> payloads;
        for (nb::handle item : modules) {
            nb::dict m = nb::cast<nb::dict>(item);
            const auto a = bf16_array(m["a"], device_);
            const auto b = bf16_array(m["b"], device_);
            if (a.ndim() != 2 || b.ndim() != 2 || a.shape(0) != b.shape(1)) {
                throw std::invalid_argument("adapter A must be [rank,in] and B [out,rank]");
            }
            payloads.push_back({nb::cast<int>(m["layer"]), nb::cast<std::string>(m["module"]),
                a.data(), b.data(), static_cast<int>(a.shape(0)), static_cast<int>(a.shape(1)),
                static_cast<int>(b.shape(0)), nb::cast<float>(m["scale"])});
        }
        nb::gil_scoped_release release;
        std::lock_guard lock(mutex_);
        require_open();
        if (published_ && version != version_ + 1) { throw std::invalid_argument("policy version must advance by one"); }
        const sinfer::ScopedDevice on_device(device_);
        service_->publish_shared_adapter(name, payloads);
        version_ = version;
        published_ = true;
    }

    nb::dict summary() {
        std::lock_guard lock(mutex_);
        require_open();
        nb::dict out;
        out["shared_base_bytes"] = shared_bytes_;
        out["serving_base_allocated_bytes"] = service_->memory_summary().weights.capacity_bytes;
        out["base_upload_bytes"] = service_->load_summary().host_to_device_bytes;
        out["policy_version"] = version_;
        out["sleeping"] = service_->is_sleeping();
        // The existing scheduler metric describes capacity while awake, even
        // when those physical pages have been released during sleep.
        out["serving_buffer_bytes"] = service_->resident_bytes();
        return out;
    }

    void close() {
        std::lock_guard lock(mutex_);
        if (!http_) { return; }
        http_->stop();
        if (thread_.joinable()) { thread_.join(); }
        http_.reset();
        service_.reset();
    }

private:
    void require_open() const { if (!service_) { throw std::runtime_error("shared server is closed"); } }
    // Retain all tensor owners through the engine and HTTP worker teardown.
    nb::dict weights_owner_;
    std::unique_ptr<sinfer::serve::GenerationService> service_;
    std::unique_ptr<sinfer::serve::HttpServer> http_;
    std::thread thread_;
    std::atomic<bool> listen_finished_{false};
    std::exception_ptr listen_error_;
    std::mutex mutex_;
    int device_ = 0;
    std::uint64_t shared_bytes_ = 0;
    std::uint64_t version_ = 0;
    bool published_ = false;
};

} // namespace

void bind_shared_server(nb::module_& m) {
    nb::class_<SharedServer>(m, "SharedServer")
        .def(nb::init<const std::string&, nb::dict, nb::dict>(),
             nb::arg("artifact"), nb::arg("weights"), nb::arg("settings"))
        .def("begin_training", &SharedServer::begin_training, nb::call_guard<nb::gil_scoped_release>())
        .def("publish", &SharedServer::publish, nb::arg("name"), nb::arg("modules"), nb::arg("version"))
        .def("summary", &SharedServer::summary)
        .def("close", &SharedServer::close, nb::call_guard<nb::gil_scoped_release>());
}
