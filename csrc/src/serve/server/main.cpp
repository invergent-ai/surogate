#include "product/load_progress/load_progress.h"
#include "core/sleep.h"
#include "serve/model_scheduler.h"
#include "serve/console_log.h"
#include "serve/generation_service.h"
#include "serve/http_server.h"
#include "serve/serve_options.h"

#include <atomic>
#include <chrono>
#include <csignal>
#include <cstddef>
#include <exception>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <utility>

namespace {

std::atomic<sinfer::serve::HttpServer*> g_server{nullptr};

void handle_signal(int) {
    sinfer::serve::HttpServer* server = g_server.load();
    if (server != nullptr) { server->stop(); }
}

std::string format_bytes(std::size_t bytes) {
    constexpr double kMiB = 1024.0 * 1024.0;
    constexpr double kGiB = 1024.0 * kMiB;
    std::ostringstream out;
    out << std::fixed << std::setprecision(2);
    if (static_cast<double>(bytes) >= kGiB) {
        out << static_cast<double>(bytes) / kGiB << " GiB";
    } else {
        out << static_cast<double>(bytes) / kMiB << " MiB";
    }
    return out.str();
}

} // namespace

int main(int argc, char** argv) {
    // Only a command-line problem earns the usage text; a failure while loading or serving is
    // reported on its own.
    sinfer::serve::ServeOptions options;
    try {
        options = sinfer::serve::parse_serve_options(argc, argv);
    } catch (const std::exception& exception) {
        sinfer::serve::write_console_log(sinfer::serve::ConsoleLogLevel::Error, exception.what());
        std::cerr << sinfer::serve::serve_usage_text(argv[0]);
        return 1;
    }
    if (options.help_requested) {
        std::cout << sinfer::serve::serve_usage_text(argv[0]);
        return 0;
    }
    try {

        using Clock = std::chrono::steady_clock;
        sinfer::serve::HttpServer server(options);
        if (!server.bind()) {
            sinfer::serve::write_console_log(sinfer::serve::ConsoleLogLevel::Error,
                                             "failed to bind " + options.host + ':' +
                                                 std::to_string(options.port));
            return 1;
        }

        sinfer::serve::write_console_log(sinfer::serve::ConsoleLogLevel::Info, "loading model...");
        auto load_progress_options        = sinfer::product::stderr_load_progress_options();
        load_progress_options.line_prefix = [] {
            return sinfer::serve::current_console_log_prefix(sinfer::serve::ConsoleLogLevel::Info);
        };
        sinfer::product::LoadProgressRenderer load_progress(std::cerr,
                                                            std::move(load_progress_options));
        const auto load_start = Clock::now();
        sinfer::serve::GenerationService service(options, load_progress.callback());
        server.attach(service);
        // Extra models: each its own Engine in this process, constructed
        // SEQUENTIALLY -- startup accounting (KV auto-size, the graph
        // allowance) measures free-VRAM deltas and assumes it owns the GPU
        // while it runs. Concurrent construction attributed one engine's
        // allocations to another's graph capture and aborted it.
        std::vector<std::unique_ptr<sinfer::serve::GenerationService>> extra_services;
        // Overcommit: with sleep mode on, a later engine that cannot fit
        // constructs after the already-built, idle ones are put to sleep. The
        // scheduler then juggles the working set per request.
        std::vector<sinfer::serve::ModelScheduler::Entry> scheduled;
        scheduled.push_back({server.public_model_id(), &service});
        for (const auto& extra : options.extra_models) {
            sinfer::serve::ServeOptions extra_options = options;
            extra_options.artifact_path             = extra.artifact_path;
            extra_options.model_id_override         = extra.name;
            extra_options.kv_capacity =
                sinfer::KvCapacityPolicy::explicit_capacity(extra.kv_tokens);
            if (extra.max_num_seqs != 0) { extra_options.max_concurrency = extra.max_num_seqs; }
            if (extra.max_context != 0) { extra_options.max_context = extra.max_context; }
            extra_options.extra_models.clear();
            extra_options.enable_lora = false;
            extra_options.lora_modules.clear();
            // Speculative backends are artifact-specific, so extras never
            // inherit the primary's flag -- they opt in per model via
            // spec=mtp|dflash[,draft-tokens=N].
            extra_options.speculative = extra.speculative;
            const auto extra_start = Clock::now();
            try {
                extra_services.push_back(std::make_unique<sinfer::serve::GenerationService>(
                    extra_options, load_progress.callback()));
            } catch (const std::exception& first_error) {
                if (!options.enable_sleep_mode) { throw; }
                // Assume memory pressure: park everything idle and retry once.
                sinfer::serve::write_console_log(
                    sinfer::serve::ConsoleLogLevel::Info,
                    std::string("model '") + extra.name + "' did not fit awake (" +
                        first_error.what() + "); sleeping idle models and retrying");
                service.sleep();
                for (auto& built : extra_services) {
                    if (built->active_requests() == 0) { built->sleep(); }
                }
                extra_services.push_back(std::make_unique<sinfer::serve::GenerationService>(
                    extra_options, load_progress.callback()));
            }
            server.attach_extra(*extra_services.back());
            scheduled.push_back({extra.name, extra_services.back().get()});
            std::ostringstream extra_loaded;
            extra_loaded << "model '" << extra.name << "' loaded in "
                         << std::chrono::duration<double>(Clock::now() - extra_start).count()
                         << " s";
            sinfer::serve::write_console_log(sinfer::serve::ConsoleLogLevel::Info,
                                             extra_loaded.str());
        }
        std::ostringstream loaded;
        loaded << "model loaded in "
               << std::chrono::duration<double>(Clock::now() - load_start).count() << " s";
        sinfer::serve::write_console_log(sinfer::serve::ConsoleLogLevel::Info, loaded.str());

        const sinfer::MemorySummary memory = service.memory_summary();
        std::ostringstream capacity;
        capacity << "KV capacity "
                 << (memory.kv_capacity_mode == sinfer::KvCapacityMode::Automatic ? "auto"
                                                                                  : "explicit")
                 << " resolved=" << memory.kv_capacity
                 << " tokens pages=" << memory.kv_capacity_page_groups << '/'
                 << memory.kv_capacity_max_page_groups
                 << " runtime=" << format_bytes(memory.runtime_reservation_bytes)
                 << " free-after-weights=" << format_bytes(memory.available_after_weights_bytes)
                 << " free-after-startup=" << format_bytes(memory.available_after_startup_bytes)
                 << " headroom=" << format_bytes(memory.kv_capacity_headroom_bytes)
                 << " slack=" << format_bytes(memory.planned_slack_bytes)
                 << " graphs=" << format_bytes(memory.cuda_graph_observed_bytes) << '/'
                 << format_bytes(memory.cuda_graph_allowance_bytes);
        if (options.enable_vision) {
            const sinfer::MediaCacheSummary media = service.media_cache_summary();
            capacity << " media-workers=" << media.preprocess_threads
                     << " media-cache=" << format_bytes(media.capacity_bytes)
                     << " media-live=" << format_bytes(media.live_capacity_bytes);
        }
        sinfer::serve::write_console_log(sinfer::serve::ConsoleLogLevel::Info, capacity.str());

        sinfer::serve::write_console_log(sinfer::serve::ConsoleLogLevel::Info, "warming up...");
        service.warmup();

        g_server.store(&server);
        std::signal(SIGINT, handle_signal);
        std::signal(SIGTERM, handle_signal);

        std::unique_ptr<sinfer::serve::ModelScheduler> scheduler;
        if (options.enable_sleep_mode && !options.extra_models.empty()) {
            // Budget = what the awake models occupy plus what is still free,
            // minus headroom for transient allocations outside the arenas.
            std::size_t awake_bytes = 0;
            for (const auto& entry : scheduled) {
                if (!entry.service->is_sleeping()) {
                    awake_bytes += entry.service->resident_bytes();
                }
            }
            const std::size_t free_bytes = sinfer::device_free_bytes(options.devices.empty()
                                                                          ? options.device
                                                                          : options.devices.front());
            const std::size_t headroom = 1ULL << 30;
            const std::size_t budget =
                awake_bytes + (free_bytes > headroom ? free_bytes - headroom : 0);
            scheduler = std::make_unique<sinfer::serve::ModelScheduler>(std::move(scheduled),
                                                                        budget);
            server.attach_scheduler(*scheduler);
            std::ostringstream plan;
            plan << "multi-model scheduler: budget "
                 << static_cast<double>(budget) / (1024.0 * 1024.0 * 1024.0) << " GiB across "
                 << (options.extra_models.size() + 1) << " models";
            sinfer::serve::write_console_log(sinfer::serve::ConsoleLogLevel::Info, plan.str());
        }

        std::ostringstream listening;
        listening << "listening on http://" << options.host << ':' << options.port
                  << " (model id: " << server.public_model_id()
                  << ", auth: " << (options.api_key.empty() ? "disabled" : "bearer") << ')';
        sinfer::serve::write_console_log(sinfer::serve::ConsoleLogLevel::Info, listening.str());

        const bool ok = server.listen();
        g_server.store(nullptr);
        if (!ok) {
            sinfer::serve::write_console_log(sinfer::serve::ConsoleLogLevel::Error,
                                             "failed to bind " + options.host + ':' +
                                                 std::to_string(options.port));
            return 1;
        }
        return 0;
    } catch (const std::exception& exception) {
        sinfer::serve::write_console_log(sinfer::serve::ConsoleLogLevel::Error, exception.what());
        return 1;
    }
}
