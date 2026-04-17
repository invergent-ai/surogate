// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//

#include "crash_handler.h"

#include <atomic>
#include <climits>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sstream>
#include <pthread.h>
#include <unistd.h>
#include <sys/types.h>
#include <execinfo.h>
#include <cxxabi.h>
#include <sys/ucontext.h>
#include <dlfcn.h>

// libdw for DWARF debug info resolution (source file/line info)
#if BACKWARD_HAS_DW
#include <elfutils/libdw.h>
#include <elfutils/libdwfl.h>
#include <dwarf.h>
#endif

// backward-cpp configuration - must be defined before including backward.hpp
// These control which debug info libraries are used for stack traces
#ifndef BACKWARD_HAS_DW
#define BACKWARD_HAS_DW 0
#endif

#ifndef BACKWARD_HAS_DWARF
#define BACKWARD_HAS_DWARF 0
#endif

#define BACKWARD_HAS_UNWIND 1
#define BACKWARD_HAS_BACKTRACE 1
#define BACKWARD_HAS_BACKTRACE_SYMBOL 1

#include <backward.hpp>

namespace surogate {

namespace {

// Thread-safe flag to prevent recursive signal handling
std::atomic<bool> g_handler_installed{false};
std::atomic<bool> g_in_signal_handler{false};

// Storage for addr2line commands to print at the end
constexpr int MAX_DECODE_ADDRS = 32;
struct DecodeAddr {
    char object[256];
    uintptr_t offset;
};
DecodeAddr g_decode_addrs[MAX_DECODE_ADDRS];
int g_num_decode_addrs = 0;

// Signal names for better error messages
const char* signal_name(int sig) {
    switch (sig) {
        case SIGSEGV: return "SIGSEGV (Segmentation fault)";
        case SIGABRT: return "SIGABRT (Abort)";
        case SIGFPE: return "SIGFPE (Floating-point exception)";
        case SIGILL: return "SIGILL (Illegal instruction)";
        case SIGBUS: return "SIGBUS (Bus error)";
        case SIGTRAP: return "SIGTRAP (Trap)";
        default: return "Unknown signal";
    }
}

// Demangle a C++ symbol name (safe version using static buffer)
void demangle_safe(const char* name, char* output, size_t output_size) {
    if (!name || !output || output_size == 0) return;

    int status = 0;
    char* demangled = abi::__cxa_demangle(name, nullptr, nullptr, &status);
    if (status == 0 && demangled) {
        strncpy(output, demangled, output_size - 1);
        output[output_size - 1] = '\0';
        free(demangled);
    } else {
        strncpy(output, name ? name : "???", output_size - 1);
        output[output_size - 1] = '\0';
    }
}

// Record an address for later decoding (fallback if libdw not available)
void record_for_decode(const char* object, uintptr_t offset) {
    if (g_num_decode_addrs >= MAX_DECODE_ADDRS) return;
    strncpy(g_decode_addrs[g_num_decode_addrs].object, object, sizeof(g_decode_addrs[0].object) - 1);
    g_decode_addrs[g_num_decode_addrs].object[sizeof(g_decode_addrs[0].object) - 1] = '\0';
    g_decode_addrs[g_num_decode_addrs].offset = offset;
    g_num_decode_addrs++;
}

#if BACKWARD_HAS_DW
// Global dwfl handle for symbol resolution (initialized once)
static Dwfl* g_dwfl = nullptr;
static Dwfl_Callbacks g_dwfl_callbacks = {.find_elf = dwfl_linux_proc_find_elf,
                                          .find_debuginfo = dwfl_standard_find_debuginfo,
                                          .section_address = nullptr,
                                          .debuginfo_path = nullptr};

// Initialize libdw for the current process
static void init_dwfl() {
    if (g_dwfl) return;
    g_dwfl = dwfl_begin(&g_dwfl_callbacks);
    if (g_dwfl) {
        dwfl_linux_proc_report(g_dwfl, getpid());
        dwfl_report_end(g_dwfl, nullptr, nullptr);
    }
}

// Resolve an address to function name, source file, and line number
// Returns true if resolved, writes results to output buffers
static bool
resolve_address(void* addr, char* func_out, size_t func_size, char* file_out, size_t file_size, int* line_out) {
    if (!g_dwfl) init_dwfl();
    if (!g_dwfl) return false;

    Dwarf_Addr dw_addr = (Dwarf_Addr)(uintptr_t)addr;
    Dwfl_Module* module = dwfl_addrmodule(g_dwfl, dw_addr);
    if (!module) return false;

    // Get function name
    const char* func_name = dwfl_module_addrname(module, dw_addr);
    if (func_name && func_out) {
        demangle_safe(func_name, func_out, func_size);
    }

    // Get source file and line
    Dwfl_Line* line = dwfl_module_getsrc(module, dw_addr);
    if (line && file_out && line_out) {
        int lineno = 0;
        const char* src = dwfl_lineinfo(line, nullptr, &lineno, nullptr, nullptr, nullptr);
        if (src) {
            strncpy(file_out, src, file_size - 1);
            file_out[file_size - 1] = '\0';
            *line_out = lineno;
            return true;
        }
    }

    return func_name != nullptr;
}
#endif

// Safe write helper - async-signal-safe
static void safe_write(const char* s) {
    if (s) write(STDERR_FILENO, s, strlen(s));
}

// Write an unsigned long as decimal (async-signal-safe)
static void safe_write_ulong(unsigned long val) {
    char buf[24];
    char* p = buf + sizeof(buf) - 1;
    *p = '\0';
    do {
        *(--p) = '0' + (val % 10);
        val /= 10;
    } while (val > 0);
    safe_write(p);
}

// Write a signed long as decimal (async-signal-safe)
static void safe_write_long(long val) {
    if (val < 0) {
        safe_write("-");
        // Handle INT_MIN edge case
        if (val == LONG_MIN) {
            safe_write_ulong((unsigned long)LONG_MAX + 1);
        } else {
            safe_write_ulong((unsigned long)(-val));
        }
    } else {
        safe_write_ulong((unsigned long)val);
    }
}

// Write a pointer as hex (async-signal-safe)
static void safe_write_ptr(const void* ptr) {
    char buf[20];
    uintptr_t val = (uintptr_t)ptr;
    buf[0] = '0';
    buf[1] = 'x';
    for (int i = 15; i >= 0; i--) {
        int nibble = (val >> (i * 4)) & 0xF;
        buf[17 - i] = nibble < 10 ? ('0' + nibble) : ('a' + nibble - 10);
    }
    buf[18] = '\0';
    safe_write(buf);
}

// The actual signal handler (using sigaction with SA_SIGINFO for context)
void crash_signal_handler(int sig, siginfo_t* info, void* context) {
    // Prevent recursive crashes in the signal handler
    bool expected = false;
    if (!g_in_signal_handler.compare_exchange_strong(expected, true)) {
        // Already in signal handler - just abort immediately
        _exit(128 + sig);
    }

    // Reset decode state
    g_num_decode_addrs = 0;

    // Print header immediately using direct writes (async-signal-safe)
    safe_write("\n================================================================================\n");
    safe_write("FATAL ERROR: ");
    safe_write(signal_name(sig));
    safe_write("\n================================================================================\n");

    // Signal info
    if (info) {
        safe_write("Signal code: ");
        safe_write_long(info->si_code);
        safe_write("\n");
        if (sig == SIGSEGV || sig == SIGBUS) {
            safe_write("Fault address: ");
            safe_write_ptr(info->si_addr);
            safe_write("\n");
        }
    }

    // Thread info
    safe_write("Thread ID: ");
    safe_write_ulong((unsigned long)gettid());
    safe_write(" (pthread: ");
    safe_write_ulong((unsigned long)pthread_self());
    safe_write(")\n");

    // Register state and crash location
    if (context) {
        ucontext_t* uc = static_cast<ucontext_t*>(context);
        mcontext_t* mc = &uc->uc_mcontext;

        safe_write("\nCPU Registers: RIP=");
        safe_write_ptr((void*)mc->gregs[REG_RIP]);
        safe_write(" RSP=");
        safe_write_ptr((void*)mc->gregs[REG_RSP]);
        safe_write(" RBP=");
        safe_write_ptr((void*)mc->gregs[REG_RBP]);
        safe_write("\n");

        // Resolve crash location
        void* rip = (void*)mc->gregs[REG_RIP];
        Dl_info dl_info;
        memset(&dl_info, 0, sizeof(dl_info));

        safe_write("\nCrash location: ");
        safe_write_ptr(rip);
        if (dladdr(rip, &dl_info) && dl_info.dli_fname) {
            safe_write(" in ");
            safe_write(dl_info.dli_fname);
            if (dl_info.dli_fbase) {
                uintptr_t offset = (uintptr_t)rip - (uintptr_t)dl_info.dli_fbase;
                safe_write(" (+");
                safe_write_ptr((void*)offset);
                safe_write(")");
                record_for_decode(dl_info.dli_fname, offset);
            }
            if (dl_info.dli_sname) {
                safe_write("\n  => ");
                char demangled[512];
                demangle_safe(dl_info.dli_sname, demangled, sizeof(demangled));
                safe_write(demangled);
            }
        }
        safe_write("\n");
    }

    // Stack trace - try multiple methods
    safe_write("\nStack trace:\n");

    constexpr int MAX_FRAMES = 64;
    void* buffer[MAX_FRAMES];
    int num_frames = 0;
    int frame_idx = 0;

    // Method 1: Manual RBP-based stack walk from signal context
    // This is more reliable in signal handlers than backtrace()
    if (context) {
        ucontext_t* uc = static_cast<ucontext_t*>(context);
        mcontext_t* mc = &uc->uc_mcontext;

        // Start with crash location (RIP)
        buffer[num_frames++] = (void*)mc->gregs[REG_RIP];

        // Walk the frame pointer chain
        void** rbp = (void**)mc->gregs[REG_RBP];
        while (num_frames < MAX_FRAMES && rbp != nullptr) {
            // Validate pointer is in a reasonable range
            if ((uintptr_t)rbp < 0x1000 || (uintptr_t)rbp > 0x7fffffffffff) break;

            // Return address is at rbp[1]
            void* ret_addr = rbp[1];
            if (ret_addr == nullptr || (uintptr_t)ret_addr < 0x1000) break;

            buffer[num_frames++] = ret_addr;

            // Previous frame pointer is at rbp[0]
            void** next_rbp = (void**)rbp[0];

            // Sanity check: frame pointers should go up in memory (stack grows down)
            if (next_rbp <= rbp) break;
            rbp = next_rbp;
        }
    }

    // Method 2: Fallback to backtrace() if RBP walk failed
    if (num_frames <= 1) {
        num_frames = backtrace(buffer, MAX_FRAMES);
        frame_idx = 3;  // Skip signal handler frames
        if (frame_idx >= num_frames) frame_idx = 0;
    }

    if (num_frames > frame_idx) {
        for (int i = frame_idx; i < num_frames; i++) {
            safe_write("  #");
            safe_write_ulong(i - frame_idx);
            safe_write(" ");

            void* addr = buffer[i];
            Dl_info dl_info;
            memset(&dl_info, 0, sizeof(dl_info));

            safe_write_ptr(addr);

#if BACKWARD_HAS_DW
            // Try libdw resolution first (gives source file:line)
            char func_name[512] = {0};
            char source_file[512] = {0};
            int line_num = 0;
            bool resolved =
                resolve_address(addr, func_name, sizeof(func_name), source_file, sizeof(source_file), &line_num);

            if (dladdr(addr, &dl_info) && dl_info.dli_fname) {
                // Show short filename for readability
                const char* short_name = strrchr(dl_info.dli_fname, '/');
                short_name = short_name ? short_name + 1 : dl_info.dli_fname;
                safe_write(" in ");
                safe_write(short_name);
            }

            if (resolved && func_name[0]) {
                safe_write("\n       ");
                safe_write(func_name);
            } else if (dl_info.dli_sname) {
                safe_write("\n       ");
                char demangled[512];
                demangle_safe(dl_info.dli_sname, demangled, sizeof(demangled));
                safe_write(demangled);
            }

            if (resolved && source_file[0] && line_num > 0) {
                safe_write("\n       at ");
                safe_write(source_file);
                safe_write(":");
                safe_write_ulong(line_num);
            }
#else
            // Fallback without libdw
            if (dladdr(addr, &dl_info) && dl_info.dli_fname) {
                safe_write(" in ");
                safe_write(dl_info.dli_fname);
                if (dl_info.dli_fbase) {
                    uintptr_t offset = (uintptr_t)addr - (uintptr_t)dl_info.dli_fbase;
                    safe_write(" (+");
                    safe_write_ptr((void*)offset);
                    safe_write(")");
                    record_for_decode(dl_info.dli_fname, offset);
                }
                if (dl_info.dli_sname) {
                    safe_write("\n       => ");
                    char demangled[512];
                    demangle_safe(dl_info.dli_sname, demangled, sizeof(demangled));
                    safe_write(demangled);
                }
            }
#endif
            safe_write("\n");
        }
    } else {
        safe_write("  (no stack frames captured)\n");
    }

#if !BACKWARD_HAS_DW
    // Only show manual decode commands if libdw is not available
    if (g_num_decode_addrs > 0) {
        safe_write("\nTo decode with source lines, run:\n");
        for (int i = 0; i < g_num_decode_addrs && i < 10; i++) {
            safe_write("  addr2line -e ");
            safe_write(g_decode_addrs[i].object);
            safe_write(" -f -C ");
            safe_write_ptr((void*)g_decode_addrs[i].offset);
            safe_write("\n");
        }
    }

    // Hints
    safe_write("\n================================================================================\n");
    safe_write("Hints:\n");
    safe_write("  - For better stack traces: sudo apt install libdw-dev\n");
    safe_write("  - For CUDA crashes: CUDA_LAUNCH_BLOCKING=1\n");
    safe_write("================================================================================\n\n");
#else
    // With libdw, just show CUDA hint
    safe_write("\n================================================================================\n");
    safe_write("Hint: For CUDA crashes, set CUDA_LAUNCH_BLOCKING=1\n");
    safe_write("================================================================================\n\n");
#endif

    // Ensure all output is flushed before terminating
    // fsync on stderr ensures kernel buffers are flushed
    fsync(STDERR_FILENO);

    // Terminate process - use _exit() which is async-signal-safe
    _exit(128 + sig);
}

}  // anonymous namespace

void install_crash_handler() {
    bool expected = false;
    if (!g_handler_installed.compare_exchange_strong(expected, true)) {
        // Already installed
        return;
    }

    // Install handlers for common crash signals using SA_SIGINFO for more context
    struct sigaction sa;
    memset(&sa, 0, sizeof(sa));
    sa.sa_sigaction = crash_signal_handler;
    sigemptyset(&sa.sa_mask);
    // SA_SIGINFO: Use sa_sigaction instead of sa_handler (provides signal context)
    // SA_RESETHAND: Reset handler to default after first signal (prevents loops)
    // SA_NODEFER: Don't block the signal while handling it
    sa.sa_flags = SA_SIGINFO | SA_RESETHAND | SA_NODEFER;

    sigaction(SIGSEGV, &sa, nullptr);
    sigaction(SIGABRT, &sa, nullptr);
    sigaction(SIGFPE, &sa, nullptr);
    sigaction(SIGILL, &sa, nullptr);
    sigaction(SIGBUS, &sa, nullptr);
    sigaction(SIGTRAP, &sa, nullptr);
}

std::string capture_stacktrace(int skip_frames, int max_frames) {
    backward::StackTrace st;
    st.load_here(max_frames + skip_frames + 1);
    st.skip_n_firsts(skip_frames + 1);  // +1 to skip this function

    std::ostringstream oss;
    backward::Printer printer;
    printer.snippet = false;  // No source snippets for string output
    printer.color_mode = backward::ColorMode::never;
    printer.address = true;
    printer.object = true;
    printer.print(st, oss);

    return oss.str();
}

void print_stacktrace(int max_frames) {
    backward::StackTrace st;
    st.load_here(max_frames + 2);
    st.skip_n_firsts(2);  // Skip this function and load_here

    backward::Printer printer;
    printer.snippet = true;
    printer.color_mode = backward::ColorMode::automatic;
    printer.address = true;
    printer.object = true;
    printer.print(st, stderr);
}

}  // namespace surogate
