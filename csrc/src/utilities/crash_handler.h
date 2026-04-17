// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//

#ifndef SUROGATE_SRC_UTILITIES_CRASH_HANDLER_H
#define SUROGATE_SRC_UTILITIES_CRASH_HANDLER_H

#include <string>

namespace surogate {

/**
 * @brief Install signal handlers that print stack traces on crash.
 *
 * Installs handlers for SIGSEGV, SIGABRT, SIGFPE, SIGILL, SIGBUS, and SIGTRAP.
 * When a crash occurs, a detailed stack trace with source file/line info
 * (if debug symbols are available) will be printed to stderr before the
 * process terminates.
 *
 * This function should be called once at program startup. Multiple calls
 * are safe but have no additional effect.
 *
 * @note For best results, compile with debug symbols (-g) and ensure
 *       libdw (elfutils) is available on the system.
 */
void install_crash_handler();

/**
 * @brief Capture and return the current stack trace as a string.
 *
 * Useful for logging or embedding stack traces in exception messages.
 *
 * @param skip_frames Number of stack frames to skip from the top (default: 1 to skip this function itself).
 * @param max_frames Maximum number of frames to capture (default: 32).
 * @return A formatted string containing the stack trace.
 */
std::string capture_stacktrace(int skip_frames = 1, int max_frames = 32);

/**
 * @brief Print the current stack trace to stderr.
 *
 * Convenience function for debugging. Skips the call to this function itself.
 *
 * @param max_frames Maximum number of frames to print (default: 32).
 */
void print_stacktrace(int max_frames = 32);

}  // namespace surogate

#endif  // SUROGATE_SRC_UTILITIES_CRASH_HANDLER_H
