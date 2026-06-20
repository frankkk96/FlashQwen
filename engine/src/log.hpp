// Minimal leveled logging to stderr. The project avoids heavy dependencies, so this is a ~30-line
// in-house logger rather than a library (spdlog et al.): a timestamp + level prefix in front of a
// printf-style message. Use LOG_INFO / LOG_WARN / LOG_ERROR.
#pragma once
#include <cstdio>
#include <cstdarg>
#include <ctime>

inline void log_line(const char* level, const char* fmt, ...) {
    char ts[16];
    std::time_t t = std::time(nullptr);
    std::strftime(ts, sizeof ts, "%H:%M:%S", std::localtime(&t));
    std::fprintf(stderr, "%s [%s] ", ts, level);
    std::va_list ap;
    va_start(ap, fmt);
    std::vfprintf(stderr, fmt, ap);
    va_end(ap);
    std::fputc('\n', stderr);
}

#define LOG_INFO(...)  log_line("INFO",  __VA_ARGS__)
#define LOG_WARN(...)  log_line("WARN",  __VA_ARGS__)
#define LOG_ERROR(...) log_line("ERROR", __VA_ARGS__)
