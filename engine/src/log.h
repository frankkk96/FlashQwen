#pragma once
#include <cstdarg>
#include <cstdio>
#include <ctime>

inline void LogLine(const char* level, const char* fmt, ...) {
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

#define LOG_INFO(...) LogLine("INFO", __VA_ARGS__)
#define LOG_WARN(...) LogLine("WARN", __VA_ARGS__)
#define LOG_ERROR(...) LogLine("ERROR", __VA_ARGS__)
