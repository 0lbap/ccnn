#include <iostream>
#include <cstdlib>

#ifdef ENABLE_PROFILING
#include "papi.h"

struct ProfilingData {
  int retval = 0;
  int code[4];
  int EventSet = PAPI_NULL;
  long long values[4] = {0, 0, 0, 0};
  clock_t start, end;
  double cpu_time_used;
  std::string name = "";
};

ProfilingData profiling_init();

void profiling_print_header();

void profiling_print_results(ProfilingData &pd);

void profiling_start(ProfilingData &pd);

void profiling_stop(ProfilingData &pd);

void profiling_shutdown();

#else // If PAPI is not available or profiling is disabled

struct ProfilingData {
    std::string name = "";
};

inline ProfilingData profiling_init() {
  std::cerr << "Error: Profiling is not supported because PAPI is not available." << std::endl;
  exit(EXIT_FAILURE);
}

inline void profiling_print_header() {
  std::cerr << "Error: Profiling is not supported because PAPI is not available." << std::endl;
  exit(EXIT_FAILURE);
}

inline void profiling_print_results(ProfilingData &pd) {
  std::cerr << "Error: Profiling is not supported because PAPI is not available." << std::endl;
  exit(EXIT_FAILURE);
}

inline void profiling_start(ProfilingData &pd) {
  std::cerr << "Error: Profiling is not supported because PAPI is not available." << std::endl;
  exit(EXIT_FAILURE);
}

inline void profiling_stop(ProfilingData &pd) {
  std::cerr << "Error: Profiling is not supported because PAPI is not available." << std::endl;
  exit(EXIT_FAILURE);
}

inline void profiling_shutdown() {
  std::cerr << "Error: Profiling is not supported because PAPI is not available." << std::endl;
  exit(EXIT_FAILURE);
}

#endif // ENABLE_PROFILING
