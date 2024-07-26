#include <iostream>
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
