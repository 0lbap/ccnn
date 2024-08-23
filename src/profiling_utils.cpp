#include <iomanip>
#include "profiling_utils.hpp"

#ifdef ENABLE_PROFILING

ProfilingData profiling_init() {
  /* PAPI variables */
  ProfilingData pd;

  /* Setup PAPI library and begin collecting data from the counters */
  pd.retval = PAPI_library_init(PAPI_VER_CURRENT);
  if (pd.retval != PAPI_VER_CURRENT){
    std::cerr << "PAPI library init error! (" << pd.retval << ")" << std::endl;
  }

  /* PAPI Events */
  pd.retval = PAPI_event_name_to_code("CPU_CYCLES", &pd.code[0]);
  pd.retval = PAPI_event_name_to_code("L2D_CACHE_ACCESS", &pd.code[1]);
  pd.retval = PAPI_event_name_to_code("L2D_CACHE_REFILL", &pd.code[2]);
  pd.retval = PAPI_event_name_to_code("L2D_WB_VICTIM", &pd.code[3]);

  /* Create the Event Set */
  pd.retval = PAPI_create_eventset(&pd.EventSet);
  if (pd.retval != PAPI_OK) {
    std::cerr << "Error: PAPI_create_eventset (" << pd.retval << ")" << std::endl;
  }
  
  /* Add Total Instructions Executed to our Event Set */
  if (PAPI_add_event(pd.EventSet, pd.code[0]) != PAPI_OK) {
    std::cerr << "Error: PAPI_add_event 0" << std::endl;
  }
  if (PAPI_add_event(pd.EventSet, pd.code[1]) != PAPI_OK) {
    std::cerr << "Error: PAPI_add_event 1" << std::endl;
  }
  if (PAPI_add_event(pd.EventSet, pd.code[2]) != PAPI_OK) {
    std::cerr << "Error: PAPI_add_event 2" << std::endl;
  }
  if (PAPI_add_event(pd.EventSet, pd.code[3]) != PAPI_OK) {
    std::cerr << "Error: PAPI_add_event 3" << std::endl;
  }

  return pd;
}

void profiling_print_header() {
  std::cout << std::setw(15) << std::left << "Name" 
            << std::setw(15) << "Cycles" 
            << std::setw(15) << "L2 Accesses" 
            << std::setw(15) << "L2 Misses" 
            << std::setw(15) << "L2 Hits" 
            << std::setw(15) << "L2 WB" 
            << std::setw(15) << "Time (ns)" 
            << std::endl;

  // Print a separator line
  std::cout << std::setw(105) << std::setfill('-') << "" << std::endl;
  std::cout << std::setfill(' '); // Reset the fill character
}

void profiling_print_results(ProfilingData &pd) {
  /* Print profiling results */
  // Data
  std::cout << std::fixed; // Ensure fixed-point notation
  std::cout << std::setw(15) << pd.name 
            << std::setw(15) << pd.values[0] 
            << std::setw(15) << pd.values[1] 
            << std::setw(15) << pd.values[2] 
            << std::setw(15) << (pd.values[1] - pd.values[2])
            << std::setw(15) << pd.values[3] 
            << std::setw(15) << pd.cpu_time_used 
            << std::endl;
}

void profiling_start(ProfilingData &pd) {
  /* Start counting events in the Event Set */
  if (PAPI_start(pd.EventSet) != PAPI_OK) {
    std::cerr << "Error: PAPI_start" << std::endl;
  }

  /* start timer */
  pd.start = clock();

  /* Reset the counting events in the Event Set */
  if (PAPI_reset(pd.EventSet) != PAPI_OK) {
    std::cerr << "Error: PAPI_reset" << std::endl;
  }
}

void profiling_stop(ProfilingData &pd) {
  /* Read the counting events in the Event Set */
  if (PAPI_read(pd.EventSet, pd.values) != PAPI_OK) {
    std::cerr << "Error: PAPI_read" << std::endl;
  }

  // Stop the counting events in the Event Set
  if (PAPI_stop(pd.EventSet, pd.values) != PAPI_OK) {
      std::cerr << "Error: PAPI_stop" << std::endl;
      exit(EXIT_FAILURE);
  }

  /* Read timer */    
  pd.end = clock();
  pd.cpu_time_used = 1000000000 * ((double)(pd.end - pd.start)) / CLOCKS_PER_SEC;
}

void profiling_shutdown() {
  PAPI_shutdown();
}

#endif // ENABLE_PROFILING
