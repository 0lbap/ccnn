cmake_minimum_required(VERSION 3.10)

# Set the project name and version
project(CCNN VERSION 1.0)

# Specify the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Specify the source directory
set(SOURCE_DIR ./src)

# Collect all .cpp files in the source directory
file(GLOB SOURCES "${SOURCE_DIR}/*.cpp")

# Add the executable
add_executable(ccnn ${SOURCES})

# Option to enable or disable profiling support
option(ENABLE_PROFILING "Enable profiling support with PAPI" ON)

if(ENABLE_PROFILING)
  # Find the PAPI library
  find_library(PAPI_LIB papi)
  
  if(PAPI_LIB)
    message(STATUS "PAPI found: ${PAPI_LIB}")
    # Define ENABLE_PROFILING macro for the source code
    target_compile_definitions(ccnn PRIVATE ENABLE_PROFILING)
    # Link PAPI to the executable
    target_link_libraries(ccnn ${PAPI_LIB})
  else()
    message(WARNING "PAPI library not found. Profiling support will be disabled.")
    # Disable profiling if PAPI is not found
    set(ENABLE_PROFILING OFF)
  endif()
else()
  message(STATUS "Profiling support is disabled.")
endif()
