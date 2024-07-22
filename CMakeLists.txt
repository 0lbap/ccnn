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
