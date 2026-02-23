# CMakeLists.txt
cmake_minimum_required(VERSION 3.15)
project(tgl_sampler)

# Require C++17 for nanobind
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find necessary packages
find_package(Python 3.10 COMPONENTS Interpreter Development.Module REQUIRED)
find_package(nanobind CONFIG REQUIRED)
find_package(TBB REQUIRED)
find_package(OpenMP) # In case omp_set_num_threads is still used

# 1. Compile simple sampler
nanobind_add_module(sampler_core_simple sampler_simple.cpp)
target_link_libraries(sampler_core_simple PRIVATE TBB::tbb)
if(OpenMP_CXX_FOUND)
    target_link_libraries(sampler_core_simple PRIVATE OpenMP::OpenMP_CXX)
endif()

# 2. Compile window sampler
nanobind_add_module(sampler_core_window sampler_window.cpp)
target_link_libraries(sampler_core_window PRIVATE TBB::tbb)
if(OpenMP_CXX_FOUND)
    target_link_libraries(sampler_core_window PRIVATE OpenMP::OpenMP_CXX)
endif()