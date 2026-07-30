cmake_minimum_required(VERSION 3.15...3.27)
project(seq_sampler_module)

# Building a Python extension only needs the interpreter plus module headers.
find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
find_package(nanobind CONFIG REQUIRED)

# Build the nanobind extension module.
nanobind_add_module(_core
    src/frost/seq_sampler/seq_sampler_main.cpp
)

# Configure compiler options.
target_compile_features(_core PRIVATE cxx_std_17)
target_compile_options(_core PRIVATE 
    -O3 
    -g 
    -fvisibility=hidden 
    -fopenmp
)

# Link OpenMP when available.
target_link_options(_core PRIVATE -fopenmp)

# Add the current source directory to the include path.
target_include_directories(_core PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/cpp)

install(TARGETS _core DESTINATION frost/seq_sampler)



