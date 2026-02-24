// sampler_main.cpp
#include <nanobind/nanobind.h>
#include "sampler_common.h"

namespace nb = nanobind;

void bind_simple_sampler(nb::module_& m);
void bind_window_sampler(nb::module_& m);

// Define the single module named "sampler"
NB_MODULE(sampler, m) {
    // 1. Bind common structs first
    bind_common(m); 
    bind_temporal_graph_block(m);

    // 2. Bind the samplers
    bind_simple_sampler(m);
    bind_window_sampler(m);
}