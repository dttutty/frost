#include <nanobind/nanobind.h>
#include "planner_ownership.hpp"
#include "sampler_common.h"
#include "sampler_seq.hpp" // Include the template implementation directly because templates cannot be compiled separately.

namespace nb = nanobind;

// Fix 5: template functions must be defined before NB_MODULE.
template <typename ID_T, typename TIME_T>
void bind_samplers(nb::module_& m, const std::string& suffix) {
    // Create type aliases to keep the binding code readable.
    using TGraph = TemporalGraph<ID_T, TIME_T>;
    using S_Seq = SeqSampler<ID_T, TIME_T>;

    // 1. Bind the suffixed TemporalGraph type.
    nb::class_<TGraph>(m, ("TemporalGraph_" + suffix).c_str())
        .def(nb::init<>())
        .def(nb::init<typename TGraph::IntArray, typename TGraph::IntArray, typename TGraph::IntArray, typename TGraph::FloatArray>(),
             nb::arg("indptr"), nb::arg("indices"), nb::arg("eid"), nb::arg("ts"))
        .def_rw("indptr", &TGraph::indptr)
        .def_rw("indices", &TGraph::indices)
        .def_rw("eid", &TGraph::eid)
        .def_rw("ts", &TGraph::ts);

    // 2. Bind the suffixed SeqSampler type and register all methods.
    nb::class_<S_Seq>(m, ("SeqSampler_" + suffix).c_str())
        .def(nb::init<const TGraph&, const std::string&, std::optional<int>, double>(),
             nb::arg("graph"),
             nb::arg("sample_neighbor_strategy") = "recent",
             nb::arg("seed") = nb::none(),
             nb::arg("time_scaling_factor") = 0.0)
        .def("get_all_first_hop_neighbors", &S_Seq::get_all_first_hop_neighbors)
        .def("get_historical_neighbors", &S_Seq::get_historical_neighbors)
        .def("get_historical_neighbors_into", &S_Seq::get_historical_neighbors_into)
        .def("reset_random_state", &S_Seq::reset_random_state);
}

NB_MODULE(_core, m) {
    // Generate the full set of bindings for each supported dtype pair.
    bind_samplers<int32_t, float>(m, "32_32");
    bind_samplers<int64_t, float>(m, "64_32");
    bind_samplers<int64_t, int64_t>(m, "64_64");
    m.def(
        "plan_owned_routes",
        &planner_ownership::plan_owned_routes,
        nb::arg("updates"),
        nb::arg("needs"),
        nb::arg("prefer_higher_rank") = true);
    m.def(
        "materialize_node_scopes_32",
        &planner_ownership::materialize_node_scopes<int32_t>,
        nb::arg("nid"),
        nb::arg("n_root"),
        nb::arg("n_positive_root"),
        nb::arg("positive_neighbor_start"),
        nb::arg("positive_neighbor_end"),
        nb::arg("requested_scopes"));
    m.def(
        "materialize_node_scopes_64",
        &planner_ownership::materialize_node_scopes<int64_t>,
        nb::arg("nid"),
        nb::arg("n_root"),
        nb::arg("n_positive_root"),
        nb::arg("positive_neighbor_start"),
        nb::arg("positive_neighbor_end"),
        nb::arg("requested_scopes"));
}
