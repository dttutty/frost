#include <nanobind/nanobind.h>
#include "sampler_common.h"
#include "sampler_seq.hpp" // 直接 include 模板实现，因为模板不能分离编译

namespace nb = nanobind;

// 修正 5：模板函数必须在 NB_MODULE 前面定义
template <typename ID_T, typename TIME_T>
void bind_samplers(nb::module_& m, const std::string& suffix) {
    // 为类型设置别名，方便代码阅读
    using TGraph = TemporalGraph<ID_T, TIME_T>;
    using S_Seq = SeqSampler<ID_T, TIME_T>;
    using S_Win = WindowSeqSampler<ID_T, TIME_T>;

    // 1. 绑定带有后缀的 TemporalGraph
    nb::class_<TGraph>(m, ("TemporalGraph_" + suffix).c_str())
        .def(nb::init<>())
        .def(nb::init<typename TGraph::IntArray, typename TGraph::IntArray, typename TGraph::IntArray, typename TGraph::FloatArray>(),
             nb::arg("indptr"), nb::arg("indices"), nb::arg("eid"), nb::arg("ts"))
        .def_rw("indptr", &TGraph::indptr)
        .def_rw("indices", &TGraph::indices)
        .def_rw("eid", &TGraph::eid)
        .def_rw("ts", &TGraph::ts);

    // 2. 绑定带有后缀的 SeqSampler 并注册它的所有方法
    nb::class_<S_Seq>(m, ("SeqSampler_" + suffix).c_str())
        .def(nb::init<const TGraph&, const std::string&, std::optional<int>>(),
             nb::arg("graph"), nb::arg("sample_neighbor_strategy") = "recent", nb::arg("seed") = nb::none())
        .def("get_all_first_hop_neighbors", &S_Seq::get_all_first_hop_neighbors)
        .def("get_historical_neighbors", &S_Seq::get_historical_neighbors)
        .def("reset_random_state", &S_Seq::reset_random_state);

    // 3. 绑定 WindowSeqSampler...
    // nb::class_<S_Win>(m, ("WindowSeqSampler_" + suffix).c_str()) ... 
}

NB_MODULE(_core, m) {
    // 自动生成两套完整的方法和类
    bind_samplers<int32_t, float>(m, "32_32");
    bind_samplers<int64_t, float>(m, "64_32");
}