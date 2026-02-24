#pragma once
#include <vector>
#include <random>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h> // <--- 必须引入 ndarray
#include <nanobind/stl/vector.h> // 让 nanobind 自动转换 vector

namespace nb = nanobind;

using NodeIDType = int64_t; // 对应 Python 的 np.int64
using EdgeIDType = int64_t; // 对应 Python 的 np.int64
using TimeStampType = float; // 对应 Python 的 np.float32


static constexpr TimeStampType kEps = 1e-7f;


// 在 sampler_common.h 中添加这个辅助函数
template <typename T>
inline nb::ndarray<nb::numpy, T> vec2npy(std::vector<T>& vec) {
    // 窃取 vector 的内存管理权，通过 capsule 在 Python 端释放
    auto* v_ptr = new std::vector<T>(std::move(vec)); 
    nb::capsule deleter(v_ptr, [](void* p) noexcept {
        delete reinterpret_cast<std::vector<T>*>(p);
    });
    return nb::ndarray<nb::numpy, T>(
        v_ptr->data(), 
        { v_ptr->size() }, // 形状
        deleter            // 绑定生命周期
    );
}

struct TemporalGraph {
    // Define 1D continuous CPU arrays, without const to accept normal numpy arrays
    using IntArray = nb::ndarray<int64_t, nb::ndim<1>, nb::c_contig, nb::device::cpu>;
    using FloatArray = nb::ndarray<float, nb::ndim<1>, nb::c_contig, nb::device::cpu>;

    IntArray indptr;
    IntArray indices;
    IntArray eid;
    FloatArray ts;

    TemporalGraph() = default;

    TemporalGraph(IntArray _indptr, IntArray _indices, IntArray _eid, FloatArray _ts)
        : indptr(_indptr), indices(_indices), eid(_eid), ts(_ts) {}

    // 辅助函数：获取原始指针 (这一步是极致性能的关键)
    const int64_t* indptr_ptr() const { return indptr.data(); }
    const int64_t* indices_ptr() const { return indices.data(); }
    const int64_t* eid_ptr() const { return eid.data(); }
    const float* ts_ptr() const { return ts.data(); }
};


// 1. 数据结构 (保持简洁)
struct TemporalGraphBlock {
    std::vector<NodeIDType> row, col, nodes;
    std::vector<EdgeIDType> eid;
    std::vector<TimeStampType> ts, dts;
    NodeIDType dim_in = 0, dim_out = 0;
    
    // Profiling
    double search_time = 0, sample_time = 0, coo_time = 0, tot_time = 0;
};

// 2. 随机数 (改用更标准的 PCG 或 Xoshiro，这里保留你的实现但封装成 Functor)
struct Xorshift32 {
    uint32_t state;
    explicit Xorshift32(uint32_t seed) : state(seed) {}
    inline uint32_t operator()() {
        state ^= state << 13; state ^= state >> 17; state ^= state << 5;
        return state;
    }
};

// 3. 绑定辅助
inline void bind_temporal_graph_block(nb::module_& m) {
    nb::class_<TemporalGraphBlock>(m, "TemporalGraphBlock")
        .def(nb::init<>())
        // 注意：vec2npy 用了 std::move，这意味着在 Python 获取后，C++ 端的 vector 会被清空
        .def("row", [](TemporalGraphBlock& b) { return vec2npy(b.row); }, nb::rv_policy::reference)
        .def("col", [](TemporalGraphBlock& b) { return vec2npy(b.col); }, nb::rv_policy::reference)
        .def("eid", [](TemporalGraphBlock& b) { return vec2npy(b.eid); }, nb::rv_policy::reference)
        .def("ts", [](TemporalGraphBlock& b) { return vec2npy(b.ts); }, nb::rv_policy::reference)
        .def("dts", [](TemporalGraphBlock& b) { return vec2npy(b.dts); }, nb::rv_policy::reference)
        .def("nodes", [](TemporalGraphBlock& b) { return vec2npy(b.nodes); }, nb::rv_policy::reference)
        .def("dim_in", [](TemporalGraphBlock& b) { return b.dim_in; })
        .def("dim_out", [](TemporalGraphBlock& b) { return b.dim_out; });
}


// 绑定辅助函数
inline void bind_common(nb::module_& m) {
    nb::class_<TemporalGraph>(m, "TemporalGraph")
        .def(nb::init<>()) // 允许 Python 端构造一个空的，然后赋值
        .def(nb::init<TemporalGraph::IntArray, TemporalGraph::IntArray, TemporalGraph::IntArray, TemporalGraph::FloatArray>(),
             nb::arg("indptr"), nb::arg("indices"), nb::arg("eid"), nb::arg("ts"))
        .def_rw("indptr", &TemporalGraph::indptr)
        .def_rw("indices", &TemporalGraph::indices)
        .def_rw("eid", &TemporalGraph::eid)
        .def_rw("ts", &TemporalGraph::ts);
}