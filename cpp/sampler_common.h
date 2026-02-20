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


struct TemporalGraph {
    // 定义 ndarray 类型：只读，一维，CPU，C连续
    using IntArray = nb::ndarray<const int64_t, nb::shape<1>, nb::c_contig, nb::device::cpu>;
    using FloatArray = nb::ndarray<const float, nb::shape<1>, nb::c_contig, nb::device::cpu>;

    IntArray indptr;
    IntArray indices;
    IntArray eid;
    FloatArray ts;

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
        // Nanobind 自动将 std::vector 转为 numpy array，无需手动封装
        .def_rw("row", &TemporalGraphBlock::row)
        .def_rw("col", &TemporalGraphBlock::col)
        .def_rw("eid", &TemporalGraphBlock::eid)
        .def_rw("ts", &TemporalGraphBlock::ts)
        .def_rw("dts", &TemporalGraphBlock::dts)
        .def_rw("nodes", &TemporalGraphBlock::nodes)
        .def_rw("dim_in", &TemporalGraphBlock::dim_in)
        .def_rw("dim_out", &TemporalGraphBlock::dim_out);
}


// 绑定辅助函数
inline void bind_common(nb::module_& m) {
    nb::class_<TemporalGraph>(m, "TemporalGraph")
        .def(nb::init<>()) // 允许 Python 端构造一个空的，然后赋值
        .def_rw("indptr", &TemporalGraph::indptr)
        .def_rw("indices", &TemporalGraph::indices)
        .def_rw("eid", &TemporalGraph::eid)
        .def_rw("ts", &TemporalGraph::ts);
}