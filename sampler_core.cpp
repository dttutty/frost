// sampler_core.cpp (refactor)
#include <algorithm>
#include <cstdint>
#include <cmath>
#include <random>
#include <vector>

#include <omp.h>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

namespace py = pybind11;

using NodeIDType = int;
using EdgeIDType = int;
using TimeStampType = float;

static constexpr TimeStampType kEps = 1e-7f;

class TemporalGraphBlock {
public:
    std::vector<NodeIDType> row;   // per-edge: dst index
    std::vector<NodeIDType> col;   // per-edge: src index
    std::vector<EdgeIDType> eid;   // per-edge: global edge id

    // per-node (src-side list): length = dim_in
    std::vector<TimeStampType> ts;   // timestamp for each block-node
    std::vector<TimeStampType> dts;  // DyX time for each block-node (root nodes -> 0)
    std::vector<NodeIDType> nodes;   // global node id for each block-node

    NodeIDType dim_in = 0;   // #src nodes in block
    NodeIDType dim_out = 0;  // #dst nodes in block

    // (可选) profiling
    double search_time = 0;
    double sample_time = 0;
    double coo_time = 0;
    double tot_time = 0;

    TemporalGraphBlock() = default;

    TemporalGraphBlock(const std::vector<NodeIDType>& _row,
                       const std::vector<NodeIDType>& _col,
                       const std::vector<EdgeIDType>& _eid,
                       const std::vector<TimeStampType>& _ts,
                       const std::vector<TimeStampType>& _dts,
                       const std::vector<NodeIDType>& _nodes,
                       NodeIDType _dim_in,
                       NodeIDType _dim_out)
        : row(_row), col(_col), eid(_eid), ts(_ts), dts(_dts),
          nodes(_nodes), dim_in(_dim_in), dim_out(_dim_out) {}
};

template <typename T>
inline py::array vec2npy(const std::vector<T>& vec) {
    auto v = new std::vector<T>(vec);
    auto capsule = py::capsule(v, [](void* p) {
        delete reinterpret_cast<std::vector<T>*>(p);
    });
    return py::array(v->size(), v->data(), capsule);
}

class ParallelSampler {
public:
    // CSR adjacency
    std::vector<EdgeIDType> indptr;          // len = num_nodes + 1
    std::vector<NodeIDType> indices;         // len = num_edges, neighbor node ids
    std::vector<EdgeIDType> edge_ids;        // len = num_edges
    std::vector<TimeStampType> edge_ts;      // len = num_edges, must be sorted within each node segment

    NodeIDType num_nodes = 0;
    EdgeIDType num_edges = 0;

    int num_thread_per_worker = 1;
    int num_workers = 1;
    int num_threads = 1;

    int num_layers = 1;
    std::vector<int> num_neighbors;  // per layer

    bool recent = false;     // if true: take most recent edges in window
    bool prop_time = false;  // if true: neighbor-node ts uses dst_ts (propagated), else uses edge_ts

    int num_history = 1;
    TimeStampType window_duration = 0;

    std::vector<TemporalGraphBlock> ret;

    ParallelSampler(const std::vector<EdgeIDType>& _indptr,
                    const std::vector<NodeIDType>& _indices,
                    const std::vector<EdgeIDType>& _eid,
                    const std::vector<TimeStampType>& _ts,
                    int _num_thread_per_worker,
                    int _num_workers,
                    int _num_layers,
                    const std::vector<int>& _num_neighbors,
                    bool _recent,
                    bool _prop_time,
                    int _num_history,
                    TimeStampType _window_duration)
        : indptr(_indptr),
          indices(_indices),
          edge_ids(_eid),
          edge_ts(_ts),
          num_thread_per_worker(_num_thread_per_worker),
          num_workers(_num_workers),
          num_layers(_num_layers),
          num_neighbors(_num_neighbors),
          recent(_recent),
          prop_time(_prop_time),
          num_history(_num_history),
          window_duration(_window_duration) {
        num_threads = std::max(1, num_thread_per_worker * num_workers);
        omp_set_num_threads(num_threads);

        num_nodes = static_cast<NodeIDType>(indptr.size() - 1);
        num_edges = static_cast<EdgeIDType>(indices.size());
    }

    void reset() {
        // 现在没有 ts_ptr 缓存，这里保留接口占位
    }

private:
    struct ThreadBuffer {
        std::vector<NodeIDType> row;
        std::vector<NodeIDType> col;
        std::vector<EdgeIDType> eid;
        std::vector<TimeStampType> ts;
        std::vector<TimeStampType> dts;
        std::vector<NodeIDType> nodes;

        void clear_and_reserve(size_t cap) {
            row.clear(); col.clear(); eid.clear();
            ts.clear(); dts.clear(); nodes.clear();
            row.reserve(cap); col.reserve(cap); eid.reserve(cap);
            ts.reserve(cap); dts.reserve(cap); nodes.reserve(cap);
        }
    };

    // 返回 [start, end]（都 inclusive），表示满足 (lower, upper) 的边索引范围。
    // 若为空，返回 start > end。
    inline std::pair<EdgeIDType, EdgeIDType> find_edge_range(NodeIDType n,
                                                            TimeStampType upper,
                                                            TimeStampType lower) const {
        const EdgeIDType b = indptr[n];
        const EdgeIDType e = indptr[n + 1];
        if (b >= e) return {1, 0};

        const auto seg_begin = edge_ts.begin() + b;
        const auto seg_end   = edge_ts.begin() + e;

        // start: first edge with ts > lower  (strict)
        auto it_start = (num_history == 1)
            ? seg_begin
            : std::upper_bound(seg_begin, seg_end, lower);

        // end: last edge with ts < upper  (strict) -> last <= upper - eps
        auto it_end_plus1 = std::upper_bound(seg_begin, seg_end, upper - kEps);

        EdgeIDType start = static_cast<EdgeIDType>(it_start - edge_ts.begin());
        EdgeIDType end   = static_cast<EdgeIDType>(it_end_plus1 - edge_ts.begin()) - 1;

        if (end < start) return {1, 0};
        return {start, end};
    }

    inline void add_neighbor(ThreadBuffer& buf,
                             NodeIDType dst_row,        // global dst index in this block
                             EdgeIDType edge_idx,       // global edge index in CSR arrays
                             TimeStampType dst_ts_raw)  // original dst timestamp (not offset)
    {
        buf.row.push_back(dst_row);
        buf.col.push_back(static_cast<NodeIDType>(buf.nodes.size())); // local src index
        buf.eid.push_back(edge_ids[edge_idx]);

        const TimeStampType e_ts = edge_ts[edge_idx];
        buf.ts.push_back(prop_time ? dst_ts_raw : e_ts);
        buf.dts.push_back(dst_ts_raw - e_ts);
        buf.nodes.push_back(indices[edge_idx]);
    }

    // very small fast RNG (xorshift32), enough for sampling
    static inline uint32_t xorshift32(uint32_t& x) {
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        return x;
    }

    void sample_one_block(TemporalGraphBlock& out,
                          const std::vector<NodeIDType>& dst_nodes,
                          const std::vector<TimeStampType>& dst_ts,
                          int neighs,
                          TimeStampType upper_offset)  // upper = dst_ts + upper_offset
    {
        const double t0 = omp_get_wtime();

        const size_t num_dst = dst_nodes.size();
        out = TemporalGraphBlock{};
        out.dim_out = static_cast<NodeIDType>(num_dst);

        // 先把 dst/root nodes 放进 block 的 node list（它们也是 src 的一部分）
        out.nodes = dst_nodes;
        out.ts = dst_ts;
        out.dts.assign(num_dst, 0.0f);

        // per-thread buffers
        std::vector<ThreadBuffer> bufs(num_threads);

        // 估一个 reserve：每个 dst 取 neighs 条边
        const size_t cap_each = (num_dst == 0) ? 0 : ( (num_dst + num_threads - 1) / num_threads ) * static_cast<size_t>(neighs);
        for (int t = 0; t < num_threads; ++t) bufs[t].clear_and_reserve(cap_each);

        double search_acc = 0.0;
        double sample_acc = 0.0;

#pragma omp parallel
        {
            const int tid = omp_get_thread_num();
            ThreadBuffer& buf = bufs[tid];

#pragma omp for schedule(static)
            for (long long j = 0; j < static_cast<long long>(num_dst); ++j) {
                const NodeIDType n = dst_nodes[static_cast<size_t>(j)];
                const TimeStampType base_ts = dst_ts[static_cast<size_t>(j)];

                // history>1: window = (upper-window_duration, upper)
                // history==1: window = (-inf, upper)  (TGAT-like)
                const TimeStampType upper = base_ts + upper_offset;
                const TimeStampType lower = (num_history == 1) ? -INFINITY : (upper - window_duration);

                const double ts_s = omp_get_wtime();
                const auto [start, end] = find_edge_range(n, upper, lower);
                search_acc += (omp_get_wtime() - ts_s);

                if (start > end) continue; // no eligible edges

                const EdgeIDType eligible = end - start + 1;

                const double ts_p = omp_get_wtime();
                if (recent || eligible <= neighs) {
                    // take most recent min(neighs, eligible)
                    const EdgeIDType take = static_cast<EdgeIDType>(std::min<EdgeIDType>(eligible, neighs));
                    EdgeIDType k = end;
                    EdgeIDType stop = end - take + 1;
                    for (; k >= stop; --k) {
                        add_neighbor(buf, static_cast<NodeIDType>(j), k, base_ts);
                        if (k == stop) break; // avoid underflow
                    }
                } else {
                    // random sampling WITH replacement (行为更接近你原版，但范围严格正确)
                    uint32_t seed = 2166136261u ^ (static_cast<uint32_t>(tid) * 16777619u) ^ static_cast<uint32_t>(j);
                    for (int s = 0; s < neighs; ++s) {
                        const uint32_t r = xorshift32(seed);
                        const EdgeIDType picked = start + static_cast<EdgeIDType>(r % static_cast<uint32_t>(eligible));
                        add_neighbor(buf, static_cast<NodeIDType>(j), picked, base_ts);
                    }
                }
                sample_acc += (omp_get_wtime() - ts_p);
            }
        } // omp parallel

        // prefix sum: each edge adds exactly one new src node (不去重，保持你原行为)
        std::vector<size_t> edge_prefix(num_threads + 1, 0);
        for (int t = 0; t < num_threads; ++t) {
            edge_prefix[t + 1] = edge_prefix[t] + bufs[t].row.size();
        }
        const size_t total_edges = edge_prefix.back();

        const double t_coo = omp_get_wtime();

        out.row.resize(total_edges);
        out.col.resize(total_edges);
        out.eid.resize(total_edges);

        out.ts.resize(num_dst + total_edges);
        out.dts.resize(num_dst + total_edges);
        out.nodes.resize(num_dst + total_edges);

#pragma omp parallel for schedule(static)
        for (int t = 0; t < num_threads; ++t) {
            const size_t eoff = edge_prefix[t];
            ThreadBuffer& buf = bufs[t];

            // edges
            for (size_t i = 0; i < buf.row.size(); ++i) {
                out.row[eoff + i] = buf.row[i]; // global dst row index
                out.col[eoff + i] = static_cast<NodeIDType>(num_dst + eoff + i); // src index in out.nodes
                out.eid[eoff + i] = buf.eid[i];
            }

            // appended src nodes (one per edge)
            for (size_t i = 0; i < buf.nodes.size(); ++i) {
                const size_t noff = num_dst + eoff + i;
                out.nodes[noff] = buf.nodes[i];
                out.ts[noff] = buf.ts[i];
                out.dts[noff] = buf.dts[i];
            }
        }

        out.dim_in = static_cast<NodeIDType>(out.nodes.size());

        out.search_time = search_acc;   // 粗略累计（跨线程累计会有误差，但量级对）
        out.sample_time = sample_acc;
        out.coo_time = omp_get_wtime() - t_coo;
        out.tot_time = omp_get_wtime() - t0;
    }

public:
    void sample(const std::vector<NodeIDType>& root_nodes,
                const std::vector<TimeStampType>& root_ts)
    {
        omp_set_num_threads(num_threads);

        ret.clear();
        ret.resize(static_cast<size_t>(num_layers * num_history));

        auto idx = [&](int layer, int h) -> size_t {
            // h: 0..num_history-1  (oldest -> newest)
            return static_cast<size_t>(layer * num_history + h);
        };

        for (int layer = 0; layer < num_layers; ++layer) {
            const int neighs = num_neighbors[layer];

            for (int h = 0; h < num_history; ++h) {
                // newest window has offset=0; older ones shift backward
                const int lag = (num_history - 1 - h);
                const TimeStampType upper_offset = (num_history == 1) ? 0.0f : (-static_cast<TimeStampType>(lag) * window_duration);

                const std::vector<NodeIDType>* in_nodes = &root_nodes;
                const std::vector<TimeStampType>* in_ts = &root_ts;

                if (layer > 0) {
                    TemporalGraphBlock& prev = ret[idx(layer - 1, h)];
                    in_nodes = &prev.nodes; // expand frontier using previous layer's src nodes
                    in_ts    = &prev.ts;
                }

                TemporalGraphBlock& out = ret[idx(layer, h)];
                sample_one_block(out, *in_nodes, *in_ts, neighs, upper_offset);
            }
        }
    }
};

PYBIND11_MODULE(sampler_core, m) {
    py::class_<TemporalGraphBlock>(m, "TemporalGraphBlock")
        .def(py::init<>())
        .def("row",   [](const TemporalGraphBlock& b){ return vec2npy(b.row); })
        .def("col",   [](const TemporalGraphBlock& b){ return vec2npy(b.col); })
        .def("eid",   [](const TemporalGraphBlock& b){ return vec2npy(b.eid); })
        .def("ts",    [](const TemporalGraphBlock& b){ return vec2npy(b.ts); })
        .def("dts",   [](const TemporalGraphBlock& b){ return vec2npy(b.dts); })
        .def("nodes", [](const TemporalGraphBlock& b){ return vec2npy(b.nodes); })
        .def("dim_in",  [](const TemporalGraphBlock& b){ return b.dim_in; })
        .def("dim_out", [](const TemporalGraphBlock& b){ return b.dim_out; })
        .def("tot_time",    [](const TemporalGraphBlock& b){ return b.tot_time; })
        .def("search_time", [](const TemporalGraphBlock& b){ return b.search_time; })
        .def("sample_time", [](const TemporalGraphBlock& b){ return b.sample_time; })
        .def("coo_time",    [](const TemporalGraphBlock& b){ return b.coo_time; });

    py::class_<ParallelSampler>(m, "ParallelSampler")
        .def(py::init<const std::vector<EdgeIDType>&,
                      const std::vector<NodeIDType>&,
                      const std::vector<EdgeIDType>&,
                      const std::vector<TimeStampType>&,
                      int, int, int,
                      const std::vector<int>&,
                      bool, bool,
                      int, TimeStampType>())
        .def("sample", &ParallelSampler::sample)
        .def("reset",  &ParallelSampler::reset)
        // 注意：这里返回会拷贝（pybind11/stl 默认行为）。要零拷贝需要 bind_vector + reference policy。
        .def("get_ret", [](const ParallelSampler& ps){ return ps.ret; });
}
