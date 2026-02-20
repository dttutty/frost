// sampler_window.cpp
#include "sampler_common.h"
#include <algorithm>
#include <cmath>
#include <tbb/tbb.h> 
using nb::arg;
namespace nb = nanobind;

class ParallelSampler {
public:
    // CSR Data



    const int64_t* indptr;
    const int64_t* indices;
    const int64_t* edge_ids;
    const float* edge_ts;
    TemporalGraph graph_ref;

    



    NodeIDType num_nodes = 0;
    EdgeIDType num_edges = 0;

    // Concurrency settings (TBB manages threads automatically, but we keep params for API compatibility)
    int num_thread_per_worker = 1; 
    int num_workers = 1; 
    int num_threads = 1;

    int num_layers = 1;
    std::vector<int> num_neighbors;

    bool recent = false;
    bool prop_time = false;

    // Window specific
    int num_history = 1;
    TimeStampType window_duration = 0;

    std::vector<TemporalGraphBlock> ret;

    ParallelSampler(const TemporalGraph& g,
                    int _num_thread_per_worker, int _num_workers,
                    int _num_layers, const std::vector<int>& _num_neighbors,
                    bool _recent, bool _prop_time,
                    int _num_history, TimeStampType _window_duration)
        : graph_ref(g), // 拷贝结构体 (只是拷贝了 nb::ndarray 的句柄，开销极小)
          indptr(g.indptr_ptr()),   // 提取指针
          indices(g.indices_ptr()), // 提取指针
          edge_ids(g.eid_ptr()),    // 提取指针
          edge_ts(g.ts_ptr()),      // 提取指针
          num_thread_per_worker(_num_thread_per_worker), num_workers(_num_workers),
          num_layers(_num_layers), num_neighbors(_num_neighbors),
          recent(_recent), prop_time(_prop_time),
          num_history(_num_history), window_duration(_window_duration) {
        
        // TBB automatically initializes the thread pool. 
        // We calculate this just for info or if we need to limit concurrency manually.
        num_threads = std::max(1, num_thread_per_worker * num_workers);
        num_nodes = static_cast<NodeIDType>(indptr.size() - 1);
        num_edges = static_cast<EdgeIDType>(indices.size());
    }

private:
    struct ThreadBuffer {
        std::vector<NodeIDType> row, col;
        std::vector<EdgeIDType> eid;
        std::vector<TimeStampType> ts, dts;
        std::vector<NodeIDType> nodes; 
    };

    // Window version: Needs explicit lower bound check
    inline std::pair<EdgeIDType, EdgeIDType> find_edge_range(NodeIDType n, TimeStampType upper, TimeStampType lower) const {
        const EdgeIDType b = indptr[n];
        const EdgeIDType e = indptr[n + 1];
        if (b >= e) return {1, 0};

        auto seg_begin = edge_ts.begin() + b;
        auto seg_end   = edge_ts.begin() + e;

        // lower bound: find first edge > lower
        auto it_start = std::upper_bound(seg_begin, seg_end, lower);
        // upper bound: find last edge < upper
        auto it_end_plus1 = std::upper_bound(seg_begin, seg_end, upper - kEps);

        EdgeIDType start = static_cast<EdgeIDType>(it_start - edge_ts.begin());
        EdgeIDType end   = static_cast<EdgeIDType>(it_end_plus1 - edge_ts.begin()) - 1;

        if (end < start) return {1, 0};
        return {start, end};
    }

    void sample_one_block(TemporalGraphBlock& out,
                          const std::vector<NodeIDType>& dst_nodes,
                          const std::vector<TimeStampType>& dst_ts,
                          int neighs,
                          TimeStampType upper_offset) {
        tbb::tick_count t0 = tbb::tick_count::now();

        const size_t num_dst = dst_nodes.size();
        out = TemporalGraphBlock{};
        out.dim_out = static_cast<NodeIDType>(num_dst);
        out.nodes = dst_nodes;
        out.ts = dst_ts;
        out.dts.assign(num_dst, 0.0f);

        // TLS Buffer
        tbb::enumerable_thread_specific<ThreadBuffer> tls_buffers;

        // Parallel Loop
        tbb::parallel_for(tbb::blocked_range<size_t>(0, num_dst), 
            [&](const tbb::blocked_range<size_t>& r) {
                ThreadBuffer& buf = tls_buffers.local();
                // Thread-local RNG
                Xorshift32 rng(2166136261u ^ std::hash<std::thread::id>{}(std::this_thread::get_id()));

                for (size_t j = r.begin(); j != r.end(); ++j) {
                    const NodeIDType n = dst_nodes[j];
                    const TimeStampType base_ts = dst_ts[j];
                    
                    // Window Logic
                    const TimeStampType upper = base_ts + upper_offset;
                    const TimeStampType lower = upper - window_duration;

                    auto [start, end] = find_edge_range(n, upper, lower);
                    if (start > end) continue;

                    const EdgeIDType eligible = end - start + 1;
                    
                    auto add_edge = [&](EdgeIDType edge_idx) {
                        buf.row.push_back(static_cast<NodeIDType>(j)); // Global Dst
                        buf.col.push_back(static_cast<NodeIDType>(buf.nodes.size())); // Local Src
                        buf.eid.push_back(edge_ids[edge_idx]);
                        buf.nodes.push_back(indices[edge_idx]);
                        
                        TimeStampType e_ts = edge_ts[edge_idx];
                        buf.ts.push_back(prop_time ? base_ts : e_ts); // Propagated time or Edge time
                        buf.dts.push_back(base_ts - e_ts);
                    };

                    if (recent || eligible <= neighs) {
                        EdgeIDType take = std::min<EdgeIDType>(eligible, neighs);
                        for (EdgeIDType k = end; k > end - take; --k) add_edge(k);
                    } else {
                        for (int s = 0; s < neighs; ++s) {
                            add_edge(start + (rng() % eligible));
                        }
                    }
                }
            }
        );

        // Merge Logic (Same as Simple version)
        size_t total_edges = 0;
        for (const auto& buf : tls_buffers) total_edges += buf.row.size();
        
        out.row.reserve(total_edges); out.col.reserve(total_edges); out.eid.reserve(total_edges);
        // ... (resize others implicitly via insert)

        NodeIDType current_src_offset = static_cast<NodeIDType>(num_dst);
        
        for (const auto& buf : tls_buffers) {
            out.row.insert(out.row.end(), buf.row.begin(), buf.row.end());
            out.eid.insert(out.eid.end(), buf.eid.begin(), buf.eid.end());
            
            // Fix col indices
            size_t start_idx = out.col.size();
            out.col.insert(out.col.end(), buf.col.begin(), buf.col.end());
            std::for_each(out.col.begin() + start_idx, out.col.end(), 
                          [current_src_offset](NodeIDType& val){ val += current_src_offset; });
            
            out.nodes.insert(out.nodes.end(), buf.nodes.begin(), buf.nodes.end());
            out.ts.insert(out.ts.end(), buf.ts.begin(), buf.ts.end());
            out.dts.insert(out.dts.end(), buf.dts.begin(), buf.dts.end());

            current_src_offset += static_cast<NodeIDType>(buf.nodes.size());
        }
        
        out.dim_in = static_cast<NodeIDType>(out.nodes.size());
        out.tot_time = (tbb::tick_count::now() - t0).seconds();
    }

public:
    void sample(const std::vector<NodeIDType>& root_nodes, const std::vector<TimeStampType>& root_ts) {
        // No explicit omp_set_num_threads needed for TBB
        ret.clear();
        ret.resize(num_layers * num_history);

        auto idx = [&](int layer, int h) { return layer * num_history + h; };

        for (int layer = 0; layer < num_layers; ++layer) {
            for (int h = 0; h < num_history; ++h) {
                const int lag = (num_history - 1 - h);
                const TimeStampType upper_offset = -static_cast<TimeStampType>(lag) * window_duration;

                const std::vector<NodeIDType>* in_nodes = &root_nodes;
                const std::vector<TimeStampType>* in_ts = &root_ts;

                if (layer > 0) {
                    TemporalGraphBlock& prev = ret[idx(layer - 1, h)];
                    in_nodes = &prev.nodes;
                    in_ts    = &prev.ts;
                }
                sample_one_block(ret[idx(layer, h)], *in_nodes, *in_ts, num_neighbors[layer], upper_offset);
            }
        }
    }
};

NB_MODULE(sampler_core_window, m) {
    bind_common(m); // 先绑定 Graph
    bind_temporal_graph_block(m); // Reuse common binding

    nb::class_<ParallelSampler>(m, "ParallelSampler")
        // 修改这里：为每个参数添加 nb::arg("参数名")
        .def(nb::init<const TemporalGraph&,
                      int, int, int, const std::vector<int>&, 
                      bool, bool, int, TimeStampType>(),
             // 显式绑定参数名，这样 Python 才能识别关键字
             nb::arg("graph"), // 名字叫 graph,
             arg("num_thread_per_worker"), arg("num_workers"), 
             arg("num_layers"), arg("num_neighbors"),
             arg("recent") = false,      // 还可以设置默认值！
             arg("prop_time") = false,   // 设置默认值
             arg("num_history") = 1,     // 设置默认值
             arg("window_duration") = 0.0f
        )
        .def("sample", &ParallelSampler::sample, 
             arg("root_nodes"), arg("root_ts")) // sample 函数也支持关键字
        .def("get_ret", [](const ParallelSampler& ps){ return ps.ret; });
}