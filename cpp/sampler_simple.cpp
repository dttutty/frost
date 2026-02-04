// sampler_simple.cpp
#include <algorithm>
#include <cstdint>
#include <cmath>
#include <random>
#include <vector>
#include <omp.h>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl.h>
#include "sampler_common.h" // 包含头文件
using nb::arg;
namespace nb = nanobind;




// --- 采样器核心 (Simple Version) ---
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

    int num_thread_per_worker = 1;
    int num_workers = 1;
    int num_threads = 1;

    int num_layers = 1;
    std::vector<int> num_neighbors;

    bool recent = false;     // true: greedy latest, false: random
    bool prop_time = false;  // true: use dst_ts, false: use edge_ts

    // Output buffer
    std::vector<TemporalGraphBlock> ret;

    ParallelSampler(const TemporalGraph& g,
                    int _num_thread_per_worker,
                    int _num_workers,
                    int _num_layers,
                    const std::vector<int>& _num_neighbors,
                    bool _recent,
                    bool _prop_time)
        : graph_ref(g), // 拷贝结构体 (只是拷贝了 nb::ndarray 的句柄，开销极小)
          indptr(g.indptr_ptr()),   // 提取指针
          indices(g.indices_ptr()), // 提取指针
          edge_ids(g.eid_ptr()),    // 提取指针
          edge_ts(g.ts_ptr()),      // 提取指针
          num_thread_per_worker(_num_thread_per_worker), num_workers(_num_workers),
          num_layers(_num_layers), num_neighbors(_num_neighbors),
          recent(_recent), prop_time(_prop_time) {
        
        num_threads = std::max(1, num_thread_per_worker * num_workers);
        omp_set_num_threads(num_threads);
        num_nodes = static_cast<NodeIDType>(indptr.size() - 1);
        num_edges = static_cast<EdgeIDType>(indices.size());
    }
private:
    // 线程局部缓存
    struct ThreadBuffer {
        std::vector<NodeIDType> row, col;
        std::vector<EdgeIDType> eid;
        std::vector<TimeStampType> ts, dts;
        std::vector<NodeIDType> nodes; // local src nodes
        
        // 预留空间辅助函数
        void reserve(size_t size) {
            row.reserve(size); col.reserve(size); eid.reserve(size);
            ts.reserve(size); dts.reserve(size); nodes.reserve(size);
        }
    };

    // 查找逻辑不变 (C++17 style)
    inline std::pair<EdgeIDType, EdgeIDType> find_edge_range(NodeIDType n, TimeStampType upper) const {
        const EdgeIDType b = indptr[n];
        const EdgeIDType e = indptr[n + 1];
        if (b >= e) return {1, 0};
        
        // 直接使用 upper_bound，逻辑清晰
        auto it = std::upper_bound(edge_ts.begin() + b, edge_ts.begin() + e, upper - kEps);
        return {b, static_cast<EdgeIDType>(it - edge_ts.begin()) - 1};
    }

    void sample_one_block(TemporalGraphBlock& out,
                          const std::vector<NodeIDType>& dst_nodes,
                          const std::vector<TimeStampType>& dst_ts,
                          int neighs) {
        tbb::tick_count t0 = tbb::tick_count::now(); // TBB 计时器
        
        const size_t num_dst = dst_nodes.size();
        out = TemporalGraphBlock{};
        out.dim_out = static_cast<NodeIDType>(num_dst);
        out.nodes = dst_nodes; // Root nodes
        out.ts = dst_ts;
        out.dts.assign(num_dst, 0.0f);

        // --- 核心变化 1: 自动线程局部存储 ---
        // TBB 会自动为每个线程创建 Buffer，无需手动管理 tid
        tbb::enumerable_thread_specific<ThreadBuffer> tls_buffers;

        // --- 核心变化 2: TBB 并行循环 ---
        tbb::parallel_for(tbb::blocked_range<size_t>(0, num_dst), 
            [&](const tbb::blocked_range<size_t>& r) {
                // 获取当前线程的 Buffer
                ThreadBuffer& buf = tls_buffers.local(); 
                
                // 简单的 RNG (线程私有)
                Xorshift32 rng(2166136261u ^ std::hash<std::thread::id>{}(std::this_thread::get_id()));

                for (size_t j = r.begin(); j != r.end(); ++j) {
                    const NodeIDType n = dst_nodes[j];
                    const TimeStampType base_ts = dst_ts[j];
                    
                    auto [start, end] = find_edge_range(n, base_ts);
                    if (start > end) continue;

                    const EdgeIDType eligible = end - start + 1;
                    
                    // 采样逻辑 (Greedy or Random)
                    // 注意：buf.col 存的是局部索引，合并时需要修正
                    auto add_edge = [&](EdgeIDType edge_idx) {
                        buf.row.push_back(static_cast<NodeIDType>(j)); // Global Dst Index
                        buf.col.push_back(static_cast<NodeIDType>(buf.nodes.size())); // Local Src Index (暂时的)
                        buf.eid.push_back(edge_ids[edge_idx]);
                        buf.nodes.push_back(indices[edge_idx]);
                        
                        TimeStampType e_ts = edge_ts[edge_idx];
                        buf.ts.push_back(prop_time ? base_ts : e_ts);
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

        // --- 核心变化 3: 优雅合并 (Merge) ---
        // 不再需要手写 Prefix Sum 循环，TBB 的数据结构支持迭代
        
        // 1. 计算总大小并预分配
        size_t total_edges = 0;
        for (const auto& buf : tls_buffers) total_edges += buf.row.size();
        
        out.row.reserve(total_edges); out.col.reserve(total_edges); out.eid.reserve(total_edges);
        // ... resize ...

        // 2. 合并数据
        // TBB 保证遍历顺序是确定性的吗？不一定。
        // 如果需要严格确定性，可以用 flatten。如果不在乎顺序，直接遍历：
        
        NodeIDType current_src_offset = static_cast<NodeIDType>(num_dst); // src node 从 num_dst 开始编号
        
        for (const auto& buf : tls_buffers) {
            // Copy rows/eids (简单拷贝)
            out.row.insert(out.row.end(), buf.row.begin(), buf.row.end());
            out.eid.insert(out.eid.end(), buf.eid.begin(), buf.eid.end());
            
            // Fix cols (局部索引 -> 全局索引)
            // buf.col 里面原本存的是 0, 1, 2... 这里加上 offset
            size_t start_idx = out.col.size();
            out.col.insert(out.col.end(), buf.col.begin(), buf.col.end());
            std::for_each(out.col.begin() + start_idx, out.col.end(), 
                          [current_src_offset](NodeIDType& val){ val += current_src_offset; });
            
            // Appending src nodes info
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
        omp_set_num_threads(num_threads);
        ret.clear();
        ret.resize(num_layers); // Only layers, no history

        const std::vector<NodeIDType>* in_nodes = &root_nodes;
        const std::vector<TimeStampType>* in_ts = &root_ts;

        for (int layer = 0; layer < num_layers; ++layer) {
            if (layer > 0) {
                in_nodes = &ret[layer - 1].nodes;
                in_ts    = &ret[layer - 1].ts;
            }
            sample_one_block(ret[layer], *in_nodes, *in_ts, num_neighbors[layer]);
        }
    }
    
    // reset interface kept
    void reset() {}
};

NB_MODULE(sampler_core_simple, m) {
    bind_common(m); // 先绑定 Graph // <--- 必须用 NB_MODULE，不能用 PYBIND11_MODULE
    bind_temporal_graph_block(m);

    nb::class_<ParallelSampler>(m, "ParallelSampler")
        // 构造函数：参数极其多，必须加 arg
        .def(nb::init<const TemporalGraph&,
                      int, int, int, const std::vector<int>&, 
                      bool, bool>(),
             nb::arg("graph"), // 名字叫 graph,
             arg("num_thread_per_worker"), 
             arg("num_workers"), 
             arg("num_layers"), 
             arg("num_neighbors"),
             arg("recent") = false,    // 支持默认值
             arg("prop_time") = false  // 支持默认值
        )
        // sample 方法：也建议加 arg
        .def("sample", &ParallelSampler::sample, 
             arg("root_nodes"), 
             arg("root_ts"))
        // get_ret：没有参数，不需要 arg
        .def("get_ret", [](const ParallelSampler& ps){ return ps.ret; });
}