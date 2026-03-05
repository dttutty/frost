#pragma once
#include "sampler_common.h"
#include <algorithm>


// Base class containing common pointers and binary search logic
template <typename ID_T, typename TIME_T>
class BaseSampler {
protected:
    TemporalGraph<ID_T, TIME_T> graph_ref;
    const ID_T* indptr;
    const ID_T* indices;
    const ID_T* edge_ids;
    const TIME_T* edge_ts;
    
    template<typename RNG>
    void perform_sampling(ID_T start, ID_T end, int n_ngh,
                          const std::string& strategy, RNG& rng,
                          ID_T* out_n, ID_T* out_e, TIME_T* out_t) const {
        
        size_t eligible = (start <= end) ? (end - start + 1) : 0;
        if (eligible == 0) return;

        if (strategy == "recent") {
            size_t take = std::min<size_t>(eligible, static_cast<size_t>(n_ngh));
            size_t copy_start = end - take + 1;
            // Calculate padding: if we take 5 neighbors but need 20, we start at index 15
            size_t pad_offset = n_ngh - take;
            
            for (size_t k = 0; k < take; ++k) {
                out_n[pad_offset + k] = indices[copy_start + k];
                out_e[pad_offset + k] = edge_ids[copy_start + k];
                out_t[pad_offset + k] = edge_ts[copy_start + k];
            }
        } 
        else { // uniform
            std::vector<size_t> sampled_idx(n_ngh);
            for (int k = 0; k < n_ngh; ++k) {
                sampled_idx[k] = start + (rng() % eligible);
            }
            // Sort by time to maintain temporal order
            std::sort(sampled_idx.begin(), sampled_idx.end(), [&](size_t a, size_t b) {
                return edge_ts[a] < edge_ts[b];
            });

            for (int k = 0; k < n_ngh; ++k) {
                out_n[k] = indices[sampled_idx[k]];
                out_e[k] = edge_ids[sampled_idx[k]];
                out_t[k] = edge_ts[sampled_idx[k]];
            }
        }
    }

public:
    ID_T num_nodes;
    BaseSampler(const TemporalGraph<ID_T, TIME_T>& g) 
        : graph_ref(g), indptr(g.indptr_ptr()), indices(g.indices_ptr()),
          edge_ids(g.eid_ptr()), edge_ts(g.ts_ptr()) {
        // 假设 indptr 的长度是 num_nodes + 1
        num_nodes = (ID_T)g.indptr.shape(0) - 1;
    }

    inline std::pair<ID_T, ID_T> find_edge_range_base(ID_T n, TIME_T upper) const {
        // 防御性检查：如果 node ID 非法，返回一个空区间
        if (n < 0 || n >= num_nodes) return {1, 0};
        
        const ID_T b = indptr[n];
        const ID_T e = indptr[n + 1];
        if (b >= e) return {1, 0};
        
        auto it = std::lower_bound(edge_ts + b, edge_ts + e, upper - kEps<TIME_T>);
        return {b, static_cast<ID_T>(it - edge_ts) - 1};
    }

    virtual ~BaseSampler() = default;

protected:


    // Binary search for discrete-time windows (upper and lower bounds)
    inline std::pair<ID_T, ID_T> find_edge_range_window(ID_T n, TIME_T upper, TIME_T lower) const {
        if (n < 0 || n >= num_nodes) return {1, 0};
        const ID_T b = indptr[n];
        const ID_T e = indptr[n + 1];
        if (b >= e) return {1, 0};

        auto seg_begin = edge_ts + b;
        auto seg_end   = edge_ts + e;

        auto it_start = std::upper_bound(seg_begin, seg_end, lower);
        auto it_end_plus1 = std::lower_bound(seg_begin, seg_end, upper - kEps<TIME_T>);

        ID_T start = static_cast<ID_T>(it_start - edge_ts);
        ID_T end   = static_cast<ID_T>(it_end_plus1 - edge_ts) - 1;

        if (end < start) return {1, 0};
        return {start, end};
    }
};