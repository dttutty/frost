#pragma once
#include "sampler_common.h"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cctype>
#include <string>
#include <vector>


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
                          double time_scaling_factor,
                          ID_T* out_n, ID_T* out_e, TIME_T* out_t) const {
        
        size_t eligible = (start <= end) ? (end - start + 1) : 0;
        if (eligible == 0) return;

        if (strategy == "recent") {
            const auto is_env_truthy = [](const char* value) {
                if (value == nullptr) {
                    return false;
                }
                std::string normalized(value);
                std::transform(normalized.begin(), normalized.end(), normalized.begin(),
                               [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
                return normalized == "1" || normalized == "true" || normalized == "yes" || normalized == "on";
            };

            size_t take = std::min<size_t>(eligible, static_cast<size_t>(n_ngh));
            const bool mimic_tgl_recent_off_by_one = is_env_truthy(
                std::getenv("FROST_DEBUG_TGL_RECENT_OFF_BY_ONE")
            );
            if (mimic_tgl_recent_off_by_one && eligible <= static_cast<size_t>(n_ngh)) {
                if (eligible <= 1) {
                    return;
                }
                take = eligible - 1;
            }

            size_t copy_start = end - take + 1;
            // Calculate padding: if we take 5 neighbors but need 20, we start at index 15
            size_t pad_offset = n_ngh - take;
            
            for (size_t k = 0; k < take; ++k) {
                out_n[pad_offset + k] = indices[copy_start + k];
                out_e[pad_offset + k] = edge_ids[copy_start + k];
                out_t[pad_offset + k] = edge_ts[copy_start + k];
            }
        } 
        else if (strategy == "uniform") {
            const auto is_env_truthy = [](const char* value) {
                if (value == nullptr) {
                    return false;
                }
                std::string normalized(value);
                std::transform(normalized.begin(), normalized.end(), normalized.begin(),
                               [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
                return normalized == "1" || normalized == "true" || normalized == "yes" || normalized == "on";
            };
            const bool mimic_tgl_low_degree_recent = is_env_truthy(
                std::getenv("FROST_DEBUG_TGL_UNIFORM_LOW_DEGREE_RECENT")
            );

            // TGL's uniform sampler switches to its recent-neighbor branch
            // when the eligible history fits in the requested fanout.  That
            // branch has an upstream off-by-one and therefore returns the
            // newest ``eligible - 1`` edges, excluding the oldest one.  Keep
            // this quirk behind the explicit parity flag: normal FROST
            // uniform sampling remains sampling-with-replacement everywhere.
            if (mimic_tgl_low_degree_recent && eligible <= static_cast<size_t>(n_ngh)) {
                if (eligible <= 1) {
                    return;
                }
                const size_t take = eligible - 1;
                const size_t copy_start = end - take + 1;
                const size_t pad_offset = n_ngh - take;
                for (size_t k = 0; k < take; ++k) {
                    out_n[pad_offset + k] = indices[copy_start + k];
                    out_e[pad_offset + k] = edge_ids[copy_start + k];
                    out_t[pad_offset + k] = edge_ts[copy_start + k];
                }
                return;
            }

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
        else if (strategy == "time_interval_aware") {
            std::vector<double> logits(eligible, -1e10);
            const double latest_time = static_cast<double>(edge_ts[end]);
            double cumulative = 0.0;

            for (size_t rel_idx = 0; rel_idx < eligible; ++rel_idx) {
                const size_t edge_idx = static_cast<size_t>(start) + rel_idx;
                const double delta =
                    static_cast<double>(edge_ts[edge_idx]) - latest_time;
                const double exp_time =
                    std::exp(time_scaling_factor * delta);
                cumulative += exp_time;

                if (std::isfinite(exp_time) && std::isfinite(cumulative) && cumulative > 0.0) {
                    logits[rel_idx] = exp_time / cumulative;
                }
            }

            const double max_logit =
                *std::max_element(logits.begin(), logits.end());
            std::vector<double> weights(eligible, 1.0);
            double weight_sum = 0.0;

            for (size_t rel_idx = 0; rel_idx < eligible; ++rel_idx) {
                const double shifted = logits[rel_idx] - max_logit;
                const double weight = std::exp(shifted);
                weights[rel_idx] = std::isfinite(weight) ? weight : 0.0;
                weight_sum += weights[rel_idx];
            }
            if (!(weight_sum > 0.0)) {
                std::fill(weights.begin(), weights.end(), 1.0);
            }

            std::discrete_distribution<size_t> distribution(
                weights.begin(), weights.end()
            );
            std::vector<size_t> sampled_idx(n_ngh);
            for (int k = 0; k < n_ngh; ++k) {
                sampled_idx[k] = static_cast<size_t>(start) + distribution(rng);
            }

            std::sort(sampled_idx.begin(), sampled_idx.end(), [&](size_t a, size_t b) {
                return edge_ts[a] < edge_ts[b];
            });

            for (int k = 0; k < n_ngh; ++k) {
                out_n[k] = indices[sampled_idx[k]];
                out_e[k] = edge_ids[sampled_idx[k]];
                out_t[k] = edge_ts[sampled_idx[k]];
            }
        } else {
            throw std::invalid_argument("Unknown sampling strategy: " + strategy);
        }
    }

public:
    ID_T num_nodes;
    BaseSampler(const TemporalGraph<ID_T, TIME_T>& g) 
        : graph_ref(g), indptr(g.indptr_ptr()), indices(g.indices_ptr()),
          edge_ids(g.eid_ptr()), edge_ts(g.ts_ptr()) {
        // Assume indptr has length num_nodes + 1.
        num_nodes = (ID_T)g.indptr.shape(0) - 1;
    }

    inline std::pair<ID_T, ID_T> find_edge_range_base(ID_T n, TIME_T upper) const {
        // Defensive check: return an empty range for invalid node IDs.
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
