// sampler_seq.cpp
#include "sampler_common.h"
#include <nanobind/nanobind.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/tuple.h>
#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <stdexcept>
#include <vector>
#include <random>
#include <string>
#include <unordered_map>
#include "sampler_base.h"
namespace nb = nanobind;



// 1. Continuous-Time Sequence Sampler (For DyGFormer, GraphMixer, TGN, CAWN)
template <typename ID_T, typename TIME_T>
class SeqSampler : public BaseSampler<ID_T, TIME_T> {
private:
    std::string sample_strategy;
    int seed_val;
    double time_scaling_factor;

    void sample_historical_neighbors_into_buffers(
        const ID_T* nid,
        const TIME_T* ts,
        size_t bs,
        int n_ngh,
        ID_T* out_n,
        ID_T* out_e,
        TIME_T* out_t) const
    {
        const auto is_env_truthy = [](const char* value) {
            if (value == nullptr) {
                return false;
            }
            std::string normalized(value);
            std::transform(normalized.begin(), normalized.end(), normalized.begin(),
                           [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
            return normalized == "1" || normalized == "true" || normalized == "yes" || normalized == "on";
        };
        const bool disable_omp = is_env_truthy(std::getenv("FROST_SEQSAMPLER_NO_OMP"));
        const bool mimic_tgl_batch_shared_recent =
            sample_strategy == "recent" &&
            is_env_truthy(std::getenv("FROST_DEBUG_TGL_BATCH_SHARED_RECENT"));

        std::unordered_map<ID_T, TIME_T> batch_max_ts;
        std::vector<TIME_T> batch_upper_ts;
        if (mimic_tgl_batch_shared_recent) {
            batch_max_ts.reserve(bs);
            for (size_t i = 0; i < bs; ++i) {
                auto it = batch_max_ts.find(nid[i]);
                if (it == batch_max_ts.end() || it->second < ts[i]) {
                    batch_max_ts[nid[i]] = ts[i];
                }
            }
            batch_upper_ts.resize(bs);
            for (size_t i = 0; i < bs; ++i) {
                batch_upper_ts[i] = batch_max_ts.at(nid[i]);
            }
        }

        const auto sample_one = [&](size_t i) {
            if (mimic_tgl_batch_shared_recent) {
                const TIME_T batch_upper = batch_upper_ts[i];
                auto [start, end] = this->find_edge_range_base(nid[i], batch_upper);
                if (start > end) {
                    return;
                }

                const size_t offset = i * n_ngh;
                const ID_T limit = ((end - start + 1) <= n_ngh)
                    ? start
                    : static_cast<ID_T>(end - n_ngh);
                size_t take = 0;
                for (ID_T k = end; k > limit; --k) {
                    if (this->edge_ts[k] < ts[i]) {
                        ++take;
                    }
                }
                if (take == 0) {
                    return;
                }

                const size_t pad_offset = n_ngh - take;
                size_t write_offset = take;
                for (ID_T k = end; k > limit; --k) {
                    if (this->edge_ts[k] < ts[i]) {
                        --write_offset;
                        const size_t out_idx = offset + pad_offset + write_offset;
                        out_n[out_idx] = this->indices[k];
                        out_e[out_idx] = this->edge_ids[k];
                        out_t[out_idx] = this->edge_ts[k];
                    }
                }
                return;
            }

            std::mt19937 local_rng(seed_val + i);
            auto [start, end] = this->find_edge_range_base(nid[i], ts[i]);
            const size_t offset = i * n_ngh;
            this->perform_sampling(
                start,
                end,
                n_ngh,
                sample_strategy,
                local_rng,
                time_scaling_factor,
                &out_n[offset],
                &out_e[offset],
                &out_t[offset]);
        };

        if (disable_omp) {
            for (size_t i = 0; i < bs; ++i) {
                sample_one(i);
            }
        } else {
            #pragma omp parallel for schedule(static)
            for (size_t i = 0; i < bs; ++i) {
                sample_one(i);
            }
        }
    }

public:
    SeqSampler(
        const TemporalGraph<ID_T, TIME_T>& g,
        const std::string& strategy = "recent",
        std::optional<int> seed = std::nullopt,
        double scaling_factor = 0.0
    )
        : BaseSampler<ID_T, TIME_T>(g),
          sample_strategy(strategy),
          time_scaling_factor(scaling_factor) {
        seed_val = seed.has_value() ? seed.value() : std::random_device{}();
    }

    void reset_random_state() { /* Random state is locally managed for thread safety */ }

    // Used uniquely by DyGFormer: returns variable-length lists. 
    // OpenMP is NOT used here because nb::list interacts with Python GIL.
    std::tuple<nb::list, nb::list, nb::list> get_all_first_hop_neighbors(
    const std::vector<ID_T>& nid,
    const std::vector<TIME_T>& ts,
    std::optional<int> max_neighbors = std::nullopt) 
    {
        nb::list res_nodes, res_edges, res_times;
        size_t bs = nid.size();

        for (size_t i = 0; i < bs; ++i) {
            auto [start, end] = this->find_edge_range_base(nid[i], ts[i]);
            
            ID_T valid_end = (start <= end) ? (end + 1) : start;
            ID_T actual_start = start;
            
            // Truncate to keep only the most recent neighbors
            if (max_neighbors.has_value() && (valid_end - start) > max_neighbors.value()) {
                actual_start = valid_end - max_neighbors.value();
            }
            
            // Extract data slice
            std::vector<ID_T> n_vec(this->indices + actual_start, this->indices + valid_end);
            std::vector<ID_T> e_vec(this->edge_ids + actual_start, this->edge_ids + valid_end);
            std::vector<TIME_T> t_vec(this->edge_ts + actual_start, this->edge_ts + valid_end);
            
            // Append to python lists
            res_nodes.append(vec2npy(n_vec));
            res_edges.append(vec2npy(e_vec));
            res_times.append(vec2npy(t_vec));
        }
        return {res_nodes, res_edges, res_times};
    }

    // Used by TGAT, GraphMixer, CAWN: returns fixed-length 2D dense arrays [Batch, Neighbors].
    // Memory is pre-allocated. OpenMP is fully utilized.
    std::tuple<nb::ndarray<nb::numpy, ID_T>, nb::ndarray<nb::numpy, ID_T>, nb::ndarray<nb::numpy, TIME_T>> 
    get_historical_neighbors(
        const std::vector<ID_T>& nid,
        const std::vector<TIME_T>& ts,
        int n_ngh = 20) 
    {
        if (nid.size() != ts.size()) {
            throw std::invalid_argument("get_historical_neighbors requires nid.size() == ts.size()");
        }
        if (n_ngh <= 0) {
            throw std::invalid_argument("get_historical_neighbors requires n_ngh > 0");
        }

        size_t bs = nid.size();
        std::vector<ID_T> out_n(bs * n_ngh, -1);
        std::vector<ID_T> out_e(bs * n_ngh, -1);
        std::vector<TIME_T> out_t(bs * n_ngh, 0.0f);

        {
            nb::gil_scoped_release release;
            sample_historical_neighbors_into_buffers(
                nid.data(), ts.data(), bs, n_ngh,
                out_n.data(), out_e.data(), out_t.data());
        } // GIL reacquired here before creating Python objects

        return { vec2npy_2d(out_n, bs, n_ngh), vec2npy_2d(out_e, bs, n_ngh), vec2npy_2d(out_t, bs, n_ngh) };
    }

    using ConstIdArray =
        nb::ndarray<const ID_T, nb::ndim<1>, nb::c_contig, nb::device::cpu>;
    using ConstTimeArray =
        nb::ndarray<const TIME_T, nb::ndim<1>, nb::c_contig, nb::device::cpu>;
    using IdOutputArray =
        nb::ndarray<ID_T, nb::ndim<2>, nb::c_contig, nb::device::cpu>;
    using TimeOutputArray =
        nb::ndarray<TIME_T, nb::ndim<2>, nb::c_contig, nb::device::cpu>;

    void get_historical_neighbors_into(
        ConstIdArray nid,
        ConstTimeArray ts,
        IdOutputArray out_n,
        IdOutputArray out_e,
        TimeOutputArray out_t)
    {
        const size_t bs = nid.shape(0);
        if (ts.shape(0) != bs) {
            throw std::invalid_argument(
                "get_historical_neighbors_into requires nid and ts to have equal lengths");
        }
        if (out_n.shape(0) != bs || out_e.shape(0) != bs || out_t.shape(0) != bs) {
            throw std::invalid_argument(
                "get_historical_neighbors_into output row counts must match the input length");
        }
        const size_t n_ngh = out_n.shape(1);
        if (n_ngh == 0 || out_e.shape(1) != n_ngh || out_t.shape(1) != n_ngh) {
            throw std::invalid_argument(
                "get_historical_neighbors_into outputs must have equal positive widths");
        }

        {
            nb::gil_scoped_release release;
            std::fill(out_n.data(), out_n.data() + bs * n_ngh, static_cast<ID_T>(-1));
            std::fill(out_e.data(), out_e.data() + bs * n_ngh, static_cast<ID_T>(-1));
            std::fill(out_t.data(), out_t.data() + bs * n_ngh, static_cast<TIME_T>(0));
            sample_historical_neighbors_into_buffers(
                nid.data(), ts.data(), bs, static_cast<int>(n_ngh),
                out_n.data(), out_e.data(), out_t.data());
        }
    }
};
