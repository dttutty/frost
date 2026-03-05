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
#include "sampler_base.h"
namespace nb = nanobind;



// 1. Continuous-Time Sequence Sampler (For DyGFormer, GraphMixer, TGN, CAWN)
template <typename ID_T, typename TIME_T>
class SeqSampler : public BaseSampler<ID_T, TIME_T> {
private:
    std::string sample_strategy;
    int seed_val;

public:
    SeqSampler(const TemporalGraph<ID_T, TIME_T>& g, const std::string& strategy = "recent", std::optional<int> seed = std::nullopt)
        : BaseSampler<ID_T, TIME_T>(g), sample_strategy(strategy) {
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

        size_t bs = nid.size();
        std::vector<ID_T> out_n(bs * n_ngh, -1);
        std::vector<ID_T> out_e(bs * n_ngh, -1);
        std::vector<TIME_T> out_t(bs * n_ngh, 0.0f);

        {
            nb::gil_scoped_release release;

            if (disable_omp) {
                for (size_t i = 0; i < bs; ++i) {
                    std::mt19937 local_rng(seed_val + i);
                    auto [start, end] = this->find_edge_range_base(nid[i], ts[i]);

                    size_t offset = i * n_ngh;
                    this->perform_sampling(start, end, n_ngh, sample_strategy, local_rng,
                                    &out_n[offset], &out_e[offset], &out_t[offset]);
                }
            } else {
                #pragma omp parallel for schedule(static)
                for (size_t i = 0; i < bs; ++i) {
                    std::mt19937 local_rng(seed_val + i);
                    auto [start, end] = this->find_edge_range_base(nid[i], ts[i]);

                    size_t offset = i * n_ngh;
                    this->perform_sampling(start, end, n_ngh, sample_strategy, local_rng,
                                    &out_n[offset], &out_e[offset], &out_t[offset]);
                }
            }
        } // GIL reacquired here before creating Python objects

        return { vec2npy_2d(out_n, bs, n_ngh), vec2npy_2d(out_e, bs, n_ngh), vec2npy_2d(out_t, bs, n_ngh) };
    }
};

// 2. Discrete-Time Multi-Window Sequence Sampler (For DySAT)
template <typename ID_T, typename TIME_T>
class WindowSeqSampler : public BaseSampler<ID_T, TIME_T> {
private:
    int num_history;
    TIME_T window_duration;
    std::string sample_strategy;
    int seed_val;

public:
    WindowSeqSampler(const TemporalGraph<ID_T, TIME_T>& g, int _num_history, TIME_T _window_duration,
                     const std::string& strategy = "recent", std::optional<int> seed = std::nullopt)
        : BaseSampler<ID_T, TIME_T>(g), num_history(_num_history), window_duration(_window_duration), sample_strategy(strategy) {
        seed_val = seed.has_value() ? seed.value() : std::random_device{}();
    }

    // Returns fixed-length 3D dense arrays [Batch, History_Len, Neighbors]
    std::tuple<nb::ndarray<nb::numpy, ID_T>, nb::ndarray<nb::numpy, ID_T>, nb::ndarray<nb::numpy, TIME_T>> 
    sample(const std::vector<ID_T>& nid, const std::vector<TIME_T>& ts, int n_ngh = 20) 
    {
        if (nid.size() != ts.size()) {
            throw std::invalid_argument("WindowSeqSampler::sample requires nid.size() == ts.size()");
        }
        if (n_ngh <= 0) {
            throw std::invalid_argument("WindowSeqSampler::sample requires n_ngh > 0");
        }

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

        size_t bs = nid.size();
        size_t total_size = bs * num_history * n_ngh;
        
        std::vector<ID_T> out_n(total_size, -1);
        std::vector<ID_T> out_e(total_size, -1);
        std::vector<TIME_T> out_t(total_size, 0.0f);

        {
            nb::gil_scoped_release release;

            if (disable_omp) {
                for (size_t i = 0; i < bs; ++i) {
                    std::mt19937 local_rng(seed_val + i);
                    TIME_T base_ts = ts[i];

                    for (int h = 0; h < num_history; ++h) {
                        TIME_T upper = base_ts - h * window_duration;
                        TIME_T lower = base_ts - (h + 1) * window_duration;
                        auto [start, end] = this->find_edge_range_window(nid[i], upper, lower);

                        size_t offset = i * (num_history * n_ngh) + h * n_ngh;
                        this->perform_sampling(start, end, n_ngh, sample_strategy, local_rng,
                                        &out_n[offset], &out_e[offset], &out_t[offset]);
                    }
                }
            } else {
                #pragma omp parallel for schedule(static)
                for (size_t i = 0; i < bs; ++i) {
                    std::mt19937 local_rng(seed_val + i);
                    TIME_T base_ts = ts[i];

                    for (int h = 0; h < num_history; ++h) {
                        TIME_T upper = base_ts - h * window_duration;
                        TIME_T lower = base_ts - (h + 1) * window_duration;
                        auto [start, end] = this->find_edge_range_window(nid[i], upper, lower);

                        size_t offset = i * (num_history * n_ngh) + h * n_ngh;
                        this->perform_sampling(start, end, n_ngh, sample_strategy, local_rng,
                                        &out_n[offset], &out_e[offset], &out_t[offset]);
                    }
                }
            }
        } // GIL reacquired here before creating Python objects

        return { vec2npy_3d(out_n, bs, num_history, n_ngh), vec2npy_3d(out_e, bs, num_history, n_ngh), vec2npy_3d(out_t, bs, num_history, n_ngh) };
    }
};
