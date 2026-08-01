#pragma once

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <tuple>
#include <unordered_map>
#include <vector>

#include <nanobind/ndarray.h>
#include <nanobind/stl/vector.h>

#include "sampler_common.h"

namespace planner_ownership {

constexpr uint8_t kAllSampled = 1U << 0;
constexpr uint8_t kRoots = 1U << 1;
constexpr uint8_t kPositiveRoots = 1U << 2;
constexpr uint8_t kPositiveNeighbors = 1U << 3;
constexpr size_t kScopeCount = 4;

using Int64Array =
    nb::ndarray<int64_t, nb::ndim<1>, nb::c_contig, nb::device::cpu>;

struct Result {
    std::vector<std::vector<int64_t>> owned;
    std::vector<int64_t> all_nodes;
    std::vector<std::vector<int64_t>> routes;
};

inline Result compute(
    const std::vector<Int64Array>& updates,
    const std::vector<Int64Array>& needs,
    bool prefer_higher_rank) {
    const size_t n_ranks = updates.size();
    if (needs.size() != n_ranks) {
        throw std::invalid_argument("updates and needs must have equal rank counts");
    }
    if (n_ranks > static_cast<size_t>(std::numeric_limits<int16_t>::max())) {
        throw std::invalid_argument("planner ownership supports at most 32767 ranks");
    }

    Result result{
        std::vector<std::vector<int64_t>>(n_ranks),
        {},
        std::vector<std::vector<int64_t>>(n_ranks * n_ranks),
    };
    size_t n_updates = 0;
    int64_t max_node = -1;
    for (const auto& update : updates) {
        n_updates += update.size();
        if (update.size() > 0) {
            max_node = std::max(max_node, update.data()[update.size() - 1]);
        }
    }
    if (n_updates == 0) {
        return result;
    }

    constexpr int64_t dense_max_nodes = 4'000'000;
    constexpr size_t dense_max_sparsity = 16;
    const int64_t dense_span = max_node + 1;
    const bool use_dense =
        max_node >= 0 && dense_span <= dense_max_nodes &&
        static_cast<uint64_t>(dense_span) <=
            std::max<uint64_t>(65'536, n_updates * dense_max_sparsity);

    std::vector<int16_t> dense_owners;
    std::vector<int16_t> sparse_owners;
    if (use_dense) {
        dense_owners.assign(static_cast<size_t>(dense_span), -1);
        if (prefer_higher_rank) {
            for (size_t rank = 0; rank < n_ranks; ++rank) {
                const auto& update = updates[rank];
                for (size_t index = 0; index < update.size(); ++index) {
                    dense_owners[static_cast<size_t>(update.data()[index])] =
                        static_cast<int16_t>(rank);
                }
            }
        } else {
            for (size_t rank = n_ranks; rank-- > 0;) {
                const auto& update = updates[rank];
                for (size_t index = 0; index < update.size(); ++index) {
                    dense_owners[static_cast<size_t>(update.data()[index])] =
                        static_cast<int16_t>(rank);
                }
            }
        }
        result.all_nodes.reserve(n_updates);
        sparse_owners.reserve(n_updates);
        for (int64_t node = 0; node < dense_span; ++node) {
            const int16_t owner = dense_owners[static_cast<size_t>(node)];
            if (owner >= 0) {
                result.all_nodes.push_back(node);
                sparse_owners.push_back(owner);
                result.owned[static_cast<size_t>(owner)].push_back(node);
            }
        }
    } else {
        std::vector<size_t> positions(n_ranks, 0);
        result.all_nodes.reserve(n_updates);
        sparse_owners.reserve(n_updates);
        while (true) {
            int64_t next_node = std::numeric_limits<int64_t>::max();
            for (size_t rank = 0; rank < n_ranks; ++rank) {
                if (positions[rank] < updates[rank].size()) {
                    next_node = std::min(
                        next_node, updates[rank].data()[positions[rank]]);
                }
            }
            if (next_node == std::numeric_limits<int64_t>::max()) {
                break;
            }
            size_t owner = prefer_higher_rank ? 0 : n_ranks;
            for (size_t rank = 0; rank < n_ranks; ++rank) {
                const auto& update = updates[rank];
                if (positions[rank] < update.size() &&
                    update.data()[positions[rank]] == next_node) {
                    owner = prefer_higher_rank ? rank : std::min(owner, rank);
                    ++positions[rank];
                }
            }
            result.all_nodes.push_back(next_node);
            sparse_owners.push_back(static_cast<int16_t>(owner));
            result.owned[owner].push_back(next_node);
        }
    }

    for (size_t receiver = 0; receiver < n_ranks; ++receiver) {
        const auto& need = needs[receiver];
        for (size_t index = 0; index < need.size(); ++index) {
            const int64_t node = need.data()[index];
            int16_t owner = -1;
            if (use_dense) {
                if (node >= 0 && node < dense_span) {
                    owner = dense_owners[static_cast<size_t>(node)];
                }
            } else {
                const auto found = std::lower_bound(
                    result.all_nodes.begin(), result.all_nodes.end(), node);
                if (found != result.all_nodes.end() && *found == node) {
                    owner = sparse_owners[static_cast<size_t>(
                        found - result.all_nodes.begin())];
                }
            }
            if (owner >= 0) {
                result.routes[static_cast<size_t>(owner) * n_ranks + receiver]
                    .push_back(node);
            }
        }
    }
    return result;
}

inline nb::tuple plan_owned_routes(
    const std::vector<Int64Array>& updates,
    const std::vector<Int64Array>& needs,
    bool prefer_higher_rank = true) {
    Result result;
    {
        nb::gil_scoped_release release;
        result = compute(updates, needs, prefer_higher_rank);
    }

    const size_t n_ranks = updates.size();
    nb::list owned;
    for (auto& nodes : result.owned) {
        owned.append(vec2npy(nodes));
    }
    nb::list routes;
    for (size_t sender = 0; sender < n_ranks; ++sender) {
        nb::list row;
        for (size_t receiver = 0; receiver < n_ranks; ++receiver) {
            row.append(vec2npy(result.routes[sender * n_ranks + receiver]));
        }
        routes.append(row);
    }
    return nb::make_tuple(owned, vec2npy(result.all_nodes), routes);
}

template <typename ID_T>
using NodeArray =
    nb::ndarray<const ID_T, nb::ndim<1>, nb::c_contig, nb::device::cpu>;

template <typename ID_T>
inline nb::tuple materialize_node_scopes(
    NodeArray<ID_T> nid,
    size_t n_root,
    size_t n_positive_root,
    size_t positive_neighbor_start,
    size_t positive_neighbor_end,
    uint8_t requested_scopes) {
    if (n_positive_root > n_root || n_root > nid.size()) {
        throw std::invalid_argument("invalid root boundaries for node-scope materialization");
    }
    if (positive_neighbor_start > positive_neighbor_end ||
        positive_neighbor_end > nid.size()) {
        throw std::invalid_argument(
            "invalid positive-neighbor boundaries for node-scope materialization");
    }

    std::array<size_t, kScopeCount> raw_counts{};
    ID_T max_node = static_cast<ID_T>(-1);
    const auto occurrence_mask = [&](size_t index) {
        uint8_t mask = 0;
        if ((requested_scopes & kAllSampled) != 0) {
            mask |= kAllSampled;
        }
        if (index < n_root && (requested_scopes & kRoots) != 0) {
            mask |= kRoots;
        }
        if (index < n_positive_root && (requested_scopes & kPositiveRoots) != 0) {
            mask |= kPositiveRoots;
        }
        if (index >= positive_neighbor_start && index < positive_neighbor_end &&
            (requested_scopes & kPositiveNeighbors) != 0) {
            mask |= kPositiveNeighbors;
        }
        return mask;
    };

    {
        nb::gil_scoped_release release;
        for (size_t index = 0; index < nid.size(); ++index) {
            const ID_T node = nid.data()[index];
            if (node < 0) {
                continue;
            }
            const uint8_t mask = occurrence_mask(index);
            if ((mask & kAllSampled) != 0) {
                ++raw_counts[0];
            }
            if ((mask & kRoots) != 0) {
                ++raw_counts[1];
            }
            if ((mask & kPositiveRoots) != 0) {
                ++raw_counts[2];
            }
            if ((mask & kPositiveNeighbors) != 0) {
                ++raw_counts[3];
            }
            max_node = std::max(max_node, node);
        }
    }

    std::array<std::vector<ID_T>, kScopeCount> outputs;
    if (max_node >= 0 && requested_scopes != 0) {
        constexpr int64_t dense_max_nodes = 4'000'000;
        constexpr size_t dense_max_sparsity = 16;
        const int64_t dense_span = static_cast<int64_t>(max_node) + 1;
        const bool use_dense =
            dense_span <= dense_max_nodes &&
            static_cast<uint64_t>(dense_span) <=
                std::max<uint64_t>(65'536, nid.size() * dense_max_sparsity);

        if (use_dense) {
            std::vector<uint8_t> marks(static_cast<size_t>(dense_span), 0);
            {
                nb::gil_scoped_release release;
                for (size_t index = 0; index < nid.size(); ++index) {
                    const ID_T node = nid.data()[index];
                    if (node >= 0) {
                        marks[static_cast<size_t>(node)] |= occurrence_mask(index);
                    }
                }
            }
            for (int64_t node = 0; node < dense_span; ++node) {
                const uint8_t mask = marks[static_cast<size_t>(node)];
                for (size_t scope = 0; scope < kScopeCount; ++scope) {
                    if ((mask & (1U << scope)) != 0) {
                        outputs[scope].push_back(static_cast<ID_T>(node));
                    }
                }
            }
        } else {
            std::unordered_map<ID_T, uint8_t> marks;
            marks.reserve(std::min<size_t>(nid.size(), 1'000'000));
            {
                nb::gil_scoped_release release;
                for (size_t index = 0; index < nid.size(); ++index) {
                    const ID_T node = nid.data()[index];
                    if (node >= 0) {
                        marks[node] |= occurrence_mask(index);
                    }
                }
            }
            std::vector<ID_T> nodes;
            nodes.reserve(marks.size());
            for (const auto& [node, _mask] : marks) {
                nodes.push_back(node);
            }
            std::sort(nodes.begin(), nodes.end());
            for (const ID_T node : nodes) {
                const uint8_t mask = marks.at(node);
                for (size_t scope = 0; scope < kScopeCount; ++scope) {
                    if ((mask & (1U << scope)) != 0) {
                        outputs[scope].push_back(node);
                    }
                }
            }
        }
    }

    nb::list scope_arrays;
    for (auto& output : outputs) {
        scope_arrays.append(vec2npy(output));
    }
    std::vector<int64_t> counts(raw_counts.begin(), raw_counts.end());
    return nb::make_tuple(scope_arrays, vec2npy(counts));
}

}  // namespace planner_ownership
