#pragma once

#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <tuple>
#include <vector>

#include <nanobind/ndarray.h>
#include <nanobind/stl/vector.h>

#include "sampler_common.h"

namespace planner_ownership {

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

}  // namespace planner_ownership
