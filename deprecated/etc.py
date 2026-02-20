# --- 1. Batch Generation Logic (Producer's core task) ---
# This is the same generator function from our previous discussion.
import numba
import numpy as np


@numba.jit(nopython=True)
def adaptive_split_generator_numba(max_batch_size, data, max_node_id):
    """
    Generates batches of edges ensuring no two edges in a batch share a node.
    This function acts as a generator.
    """

    num_edges = len(data)
    if num_edges == 0:
        return

    is_grouped = np.zeros(num_edges, dtype=np.bool_)
    num_grouped_edges = 0
    search_start_idx = 0

    while num_grouped_edges < num_edges:
        current_batch_indices = np.full(max_batch_size, -1, dtype=np.int64)
        current_batch_len = 0

        seen_node_mask = np.zeros(max_node_id + 1, dtype=np.bool_)

        for i in range(search_start_idx, num_edges):
            if current_batch_len == max_batch_size:
                break

            if not is_grouped[i]:
                source_node, target_node = data[i]

                if not seen_node_mask[source_node] and not seen_node_mask[target_node]:
                    seen_node_mask[source_node] = True
                    seen_node_mask[target_node] = True

                    current_batch_indices[current_batch_len] = i
                    current_batch_len += 1

                    is_grouped[i] = True

        if current_batch_len > 0:
            if current_batch_len < 2:
                break
            num_grouped_edges += current_batch_len
            yield current_batch_indices[:current_batch_len]

        while search_start_idx < num_edges and is_grouped[search_start_idx]:
            search_start_idx += 1

        if search_start_idx == num_edges:
            break


# --- 2. The Producer Function (to be run in a separate thread) ---
def batch_producer(train_data, max_batch_size, batch_queue):
    """
    Continuously generates batches and puts them into a thread-safe queue.

    Args:
        data (np.ndarray): The full edge dataset.
        max_batch_size (int): The max size for each batch.
        batch_queue (queue.Queue): The thread-safe queue to put batches into.
    """
    logger.debug("\n[Producer] Starting to generate batches...")
    # Create a generator instance
    train_data = train_data.astype(np.int64)
    max_node_id = int(train_data.max())
    generator = adaptive_split_generator_numba(max_batch_size, train_data, max_node_id)

    # Iterate through the generator and put each batch into the queue
    for batch in generator:
        batch_queue.put(batch)
        # Optional: sleep if you want to simulate I/O or other work
        # time.sleep(0.1)

    # --- Sentinel Value ---
    # Put a special value (None) into the queue to signal that production is complete.
    logger.debug("\n[Producer] All batches generated. Signaling completion.")
    batch_queue.put(None)


def adaptive_split(threshold, data, group_index):
    start_list = []
    start = 0
    while start < len(data) - 1:
        count_nodes = 0
        count_edges = 0
        node_set = set()
        for i in range(start, len(data)):
            count_edges += 2
            for j in data[i]:
                if j not in node_set:
                    node_set.add(j)
                    count_nodes += 1
            if count_edges - count_nodes > threshold or i == len(data) - 1:
                end = i
                break
        start_list.append(end)
        start = end
    start_list = [0] + start_list
    start_list[-1] = len(data)
    # generate the split
    split = []
    for i in range(len(start_list)):
        if i != len(start_list) - 1:
            split.append(range(start_list[i], start_list[i + 1]))
    for i in range(len(split)):
        group_index[split[i]] = i

    return group_index


def adaptive_split_improved(max_batch_size, data, group_index):
    if not isinstance(data, np.ndarray) or data.ndim != 2:
        # Basic input validation
        raise TypeError("Input 'data' must be a 2D NumPy array.")
    num_edges = len(data)
    group_mark = [0] * num_edges  # 0 means not grouped yet
    current_group_id = 1  # assignment start from group 1
    num_grouped_edges = 0
    search_start_idx = 0

    while num_grouped_edges < num_edges:
        logger.debug("remaining_edges", num_edges - num_grouped_edges)

        current_group_size = 0
        seen_node_set = set()
        for i in range(search_start_idx, num_edges):
            if group_mark[i] == 0:
                if data[i][1] not in seen_node_set and data[i][0] not in seen_node_set:
                    seen_node_set.add(data[i][1])
                    seen_node_set.add(data[i][0])

                    group_mark[i] = current_group_id
                    current_group_size += 1
                    num_grouped_edges += 1

                    if current_group_size == max_batch_size:
                        break
        current_group_id += 1

        while search_start_idx < num_edges and group_mark[search_start_idx] != 0:
            search_start_idx += 1

    return group_mark


def get_staleness_constraint(data, batch_size):
    range_ = np.arange(0, len(data), batch_size)
    if range_[-1] != len(data):
        range_ = np.append(range_, [len(data)])
    # Calculate the staleness
    staleness = []
    for i in range(len(range_)):
        if i != len(range_) - 1:
            update = 2 * (range_[i + 1] - range_[i])
            nodes = len(np.unique(data[range_[i] : range_[i + 1]]))
            staleness.append(update - nodes)
    staleness = np.array(staleness)
    # return the upper bound for the staleness
    return max(staleness)
