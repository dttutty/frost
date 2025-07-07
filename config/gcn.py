import torch
import torch.nn as nn
import numpy as np
import math
import time
from torch.utils.data import DataLoader, Dataset # For training loop example

# ============== 1. Helper Modules ==============

class TimeEncoder(nn.Module):
    """
    Sinusoidal Time Encoder from the TGN paper.
    Input shape: (batch_size, 1) tensor of time differences
    Output shape: (batch_size, time_dim)
    """
    def __init__(self, time_dim):
        super().__init__()
        self.time_dim = time_dim
        # Basis frequency coefficient (adjust if needed)
        self.basis_freq = nn.Parameter(torch.from_numpy(1 / 10 ** np.linspace(0, 9, time_dim)).float())
        self.phase = nn.Parameter(torch.zeros(time_dim).float())

    def forward(self, ts_diff):
        # ts_diff shape: [batch_size, 1]
        batch_size = ts_diff.size(0)
        # Ensure ts_diff is float and on the correct device
        ts_diff = ts_diff.float().to(self.basis_freq.device)

        map_ts = ts_diff * self.basis_freq.view(1, -1) + self.phase.view(1, -1) # [batch_size, time_dim]
        harmonic = torch.cos(map_ts)
        return harmonic

class MessageFunction(nn.Module):
    """
    Computes messages. Can be MLP, Identity, etc.
    Input: source_memory, dest_memory, time_encoding, edge_features
    Output: message vector
    """
    def __init__(self, memory_dim, time_dim, edge_feat_dim, message_dim):
        super().__init__()
        input_dim = 2 * memory_dim + time_dim + edge_feat_dim
        self.layer = nn.Sequential(
            nn.Linear(input_dim, message_dim),
            nn.ReLU()
            # Optionally add more layers
        )

    def forward(self, src_mem, dst_mem, time_enc, edge_feat):
        input_features = torch.cat([src_mem, dst_mem, time_enc, edge_feat], dim=1)
        message = self.layer(input_features)
        return message

class MessageAggregator(nn.Module):
    """
    Aggregates messages for a node. 'mean' or 'last'.
    Input: node_ids (for mapping), messages (associated with node_ids)
    Output: aggregated_messages (one per unique node_id)
    """
    def __init__(self, agg_method='mean'):
        super().__init__()
        if agg_method not in ['mean', 'last']:
            raise ValueError("Aggregation method must be 'mean' or 'last'")
        self.agg_method = agg_method

    def forward(self, node_ids, messages):
        """
        Aggregates messages for the same node_id.
        Assumes messages for the *same* interaction event (e.g., u->v and v->u)
        are handled appropriately *before* calling this if only one message per
        node update is desired (often the case). This function aggregates
        messages potentially arriving *between* memory updates.

        Args:
            node_ids (Tensor): Shape [num_messages], IDs of nodes receiving messages.
            messages (Tensor): Shape [num_messages, message_dim].

        Returns:
            unique_node_ids (Tensor): Unique node IDs that received messages.
            agg_messages (Tensor): Aggregated message for each unique node ID.
                                    Shape [num_unique_nodes, message_dim].
        """
        unique_node_ids = torch.unique(node_ids)
        agg_messages = []

        # Ensure device consistency
        device = messages.device
        unique_node_ids = unique_node_ids.to(device)

        for nid in unique_node_ids:
            node_messages = messages[node_ids == nid]
            if self.agg_method == 'mean':
                agg_messages.append(node_messages.mean(dim=0))
            elif self.agg_method == 'last':
                # This assumes messages are chronologically ordered *for each node*
                # In many batch implementations, 'mean' is easier.
                agg_messages.append(node_messages[-1])

        if not agg_messages:
             return unique_node_ids, torch.empty((0, messages.shape[1]), device=device) # Handle case with no messages

        agg_messages = torch.stack(agg_messages)
        return unique_node_ids, agg_messages


class MemoryUpdater(nn.Module):
    """
    Updates node memory using aggregated messages. GRU or RNN.
    Input: aggregated_message, previous_memory
    Output: new_memory
    """
    def __init__(self, memory_dim, message_dim, update_type='gru'):
        super().__init__()
        if update_type == 'gru':
            self.updater = nn.GRUCell(message_dim, memory_dim)
        elif update_type == 'rnn':
            self.updater = nn.RNNCell(message_dim, memory_dim)
        else:
            raise ValueError("Memory update type must be 'gru' or 'rnn'")

    def forward(self, agg_message, prev_memory):
        # GRUCell/RNNCell expect input [batch_size, input_size] and hidden [batch_size, hidden_size]
        # Here, batch_size corresponds to the number of nodes being updated.
        new_memory = self.updater(agg_message, prev_memory)
        return new_memory

class EmbeddingModule(nn.Module):
    """
    Computes temporal node embeddings using memory and recent messages/neighbors.
    This is a simplified version using just the memory.
    A full implementation might use graph attention over recent interactions.
    Input: node_ids, node_memory (at the time of embedding computation)
    Output: node_embeddings
    """
    def __init__(self, memory_dim, embedding_dim):
        super().__init__()
        # Simple projection from memory to embedding space
        # More complex versions (like Graph Attention) could be used here
        self.embedding_layer = nn.Linear(memory_dim, embedding_dim)

    def forward(self, node_memory):
        # node_memory shape: [num_nodes_to_embed, memory_dim]
        node_embeddings = self.embedding_layer(node_memory)
        return node_embeddings


# ============== 2. TGN Model ==============

class TGN(nn.Module):
    def __init__(self, num_nodes, edge_feat_dim,
                 memory_dim, time_dim, embedding_dim, message_dim,
                 agg_method='mean', mem_update_type='gru', dropout=0.1):
        super().__init__()

        self.num_nodes = num_nodes
        self.edge_feat_dim = edge_feat_dim
        self.memory_dim = memory_dim
        self.time_dim = time_dim
        self.embedding_dim = embedding_dim
        self.message_dim = message_dim

        # --- State (managed outside forward, updated by methods) ---
        # Node memory vectors
        self.memory = nn.Parameter(torch.zeros((num_nodes, memory_dim)), requires_grad=False)
        # Last interaction timestamp for each node
        self.last_update_ts = nn.Parameter(torch.zeros(num_nodes), requires_grad=False)

        # --- Modules ---
        self.time_encoder = TimeEncoder(time_dim)
        self.message_fn = MessageFunction(memory_dim, time_dim, edge_feat_dim, message_dim)
        self.aggregator = MessageAggregator(agg_method)
        self.memory_updater = MemoryUpdater(memory_dim, message_dim, mem_update_type)
        self.embedding_module = EmbeddingModule(memory_dim, embedding_dim)

        self.dropout = nn.Dropout(dropout)

    def compute_embedding(self, node_ids):
        """
        Computes embeddings for given nodes *at the current memory state*.
        Args:
            node_ids (Tensor): LongTensor of node IDs.
        Returns:
            embeddings (Tensor): Computed embeddings. Shape [len(node_ids), embedding_dim].
        """
        node_ids = node_ids.long() # Ensure correct type
        # Fetch current memory for the nodes
        # Use clone().detach() if memory is modified elsewhere and you need a snapshot
        node_memory = self.memory[node_ids]
        # Compute embeddings using the embedding module
        embeddings = self.embedding_module(node_memory)
        return embeddings

    def update_state(self, src_nodes, dst_nodes, timestamps, edge_features):
        """
        Updates the memory and last_update_ts based on a batch of interactions.
        This function implements the core TGN logic: compute messages, aggregate, update memory.

        Args:
            src_nodes (Tensor): Source node IDs of interactions.
            dst_nodes (Tensor): Destination node IDs of interactions.
            timestamps (Tensor): Timestamps of interactions.
            edge_features (Tensor): Features of interactions. Shape [batch_size, edge_feat_dim].
        """
        batch_size = len(src_nodes)
        device = src_nodes.device # Get device from input tensors

        # Ensure inputs are LongTensors where needed
        src_nodes = src_nodes.long()
        dst_nodes = dst_nodes.long()

        # --- Get previous states ---
        # Clone memory to avoid modifying it while fetching for messages
        memory_snapshot = self.memory.clone()
        prev_src_mem = memory_snapshot[src_nodes]
        prev_dst_mem = memory_snapshot[dst_nodes]

        prev_src_ts = self.last_update_ts[src_nodes]
        prev_dst_ts = self.last_update_ts[dst_nodes]

        # --- Compute time differences and encodings ---
        # Timestamps tensor needs to be [batch_size, 1] for TimeEncoder
        ts_tensor = timestamps.float().view(-1, 1).to(device)
        time_diff_src = ts_tensor - prev_src_ts.view(-1, 1)
        time_diff_dst = ts_tensor - prev_dst_ts.view(-1, 1)

        time_enc_src = self.time_encoder(time_diff_src)
        time_enc_dst = self.time_encoder(time_diff_dst)

        # --- Compute messages ---
        # Message from source perspective (for destination memory update)
        msg_for_dst = self.message_fn(prev_src_mem, prev_dst_mem, time_enc_dst, edge_features)
        # Message from destination perspective (for source memory update)
        msg_for_src = self.message_fn(prev_dst_mem, prev_src_mem, time_enc_src, edge_features)

        # Apply dropout to messages
        msg_for_dst = self.dropout(msg_for_dst)
        msg_for_src = self.dropout(msg_for_src)

        # --- Aggregate messages per node ---
        # Combine all nodes involved and corresponding messages
        all_nodes = torch.cat([src_nodes, dst_nodes])
        all_messages = torch.cat([msg_for_src, msg_for_dst])

        unique_update_nodes, aggregated_messages = self.aggregator(all_nodes, all_messages)

        if unique_update_nodes.nelement() == 0: # Handle case with no messages/updates
             print("Warning: No nodes to update.")
             return # Nothing to update

        # --- Update memory for nodes that received messages ---
        # Fetch previous memory for the unique nodes needing updates
        prev_mem_to_update = memory_snapshot[unique_update_nodes]

        # Update memory using the updater module
        new_node_memory = self.memory_updater(aggregated_messages, prev_mem_to_update)

        # Apply dropout to memory update
        new_node_memory = self.dropout(new_node_memory)

        # --- Store updated memory and timestamps ---
        # Use scatter_ to update only the relevant rows in self.memory
        # Make sure gradients are not tracked for direct memory modification
        with torch.no_grad():
            self.memory[unique_update_nodes] = new_node_memory

            # Find the latest timestamp for each node involved in this update batch
            # This is a simplification; assumes batch timestamps can update last_update_ts
            # A more precise way might involve iterating or using scatter_max
            max_ts_map = {}
            all_involved_nodes_unique = torch.unique(all_nodes)
            for nid in all_involved_nodes_unique.cpu().numpy(): # Iterate on CPU for efficiency if many nodes
                node_timestamps = timestamps[(src_nodes == nid) | (dst_nodes == nid)]
                if len(node_timestamps) > 0:
                    max_ts_map[nid] = torch.max(node_timestamps)

            for nid, max_ts in max_ts_map.items():
                 # Update only if the current batch time is later
                 if max_ts > self.last_update_ts[nid]:
                      self.last_update_ts[nid] = max_ts.to(device)


    def detach_memory(self):
        """Detaches the memory and last_update_ts from the computation graph."""
        self.memory.detach_()
        self.last_update_ts.detach_()

    def reset_state(self):
        """Resets memory and last update times to zero."""
        with torch.no_grad():
            self.memory.fill_(0.)
            self.last_update_ts.fill_(0.)

# ============== 3. Conceptual Training Loop ==============

# --- Dummy Data Generation ---
# Replace with your actual data loading
class DummyTemporalDataset(Dataset):
    def __init__(self, num_interactions=1000, num_nodes=100, edge_feat_dim=16):
        self.num_interactions = num_interactions
        self.num_nodes = num_nodes
        self.edge_feat_dim = edge_feat_dim

        # Generate random interactions - MUST BE SORTED BY TIME for TGN
        self.src = np.random.randint(0, num_nodes, num_interactions)
        self.dst = np.random.randint(0, num_nodes, num_interactions)
        # Ensure src != dst for simplicity here
        for i in range(num_interactions):
            while self.dst[i] == self.src[i]:
                self.dst[i] = np.random.randint(0, num_nodes)

        # Generate increasing timestamps
        self.ts = np.sort(np.random.rand(num_interactions) * num_interactions).astype(np.float32)
        self.edge_feats = np.random.randn(num_interactions, edge_feat_dim).astype(np.float32)
        # Labels (e.g., 1 for existing edge) - needed for loss
        self.labels = np.ones(num_interactions).astype(np.float32)

    def __len__(self):
        return self.num_interactions

    def __getitem__(self, idx):
        return self.src[idx], self.dst[idx], self.ts[idx], self.edge_feats[idx], self.labels[idx]

def collate_fn(batch):
    src, dst, ts, edge_feats, labels = zip(*batch)
    return (torch.tensor(src, dtype=torch.long),
            torch.tensor(dst, dtype=torch.long),
            torch.tensor(ts, dtype=torch.float),
            torch.from_numpy(np.stack(edge_feats, axis=0)),
            torch.tensor(labels, dtype=torch.float))

# --- Training Setup ---
NUM_NODES = 100
EDGE_FEAT_DIM = 16
MEMORY_DIM = 32
TIME_DIM = 16
EMBEDDING_DIM = 32
MESSAGE_DIM = 32
BATCH_SIZE = 64
LEARNING_RATE = 0.0003
EPOCHS = 5
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- Instantiate Model, Optimizer, Loss ---
model = TGN(
    num_nodes=NUM_NODES,
    edge_feat_dim=EDGE_FEAT_DIM,
    memory_dim=MEMORY_DIM,
    time_dim=TIME_DIM,
    embedding_dim=EMBEDDING_DIM,
    message_dim=MESSAGE_DIM
).to(DEVICE)

optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = nn.BCEWithLogitsLoss() # For link prediction

# --- Create DataLoader ---
dataset = DummyTemporalDataset(num_interactions=5000, num_nodes=NUM_NODES, edge_feat_dim=EDGE_FEAT_DIM)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn) # shuffle=False is crucial for time order

# --- Training Loop ---
print(f"Starting training on {DEVICE}...")
for epoch in range(EPOCHS):
    start_epoch_time = time.time()
    model.train()
    model.reset_state() # Reset memory at the start of each epoch
    total_loss = 0

    for batch_idx, (src, dst, ts, edge_feats, labels) in enumerate(dataloader):
        optimizer.zero_grad()

        src, dst, ts, edge_feats, labels = src.to(DEVICE), dst.to(DEVICE), ts.to(DEVICE), edge_feats.to(DEVICE), labels.to(DEVICE)

        # --- TGN Core Logic for Link Prediction ---
        # 1. Compute embeddings for positive edges *at interaction time*
        #    This requires getting the memory state *before* the update from this batch.
        #    In this simplified loop, we call compute_embedding *before* update_state.
        #    A more advanced implementation might need to manage memory snapshots carefully.
        pos_src_emb = model.compute_embedding(src)
        pos_dst_emb = model.compute_embedding(dst)

        # 2. Generate Negative Samples (Simple Random Sampling)
        #    A better strategy uses historical or degree-based sampling.
        num_neg_samples = 1 # Number of negative samples per positive one
        neg_dst = torch.randint(0, NUM_NODES, (len(src) * num_neg_samples,), device=DEVICE)
        # TODO: Ensure neg_dst != actual dst for corresponding src

        neg_src_emb = model.compute_embedding(src.repeat_interleave(num_neg_samples)) # Repeat src for each neg sample
        neg_dst_emb = model.compute_embedding(neg_dst)

        # 3. Calculate Scores (Dot product is common)
        pos_score = torch.sum(pos_src_emb * pos_dst_emb, dim=1)
        neg_score = torch.sum(neg_src_emb * neg_dst_emb, dim=1)

        # 4. Compute Loss
        scores = torch.cat([pos_score, neg_score])
        pos_labels = torch.ones_like(pos_score)
        neg_labels = torch.zeros_like(neg_score)
        all_labels = torch.cat([pos_labels, neg_labels])

        loss = criterion(scores, all_labels)

        # 5. Backpropagation
        loss.backward()
        optimizer.step()

        # --- TGN State Update ---
        # Update memory *after* loss computation based on the batch's interactions
        model.update_state(src, dst, ts, edge_feats)

        # --- Detach memory ---
        # Crucial to prevent gradients flowing across batches
        model.detach_memory()

        total_loss += loss.item()

        # Optional: Print progress
        if (batch_idx + 1) % 20 == 0:
            print(f'Epoch {epoch+1}/{EPOCHS}, Batch {batch_idx+1}/{len(dataloader)}, Loss: {loss.item():.4f}')


    epoch_time = time.time() - start_epoch_time
    avg_loss = total_loss / len(dataloader)
    print(f'Epoch {epoch+1} finished. Avg Loss: {avg_loss:.4f}, Time: {epoch_time:.2f}s')

print("Training finished.")

# --- Evaluation (Conceptual) ---
# model.eval()
# model.reset_state() # Reset for evaluation if needed
# with torch.no_grad():
#    for batch in eval_dataloader:
#        # Similar logic: compute embeddings, make predictions (e.g., rank links)
#        # Update state using update_state as you process eval data chronologically
#        # Detach memory
#        # Calculate metrics (AUC, MRR, etc.)
