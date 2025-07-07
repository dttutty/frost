import torch
import torch.nn as nn
import math

class JODIE(nn.Module):
    """
    JODIE model implementation.

    Args:
        n_users (int): Number of unique users.
        n_items (int): Number of unique items.
        embed_dim (int): Dimension of user and item embeddings.
        feature_dim (int): Dimension of interaction features.
        hidden_dim (int): Hidden dimension for the RNNs (GRUs).
        dropout_rate (float): Dropout rate.
        use_features (bool): Whether to use interaction features. Defaults to True.
    """
    def __init__(self, n_users, n_items, embed_dim, feature_dim, hidden_dim, dropout_rate=0.1, use_features=True):
        super().__init__()

        self.n_users = n_users
        self.n_items = n_items
        self.embed_dim = embed_dim
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.use_features = use_features

        # --- Embedding Layers ---
        # Initialize embeddings for users and items
        # Use nn.Parameter directly to manage them as learnable tensors
        self.user_embeddings = nn.Parameter(torch.Tensor(n_users, embed_dim))
        self.item_embeddings = nn.Parameter(torch.Tensor(n_items, embed_dim))

        # --- RNN Layers (GRU) for state updates ---
        # Input to RNN: concatenation of other entity's embedding and interaction feature
        rnn_input_dim = embed_dim + feature_dim if use_features else embed_dim

        # User RNN updates user embedding based on interaction with an item
        self.user_rnn = nn.GRUCell(input_size=rnn_input_dim, hidden_size=embed_dim)
        # Item RNN updates item embedding based on interaction with a user
        self.item_rnn = nn.GRUCell(input_size=rnn_input_dim, hidden_size=embed_dim)

        # --- Projection Layers ---
        # Predicts the embedding of the *other* entity for the *next* interaction
        # Input to projection: updated embedding of the current entity + interaction feature
        proj_input_dim = embed_dim + feature_dim if use_features else embed_dim

        # Project user state to predict next interacted item's embedding
        self.user_projection = nn.Linear(proj_input_dim, embed_dim)
        # Project item state to predict next interacting user's embedding
        self.item_projection = nn.Linear(proj_input_dim, embed_dim)

        # --- Utilities ---
        self.dropout = nn.Dropout(p=dropout_rate)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        # Initialize embeddings (e.g., Xavier initialization)
        nn.init.xavier_uniform_(self.user_embeddings)
        nn.init.xavier_uniform_(self.item_embeddings)
        # GRU weights/biases are initialized by default, which is often okay.
        # Initialize projection layers
        nn.init.xavier_uniform_(self.user_projection.weight)
        nn.init.zeros_(self.user_projection.bias)
        nn.init.xavier_uniform_(self.item_projection.weight)
        nn.init.zeros_(self.item_projection.bias)

    def forward(self, user_ids, item_ids, features, current_user_embeds, current_item_embeds):
        """
        Performs one step of JODIE update and projection.

        Args:
            user_ids (torch.Tensor): Tensor of user IDs for the current interactions.
            item_ids (torch.Tensor): Tensor of item IDs for the current interactions.
            features (torch.Tensor): Tensor of interaction features. Shape (batch_size, feature_dim).
                                      If use_features is False, this might be ignored or zeros.
            current_user_embeds (torch.Tensor): User embeddings *before* this interaction.
                                                Shape (batch_size, embed_dim).
            current_item_embeds (torch.Tensor): Item embeddings *before* this interaction.
                                                Shape (batch_size, embed_dim).

        Returns:
            tuple: Contains:
                - updated_user_embeds (torch.Tensor): User embeddings *after* this interaction update.
                - updated_item_embeds (torch.Tensor): Item embeddings *after* this interaction update.
                - predicted_next_user_embeds (torch.Tensor): Projected user embeddings for the *next* interaction (based on updated item state).
                - predicted_next_item_embeds (torch.Tensor): Projected item embeddings for the *next* interaction (based on updated user state).
        """
        if not self.use_features:
            # Create zero features if not using them, simplifies concatenation
            features = torch.zeros(user_ids.size(0), self.feature_dim,
                                   device=user_ids.device, dtype=current_user_embeds.dtype)

        # --- 1. Prepare RNN Inputs ---
        # User RNN input: Item embedding + features
        user_rnn_input = torch.cat([current_item_embeds, features], dim=-1)
        # Item RNN input: User embedding + features
        item_rnn_input = torch.cat([current_user_embeds, features], dim=-1)

        # Apply dropout to RNN inputs
        user_rnn_input = self.dropout(user_rnn_input)
        item_rnn_input = self.dropout(item_rnn_input)

        # --- 2. Update Embeddings using RNNs ---
        # The hidden state of the GRU is the user/item embedding
        updated_user_embeds = self.user_rnn(user_rnn_input, current_user_embeds)
        updated_item_embeds = self.item_rnn(item_rnn_input, current_item_embeds)

        # Apply dropout to updated embeddings
        updated_user_embeds_proj_in = self.dropout(updated_user_embeds)
        updated_item_embeds_proj_in = self.dropout(updated_item_embeds)

        # --- 3. Project to Predict Next Embeddings ---
        # Prepare projection inputs
        # User projection input: Updated user embedding + features
        user_proj_input = torch.cat([updated_user_embeds_proj_in, features], dim=-1)
        # Item projection input: Updated item embedding + features
        item_proj_input = torch.cat([updated_item_embeds_proj_in, features], dim=-1)

        # Perform projection
        predicted_next_item_embeds = self.user_projection(user_proj_input)
        predicted_next_user_embeds = self.item_projection(item_proj_input)

        return updated_user_embeds, updated_item_embeds, predicted_next_user_embeds, predicted_next_item_embeds

    def get_embeddings(self, user_ids, item_ids):
        """Helper function to get current embeddings by ID."""
        # Ensure IDs are LongTensors
        if user_ids.dtype != torch.long:
           user_ids = user_ids.long()
        if item_ids.dtype != torch.long:
            item_ids = item_ids.long()

        user_embeds = self.user_embeddings[user_ids]
        item_embeds = self.item_embeddings[item_ids]
        return user_embeds, item_embeds

# --- How to use it in a training loop (Conceptual) ---

# Assume `interactions` is a list or array of tuples: (user_id, item_id, timestamp, feature_vector)
# Interactions MUST be sorted by timestamp!

model = JODIE(n_users, n_items, embed_dim, feature_dim, hidden_dim)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# --- State Management (Crucial for JODIE) ---
# Keep track of the *most recent* embedding for each user and item encountered so far.
# Initialize with the static embeddings from the model. Detach initially.
user_embed_memory = model.user_embeddings.clone().detach()
item_embed_memory = model.item_embeddings.clone().detach()

model.train()
total_loss = 0

for u, i, ts, feat in interactions: # Iterate in temporal order
    optimizer.zero_grad()

    # 1. Get current embeddings from memory
    # Ensure requires_grad=True for backpropagation through the memory state
    # If it's the first time seeing u or i, use the initial embedding.
    current_u_embed = user_embed_memory[u].clone().detach().requires_grad_(True)
    current_i_embed = item_embed_memory[i].clone().detach().requires_grad_(True)

    # Convert features to tensor
    feat_tensor = torch.tensor(feat, dtype=torch.float32).unsqueeze(0) # Add batch dim
    u_tensor = torch.tensor([u], dtype=torch.long)
    i_tensor = torch.tensor([i], dtype=torch.long)

    # 2. Run the JODIE forward pass
    updated_u_embed, updated_i_embed, pred_next_u_embed, pred_next_i_embed = model(
        u_tensor, i_tensor, feat_tensor, current_u_embed.unsqueeze(0), current_i_embed.unsqueeze(0)
    )
    updated_u_embed = updated_u_embed.squeeze(0) # Remove batch dim
    updated_i_embed = updated_i_embed.squeeze(0)
    pred_next_u_embed = pred_next_u_embed.squeeze(0)
    pred_next_i_embed = pred_next_i_embed.squeeze(0)


    # --- 3. Calculate Loss ---
    # Loss has two main components in JODIE:
    # a) Prediction Loss: How well the projected embedding predicts the actual *next* interaction's embedding.
    #    This often uses negative sampling. Predict the *updated* item embedding using the projection
    #    from the updated user embedding.
    # b) (Optional) Regularization/Temporal Loss: Might encourage smoothness or penalize drift.

    # Example: Prediction Loss using Negative Sampling (Simplified)
    # We want pred_next_i_embed (derived from user u) to be close to updated_i_embed
    # and far from embeddings of negative items.
    # You'd need a negative sampling strategy here.
    num_neg_samples = 5
    neg_item_ids = torch.randint(0, model.n_items, (num_neg_samples,))
    # Avoid sampling the actual item i
    neg_item_ids = neg_item_ids[neg_item_ids != i]
    if len(neg_item_ids) > 0: # Handle edge case if n_items is small
        neg_item_embeds = item_embed_memory[neg_item_ids].clone().detach() # Use memory state

        positive_score = torch.sigmoid(torch.sum(pred_next_i_embed * updated_i_embed, dim=-1))
        negative_scores = torch.sigmoid(torch.sum(pred_next_i_embed.unsqueeze(0) * neg_item_embeds, dim=-1))

        loss_item_pred = -torch.log(positive_score + 1e-6) - torch.sum(torch.log(1 - negative_scores + 1e-6))
    else:
        loss_item_pred = torch.tensor(0.0, device=updated_i_embed.device)

    # Similarly predict user embedding based on item projection (loss_user_pred)
    # ... (implementation similar to loss_item_pred)
    loss_user_pred = torch.tensor(0.0) # Placeholder

    loss = loss_item_pred + loss_user_pred # Combine losses

    # 4. Backpropagation
    loss.backward()

     # Gradient clipping is often useful with RNNs
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    torch.nn.utils.clip_grad_norm_([current_u_embed, current_i_embed], max_norm=1.0) # Clip gradients flowing *into* memory


    optimizer.step()

    # --- 5. Update Embedding Memory (CRUCIAL) ---
    # Update the memory with the *newly computed* embeddings. Detach to stop gradient flow beyond this step.
    user_embed_memory[u] = updated_u_embed.clone().detach()
    item_embed_memory[i] = updated_i_embed.clone().detach()

    total_loss += loss.item()

print(f"Total Loss: {total_loss}")
