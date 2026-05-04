"""
aggregation.py — Token aggregation strategy and feature extraction.

Converts per-token, per-layer hidden states from the extraction loop in
``solution.py`` into flat feature vectors for the probe classifier.

Two stages can be customised independently:

  1. ``aggregate`` — select layers and token positions, pool into a vector.
  2. ``extract_geometric_features`` — optional hand-crafted features
     (enabled by setting ``USE_GEOMETRIC = True`` in ``solution.py``).

Both stages are combined by ``aggregation_and_feature_extraction``, the
single entry point called from the notebook.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

TAIL_TOKEN_COUNT = 16
AGG_LAYERS = 4
EXCLUDE_LAST_TOKEN = True


def _real_token_positions(attention_mask: torch.Tensor, device: torch.device) -> torch.Tensor:
    positions = attention_mask.detach().cpu().nonzero(as_tuple=False).flatten()
    return positions.to(device=device, dtype=torch.long)


def _tail_positions(real_positions: torch.Tensor) -> torch.Tensor:
    positions = real_positions
    if EXCLUDE_LAST_TOKEN and positions.numel() > 1:
        positions = positions[:-1]
    if positions.numel() == 0:
        positions = real_positions[-1:]
    return positions[-min(TAIL_TOKEN_COUNT, int(positions.numel())) :]


def _selected_layers(hidden_states: torch.Tensor) -> torch.Tensor:
    n_layers = hidden_states.shape[0]
    start = max(1, n_layers - AGG_LAYERS)
    return hidden_states[start:n_layers]


def aggregate(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Return one fixed-size vector per sample."""
    hidden_dim = hidden_states.shape[-1]
    real_positions = _real_token_positions(attention_mask, hidden_states.device)
    if real_positions.numel() == 0:
        return hidden_states.new_zeros(hidden_dim)

    layers = _selected_layers(hidden_states)
    tail_pos = _tail_positions(real_positions)
    tail_vectors = layers[:, tail_pos, :].mean(dim=0)
    return tail_vectors.mean(dim=0)


def extract_geometric_features(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Return a small fixed set of hardcoded geometric features."""
    work = hidden_states.float()
    real_positions = _real_token_positions(attention_mask, work.device)
    if real_positions.numel() == 0:
        real_positions = torch.zeros(1, dtype=torch.long, device=work.device)

    target_pos = int(real_positions[-1].item())
    tail_pos = _tail_positions(real_positions)

    layers = _selected_layers(work)
    final_token = layers[-1, target_pos, :]
    penult_token = layers[-2, target_pos, :] if layers.shape[0] > 1 else final_token

    tail_final = layers[-1, tail_pos, :]
    layer_target_tokens = layers[:, target_pos, :]

    seq_len = work.new_tensor(float(real_positions.numel()))
    seq_cap = work.new_tensor(float(attention_mask.numel()))

    layer_norms = torch.linalg.vector_norm(layer_target_tokens, dim=-1)
    tail_norms = torch.linalg.vector_norm(tail_final, dim=-1)

    features = torch.stack(
        [
            seq_len,
            seq_len / torch.clamp(seq_cap, min=1.0),
            torch.linalg.vector_norm(final_token),
            torch.linalg.vector_norm(penult_token),
            torch.linalg.vector_norm(final_token - penult_token),
            tail_norms.mean(),
            tail_norms.std(unbiased=False),
            layer_norms.mean(),
            layer_norms.std(unbiased=False),
            F.cosine_similarity(final_token, penult_token, dim=0),
        ]
    )

    return features.to(dtype=hidden_states.dtype, device=hidden_states.device)


def aggregation_and_feature_extraction(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    use_geometric: bool = False,
) -> torch.Tensor:
    agg_features = aggregate(hidden_states, attention_mask)
    if use_geometric:
        geo_features = extract_geometric_features(hidden_states, attention_mask)
        return torch.cat([agg_features, geo_features], dim=0)
    return agg_features
