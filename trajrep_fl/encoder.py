"""
trajrep_fl/encoder.py
=====================

"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Sparse Random Projection 
# ---------------------------------------------------------------------------

class SparseRandomProjection(nn.Module):
    """Fixed sparse signed CountSketch-style random projection.

    Each input coordinate is mapped to `nonzeros_per_column` output positions
    with deterministic seeded hashes. No d_x by d parameter matrix is stored.
    This is a sparse random projection, though not the original dense-allocated
    iid sparse matrix; see ALIGNMENT.md for the implementation distinction.
    """
    def __init__(self, model_dim: int, projection_dim: int = 128, sparsity: float = 1/3,
                 seed: int = 42, nonzeros_per_column: int = 3, chunk_size: int = 262144):
        super().__init__()
        self.model_dim = int(model_dim)
        self.projection_dim = int(projection_dim)
        self.seed = int(seed)
        self.nonzeros_per_column = max(1, min(self.projection_dim, int(nonzeros_per_column)))
        self.chunk_size = int(chunk_size)
        self.register_buffer('_device_marker', torch.empty(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self.model_dim:
            raise ValueError(f'expected flattened dimension {self.model_dim}, got {x.shape[-1]}')
        original_shape = x.shape[:-1]
        values = x.reshape(-1, self.model_dim)
        out = values.new_zeros((values.size(0), self.projection_dim))
        scale = self.nonzeros_per_column ** -0.5
        for start in range(0, self.model_dim, self.chunk_size):
            stop = min(start + self.chunk_size, self.model_dim)
            pos = torch.arange(start, stop, dtype=torch.int64, device=x.device)
            v = values[:, start:stop] * scale
            for k in range(self.nonzeros_per_column):
                salt = (self.seed + 104729 * (k + 1)) % 2147483647
                h = (pos * (1103515245 + 277 * k) + salt) % 2147483647
                bucket = (h % self.projection_dim).long()
                sign_hash = (pos * (214013 + 43 * k) + salt + 2531011) % 2147483647
                signs = torch.where(sign_hash.remainder(2).eq(0), 1.0, -1.0).to(x.dtype)
                out.scatter_add_(1, bucket.expand(values.size(0), -1), v * signs)
        return out.reshape(*original_shape, self.projection_dim)


# ---------------------------------------------------------------------------
# Temporal Encoder (GRU)
# ---------------------------------------------------------------------------

class TrajectoryEncoder(nn.Module):
    

    def __init__(
        self,
        projection_dim: int = 128,
        embedding_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.projection_dim = projection_dim
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers

        self.gru = nn.GRU(
            input_size=projection_dim,
            hidden_size=embedding_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # Linear output projection 
        self.out_proj = nn.Linear(embedding_dim, embedding_dim)

        self._init_weights()

    def _init_weights(self):
        for name, param in self.gru.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param.data)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param.data)
            elif "bias" in name:
                nn.init.zeros_(param.data)
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(
        self,
        x: torch.Tensor,
        hidden: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        
        out, hidden = self.gru(x, hidden)        
        z = self.out_proj(out[:, -1, :])          
        return z, hidden


# ---------------------------------------------------------------------------
# Auxiliary Decoder (self-supervised calibration)
# ---------------------------------------------------------------------------

class TrajectoryDecoder(nn.Module):
    
    def __init__(self, embedding_dim: int = 64, projection_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embedding_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, projection_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


# ---------------------------------------------------------------------------
# Client State Manager
# ---------------------------------------------------------------------------

class ClientStateManager:
    """
    Per-client server-side state: GRU hidden state and latest embedding.

    """

    def __init__(self, embedding_dim: int, num_gru_layers: int, device: torch.device):
        self.embedding_dim = embedding_dim
        self.num_gru_layers = num_gru_layers
        self.device = device

        self._hidden: Dict[int, torch.Tensor] = {}
        self._embeddings: Dict[int, torch.Tensor] = {}
        self._participation_count: Dict[int, int] = {}
        self._last_participation_round: Dict[int, int] = {}

    # -- hidden state --

    def get_hidden(self, client_id: int) -> Optional[torch.Tensor]:
        return self._hidden.get(client_id)

    def set_hidden(self, client_id: int, hidden: torch.Tensor):
        self._hidden[client_id] = hidden.detach()

    # -- embeddings --

    def get_embedding(self, client_id: int) -> Optional[torch.Tensor]:
        return self._embeddings.get(client_id)

    def set_embedding(self, client_id: int, z: torch.Tensor):
        self._embeddings[client_id] = z.detach()

    # -- participation bookkeeping --

    def increment(self, client_id: int, round_idx: int):
        self._participation_count[client_id] = (
            self._participation_count.get(client_id, 0) + 1
        )
        self._last_participation_round[client_id] = round_idx

    def participation_count(self, client_id: int) -> int:
        return self._participation_count.get(client_id, 0)

    def last_participation_round(self, client_id: int) -> Optional[int]:
        return self._last_participation_round.get(client_id)

    def is_new(self, client_id: int) -> bool:
        return client_id not in self._embeddings

    def reset(self, client_id: int):
        for d in (self._hidden, self._embeddings,
                  self._participation_count, self._last_participation_round):
            d.pop(client_id, None)
