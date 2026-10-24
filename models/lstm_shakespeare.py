

from __future__ import annotations

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Building block
# ---------------------------------------------------------------------------

class _NormLSTMLayer(nn.Module):


    def __init__(
        self,
        input_size:  int,
        hidden_size: int,
        dropout:     float = 0.0,
    ):
        super().__init__()
        self.lstm       = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout    = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def forward(
        self,
        x:      torch.Tensor,
        hidden: tuple | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:

        out, (h, c) = self.lstm(x, hidden)
        out = self.layer_norm(out)
        out = self.dropout(out)
        return out, (h, c)


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class ShakespeareLSTM(nn.Module):


    def __init__(
        self,
        vocab_size:  int   = 80,
        embed_dim:   int   = 8,
        hidden_size: int   = 256,
        num_layers:  int   = 2,
        dropout:     float = 0.2,
    ):
        super().__init__()
        self.vocab_size  = vocab_size
        self.hidden_size = hidden_size
        self.num_layers  = num_layers

        # ── Input embedding with dropout ──────────────────────────────────
        self.embedding      = nn.Embedding(vocab_size, embed_dim)
        self.embed_dropout  = nn.Dropout(dropout)

        # ── Stacked LSTM layers, each with LayerNorm ─────────────────────
        self.lstm_layers = nn.ModuleList()
        for i in range(num_layers):
            in_size    = embed_dim if i == 0 else hidden_size
            # Apply inter-layer dropout between all layers except the last
            layer_drop = dropout if i < num_layers - 1 else 0.0
            self.lstm_layers.append(
                _NormLSTMLayer(in_size, hidden_size, dropout=layer_drop)
            )

        # ── Output head: LayerNorm → FC → ReLU → Dropout → FC ───────────
        self.output_norm = nn.LayerNorm(hidden_size)
        self.output_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, vocab_size),
        )

    # -----------------------------------------------------------------------

    def forward(
        self,
        x:      torch.Tensor,
        hidden: tuple | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:

        # ── Split  into per-layer tuples ───────────────────────────
        if hidden is None:
            per_layer_hidden: list = [None] * self.num_layers
        else:
            h_all, c_all = hidden          # 
            per_layer_hidden = [
                (h_all[i : i + 1], c_all[i : i + 1])
                for i in range(self.num_layers)
            ]

        # ── Embedding ─────────────────────────────────────────────────────
        out = self.embed_dropout(self.embedding(x))   

        # ── LSTM stack ────────────────────────────────────────────────────
        new_hidden: list = []
        for i, layer in enumerate(self.lstm_layers):
            out, hc = layer(out, per_layer_hidden[i])
            new_hidden.append(hc)          

        # ── Output head ───────────────────────────────────────────────────
        out    = self.output_norm(out)
        logits = self.output_head(out)     

        # ── Re-stack per-layer states to standard  ─────
        h_out = torch.cat([h for h, _ in new_hidden], dim=0)
        c_out = torch.cat([c for _, c in new_hidden], dim=0)

        return logits, (h_out, c_out)

    # -----------------------------------------------------------------------

    def init_hidden(
        self,
        batch_size: int,
        device:     torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        h = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
        c = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
        return (h, c)

    @property
    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ---------------------------------------------------------------------------
# Factory 
# ---------------------------------------------------------------------------

def get_shakespeare_model(vocab_size: int = 80) -> ShakespeareLSTM:
    return ShakespeareLSTM(vocab_size=vocab_size)
