import torch
import torch.nn as nn


class TransformerEncoderRegressor(nn.Module):
    """
    Encoder-only Transformer for sequence-to-one regression.
    Input:  (B, T, D)
    Output: (B,)
    """
    def __init__(
        self,
        input_dim: int,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
        pooling: str = "last",   # "last" or "mean"
    ):
        super().__init__()
        if d_model % nhead != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by nhead ({nhead}).")

        self.pooling = pooling

        self.in_proj = nn.Linear(input_dim, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.out = nn.Linear(d_model, 1)

    def forward(self, x):
        # x: (B, T, D)
        h = self.in_proj(x)          # (B, T, d_model)
        h = self.encoder(h)          # (B, T, d_model)

        if self.pooling == "mean":
            z = h.mean(dim=1)        # (B, d_model)
        else:  # "last"
            z = h[:, -1, :]          # (B, d_model)

        y = self.out(z).squeeze(-1)  # (B,)
        return y
