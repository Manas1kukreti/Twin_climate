"""Lightweight Transformer encoder model for next-step climate forecasting.

Architecture
------------
.. code-block:: text

    input (batch, input_window, n_features)
        → Linear feature projection (n_features → d_model)
        → sinusoidal positional encoding
        → Transformer encoder (num_encoder_layers, nhead, dim_feedforward, dropout)
        → final timestep representation (batch, d_model)
        → Linear regression head
        → output (batch, n_targets)

This architecture is intentionally lightweight (starting region per the
ClimateTwin specification, not an Aurora-derived design) and must be
reused unchanged for Phase 8 pretraining and Phase 9 fine-tuning so that
scratch and transfer-learning parameter counts are identical.
"""

from __future__ import annotations

import logging
import math

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class SinusoidalPositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding (Vaswani et al., 2017).

    Parameters
    ----------
    d_model : int
        Embedding dimensionality.
    max_len : int
        Maximum sequence length supported.
    """

    def __init__(self, d_model: int, max_len: int = 512) -> None:
        super().__init__()

        position = torch.arange(max_len).unsqueeze(1).float()  # (max_len, 1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )  # (d_model/2,)

        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 0:
            pe[:, 1::2] = torch.cos(position * div_term)
        else:
            pe[:, 1::2] = torch.cos(position * div_term[: pe[:, 1::2].shape[1]])

        # Register as buffer so it moves with .to(device) but is not a parameter
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input embeddings.

        Parameters
        ----------
        x : Tensor
            Shape ``(batch, seq_len, d_model)``.

        Returns
        -------
        Tensor
            Same shape, with positional encoding added.
        """
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :]


class ClimateTransformer(nn.Module):
    """Lightweight Transformer encoder for one-step climate forecasting.

    Parameters
    ----------
    n_features : int
        Number of input features per timestep.
    n_targets : int
        Number of output target variables.
    d_model : int
        Transformer embedding dimensionality.
    nhead : int
        Number of attention heads.
    num_encoder_layers : int
        Number of stacked Transformer encoder layers.
    dim_feedforward : int
        Dimensionality of the feedforward sublayer.
    dropout : float
        Dropout rate.
    max_len : int
        Maximum sequence length supported by positional encoding.

    Input shape
    -----------
    ``(batch, input_window, n_features)``

    Output shape
    ------------
    ``(batch, n_targets)``
    """

    def __init__(
        self,
        n_features: int,
        n_targets: int,
        d_model: int,
        nhead: int,
        num_encoder_layers: int,
        dim_feedforward: int,
        dropout: float,
        max_len: int = 512,
    ) -> None:
        super().__init__()

        self.n_features = n_features
        self.n_targets = n_targets
        self.d_model = d_model

        # Feature projection: n_features -> d_model
        self.input_projection = nn.Linear(n_features, d_model)

        # Sinusoidal positional encoding
        self.positional_encoding = SinusoidalPositionalEncoding(d_model, max_len=max_len)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        # Regression head: uses the representation of the final timestep
        self.regression_head = nn.Linear(d_model, n_targets)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : Tensor
            Shape ``(batch, input_window, n_features)``.

        Returns
        -------
        Tensor
            Shape ``(batch, n_targets)``.
        """
        # (batch, seq_len, n_features) -> (batch, seq_len, d_model)
        h = self.input_projection(x)
        h = self.positional_encoding(h)

        # Transformer encoder: (batch, seq_len, d_model) -> (batch, seq_len, d_model)
        h = self.encoder(h)

        # Use the final timestep's representation for next-step prediction
        final_repr = h[:, -1, :]  # (batch, d_model)

        out = self.regression_head(final_repr)  # (batch, n_targets)
        return out

    def count_parameters(self) -> int:
        """Return the total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
