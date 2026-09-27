"""Multivariate LSTM model for next-step climate forecasting.

Architecture
------------
.. code-block:: text

    input (batch, input_window, n_features)
        → LSTM encoder (num_layers, hidden_dim, dropout)
        → final hidden state (batch, hidden_dim)
        → Linear regression head
        → output (batch, n_targets)

The model consumes a sequence of previous multivariate climate states
and predicts the next complete climate state.
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class ClimateLSTM(nn.Module):
    """Multivariate LSTM for one-step climate forecasting.

    Parameters
    ----------
    n_features : int
        Number of input features per timestep.
    n_targets : int
        Number of output target variables.
    hidden_dim : int
        LSTM hidden state dimensionality.
    num_layers : int
        Number of stacked LSTM layers.
    dropout : float
        Dropout rate between LSTM layers (applied only if ``num_layers > 1``).

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
        hidden_dim: int,
        num_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()

        self.n_features = n_features
        self.n_targets = n_targets
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.fc = nn.Linear(hidden_dim, n_targets)

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
        # lstm_out: (batch, seq_len, hidden_dim)
        # h_n:     (num_layers, batch, hidden_dim)
        lstm_out, (h_n, _) = self.lstm(x)

        # Use the last layer's final hidden state
        # h_n shape: (num_layers, batch, hidden_dim) → take [-1]
        last_hidden = h_n[-1]  # (batch, hidden_dim)

        out = self.fc(last_hidden)  # (batch, n_targets)
        return out

    def count_parameters(self) -> int:
        """Return the total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
