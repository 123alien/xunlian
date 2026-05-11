"""Forecasting models: LSTM and DLinear."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTMForecaster(nn.Module):
    """LSTM for single-step time-series forecasting.

    Input:  (B, window_size, input_dim)  — past W hours with features
    Output: (B, 1)                        — predicted next-hour energy
    """

    def __init__(self, input_dim: int, hidden_dim: int = 128,
                 num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, seq_len, input_dim)
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])  # (B, 1) — last timestep


class MovingAvg(nn.Module):
    """Moving average for time-series decomposition (DLinear helper)."""

    def __init__(self, kernel_size: int, stride: int = 1):
        super().__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, seq_len, channels) → (B, seq_len, channels)
        x_padded = F.pad(x.permute(0, 2, 1), (self.kernel_size // 2, self.kernel_size // 2),
                         mode="reflect")
        return self.avg(x_padded).permute(0, 2, 1)


class DLinear(nn.Module):
    """DLinear: Decomposition + Linear layers (AAAI 2023).

    For each input channel: decomposes into trend (moving avg) and seasonal
    (residual), applies separate linear projections, sums across channels.

    Input:  (B, window_size, input_dim)
    Output: (B, 1) — single-step forecast
    """

    def __init__(self, window_size: int, input_dim: int,
                 horizon: int = 1, kernel_size: int = 25):
        super().__init__()
        self.window_size = window_size
        self.input_dim = input_dim
        self.horizon = horizon

        self.decomp = MovingAvg(kernel_size)
        self.linear_trend = nn.ModuleList([
            nn.Linear(window_size, horizon) for _ in range(input_dim)
        ])
        self.linear_seasonal = nn.ModuleList([
            nn.Linear(window_size, horizon) for _ in range(input_dim)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, seq_len, input_dim)
        trend = self.decomp(x)                  # (B, seq_len, input_dim)
        seasonal = x - trend                     # (B, seq_len, input_dim)

        out = torch.zeros(x.size(0), self.horizon, device=x.device)
        for i in range(self.input_dim):
            t_i = trend[:, :, i]                 # (B, seq_len)
            s_i = seasonal[:, :, i]              # (B, seq_len)
            out += self.linear_trend[i](t_i) + self.linear_seasonal[i](s_i)

        return out  # (B, 1)
