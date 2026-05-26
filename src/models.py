"""Forecasting models: LSTM, DLinear, and PatchTST-style Transformer."""
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


class PatchTSTForecaster(nn.Module):
    """Compact PatchTST-style baseline for single-step load forecasting.

    The implementation keeps the key PatchTST idea used in recent forecasting
    papers: split each variable into temporal patches, embed patches, encode
    them with a Transformer, and aggregate across variables for prediction.
    It is intentionally small because the input window is only 24 hours.
    """

    def __init__(
        self,
        window_size: int,
        input_dim: int,
        patch_len: int = 6,
        stride: int = 3,
        d_model: int = 64,
        n_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        if patch_len > window_size:
            raise ValueError("patch_len must be <= window_size")
        self.window_size = window_size
        self.input_dim = input_dim
        self.patch_len = patch_len
        self.stride = stride
        self.n_patches = 1 + (window_size - patch_len) // stride

        self.patch_proj = nn.Linear(patch_len, d_model)
        self.pos_embedding = nn.Parameter(torch.zeros(1, self.n_patches, d_model))
        self.channel_embedding = nn.Parameter(torch.zeros(1, input_dim, 1, d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 1),
        )
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)
        nn.init.trunc_normal_(self.channel_embedding, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C) -> patches: (B, C, N, patch_len)
        patches = x.permute(0, 2, 1).unfold(dimension=-1, size=self.patch_len, step=self.stride)
        z = self.patch_proj(patches)
        z = z + self.pos_embedding[:, None, :, :] + self.channel_embedding
        bsz, channels, n_patches, d_model = z.shape
        z = z.reshape(bsz * channels, n_patches, d_model)
        z = self.encoder(z)
        z = z.mean(dim=1).reshape(bsz, channels, d_model)
        z = z.mean(dim=1)
        return self.head(z)
