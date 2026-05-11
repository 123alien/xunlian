"""Anomaly detection baselines: Isolation Forest, LSTM-AE.

Each baseline is adapted to the cold-start setting:
- Train on source building data (or features)
- Optionally fine-tune on k days of target building
- Score target building test data
"""
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


# ---------------------------------------------------------------------------
# Scikit-learn baselines
# ---------------------------------------------------------------------------

def run_isolation_forest(X_train_source: np.ndarray,
                         X_test_target: np.ndarray,
                         contamination: float = 0.05,
                         seed: int = 42) -> np.ndarray:
    """Isolation Forest adapted to cold-start setting.

    Args:
        X_train_source: flattened source building features (n_src, n_features)
        X_test_target: flattened target building features (n_tgt, n_features)
        contamination: expected anomaly fraction
        seed: random seed

    Returns:
        anomaly_scores: higher = more anomalous, shape (n_tgt,)
    """
    # Flatten if needed
    if X_train_source.ndim == 3:
        n_src, w, f = X_train_source.shape
        X_train_source = X_train_source.reshape(n_src, w * f)
    if X_test_target.ndim == 3:
        n_tgt, w, f = X_test_target.shape
        X_test_target = X_test_target.reshape(n_tgt, w * f)

    model = IsolationForest(
        contamination=contamination,
        random_state=seed,
        n_estimators=100,
    )
    model.fit(X_train_source)
    # Convert to anomaly scores (higher = more anomalous)
    scores_raw = -model.score_samples(X_test_target)
    # Normalize to z-score
    scores = (scores_raw - np.mean(scores_raw)) / (np.std(scores_raw) + 1e-8)
    return scores


def run_lof(X_train_source: np.ndarray,
            X_test_target: np.ndarray) -> np.ndarray:
    """Local Outlier Factor adapted to cold-start setting.

    Note: LOF is transductive — fits and scores the same data.
    We concatenate source and target, score all, then extract target.
    """
    if X_train_source.ndim == 3:
        n_src, w, f = X_train_source.shape
        X_train_source = X_train_source.reshape(n_src, w * f)
    if X_test_target.ndim == 3:
        n_tgt, w, f = X_test_target.shape
        X_test_target = X_test_target.reshape(n_tgt, w * f)

    X_all = np.concatenate([X_train_source, X_test_target], axis=0)
    model = LocalOutlierFactor(novelty=False, n_neighbors=20)
    # fit_predict returns -1 for outliers, 1 for inliers
    labels = model.fit_predict(X_all)
    scores_raw = -model.negative_outlier_factor_
    # Normalize
    scores = (scores_raw - np.mean(scores_raw)) / (np.std(scores_raw) + 1e-8)
    return scores[-len(X_test_target):]


# ---------------------------------------------------------------------------
# LSTM Autoencoder baseline
# ---------------------------------------------------------------------------

class LSTMAutoencoder(nn.Module):
    """LSTM Autoencoder for unsupervised anomaly detection.

    Encoder: LSTM compresses input → latent vector.
    Decoder: LSTM reconstructs input from latent vector.

    Reconstruction error = anomaly score.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 64,
                 latent_dim: int = 32, num_layers: int = 1, dropout: float = 0.1):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        # Encoder
        self.encoder_lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0.0,
        )
        self.encoder_fc = nn.Linear(hidden_dim, latent_dim)

        # Decoder
        self.decoder_fc = nn.Linear(latent_dim, hidden_dim)
        self.decoder_lstm = nn.LSTM(
            hidden_dim, input_dim, num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0.0,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, seq_len, input_dim)
        B, seq_len, _ = x.shape

        # Encode
        enc_out, (h_n, c_n) = self.encoder_lstm(x)
        latent = self.encoder_fc(enc_out[:, -1, :])  # (B, latent_dim)

        # Decode
        dec_input = self.decoder_fc(latent).unsqueeze(1).repeat(1, seq_len, 1)
        dec_out, _ = self.decoder_lstm(dec_input)

        return dec_out  # (B, seq_len, input_dim)


def train_lstm_ae(model: LSTMAutoencoder, train_loader: DataLoader,
                  val_loader: DataLoader, epochs: int = 50,
                  lr: float = 1e-3, patience: int = 10,
                  device: torch.device = torch.device("cpu"),
                  verbose: bool = True) -> dict:
    """Train LSTM Autoencoder on reconstruction loss."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    best_val_loss = float("inf")
    best_weights = {k: v.clone() for k, v in model.state_dict().items()}
    patience_counter = 0
    history = {"train_loss": [], "val_loss": []}

    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0.0
        for X_batch, _ in train_loader:
            X_batch = X_batch.to(device)
            optimizer.zero_grad()
            recon = model(X_batch)
            loss = criterion(recon, X_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * X_batch.size(0)
        train_loss /= len(train_loader.dataset)
        history["train_loss"].append(train_loss)

        # Validate
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, _ in val_loader:
                X_batch = X_batch.to(device)
                recon = model(X_batch)
                loss = criterion(recon, X_batch)
                val_loss += loss.item() * X_batch.size(0)
        val_loss /= len(val_loader.dataset)
        history["val_loss"].append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            if verbose:
                print(f"  LSTM-AE early stopping at epoch {epoch+1}")
            break

    model.load_state_dict(best_weights)
    history["best_val_loss"] = best_val_loss
    return history


def lstm_ae_anomaly_scores(model: LSTMAutoencoder, loader: DataLoader,
                           device: torch.device = torch.device("cpu")) -> np.ndarray:
    """Compute reconstruction error as anomaly scores.

    Returns per-timestep scores: mean squared error across input features.
    """
    model.eval()
    all_scores = []
    with torch.no_grad():
        for X_batch, _ in loader:
            X_batch = X_batch.to(device)
            recon = model(X_batch)
            # MSE per sample across features and timesteps
            se = ((X_batch - recon) ** 2).mean(dim=(1, 2))  # (B,)
            all_scores.append(se.cpu().numpy())
    scores = np.concatenate(all_scores)
    return (scores - np.mean(scores)) / (np.std(scores) + 1e-8)
