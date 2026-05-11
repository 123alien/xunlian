"""Rolling-MAD anomaly scoring (Module C of TL-TFAD)."""
import numpy as np


def rolling_mad_scores(errors: np.ndarray, window: int = 24,
                       epsilon: float = 1e-4) -> np.ndarray:
    """Compute rolling-MAD-normalized anomaly scores.

    For each timestep t:
      rolling_median[t] = median(errors[t-window:t])
      rolling_MAD[t]   = median(|errors - rolling_median[t]|)
      score[t]         = (errors[t] - rolling_median[t]) / max(rolling_MAD[t], 1% of global std)

    The MAD floor prevents numerical explosion when prediction errors are
    extremely stable (MAD close to 0). Scores are clipped to [-50, 50].

    Args:
        errors: absolute prediction errors |y_t - y_hat_t|
        window: MAD window size in hours (default 24)
        epsilon: not used directly (kept for compatibility)

    Returns:
        anomaly scores, shape (n,) — higher = more anomalous
    """
    n = len(errors)
    scores = np.full(n, np.nan)
    global_std = np.std(errors, ddof=1)
    mad_floor = max(global_std * 0.01, 1e-6)

    for t in range(n):
        start = max(0, t - window)
        window_errors = errors[start:t + 1]

        med = np.median(window_errors)
        mad = np.median(np.abs(window_errors - med))
        mad = max(mad, mad_floor)

        s = (errors[t] - med) / mad
        scores[t] = np.clip(s, -50.0, 50.0)

    return scores


def detect_anomalies(scores: np.ndarray, threshold: float = 3.0) -> np.ndarray:
    """Flag anomalies where score exceeds threshold.

    Args:
        scores: anomaly scores
        threshold: score > threshold → anomaly

    Returns:
        binary array, 1 = anomaly
    """
    return (scores > threshold).astype(int)
