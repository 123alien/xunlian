"""Metrics: forecasting accuracy + prediction stability."""
import numpy as np


def compute_all(y_true: np.ndarray, y_pred: np.ndarray,
                anomaly_scores: np.ndarray) -> dict:
    """Compute all metrics from predictions and anomaly scores.

    Args:
        y_true: ground-truth energy values, shape (n,)
        y_pred: predicted energy values, shape (n,)
        anomaly_scores: rolling-MAD anomaly scores, shape (n,)

    Returns:
        dict with MAE, RMSE, sMAPE, sigma_err, sigma_score
    """
    errors = np.abs(y_true - y_pred)

    mae = float(np.mean(errors))
    rmse = float(np.sqrt(np.mean(errors ** 2)))
    smape = float(np.mean(2 * errors / (np.abs(y_true) + np.abs(y_pred) + 1e-8)) * 100)

    sigma_err = float(np.std(errors, ddof=1))
    sigma_score = float(np.std(anomaly_scores, ddof=1))

    return {
        "MAE": mae,
        "RMSE": rmse,
        "sMAPE": smape,
        "sigma_err": sigma_err,
        "sigma_score": sigma_score,
    }
