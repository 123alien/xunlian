"""Controlled anomaly injection for test-set evaluation (Scenario D.2).

All injections modify COPIES of test data. Original data is never touched.
"""
import numpy as np


def inject_anomalies(y_test: np.ndarray, injection_rate: float = 0.05,
                     seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """Inject synthetic anomalies into test data (read-only copy).

    Args:
        y_test: original test values, shape (n,)
        injection_rate: fraction of points to inject (default 0.05 = 5%)
        seed: random seed for reproducibility

    Returns:
        y_injected: modified test values, shape (n,)
        labels: binary anomaly labels, shape (n,) — 1 = injected anomaly
    """
    rng = np.random.RandomState(seed)
    y = y_test.copy().astype(np.float64)
    n = len(y)
    labels = np.zeros(n, dtype=int)

    n_anomalies = max(1, int(n * injection_rate))

    # Allocate anomalies across types
    n_spike = n_anomalies // 6
    n_drop = n_anomalies // 6
    n_scale = n_anomalies // 6
    n_drift = n_anomalies // 6
    n_missing = n_anomalies // 6
    n_stuck = n_anomalies - (n_spike + n_drop + n_scale + n_drift + n_missing)

    local_std = max(np.std(y_test) * 0.5, 1e-6)
    local_mean = max(np.mean(y_test), 1e-6)

    # 1. Point spike
    _inject_spike(y, labels, n_spike, local_std, local_mean, rng)

    # 2. Point drop
    _inject_drop(y, labels, n_drop, local_std, local_mean, rng)

    # 3. Level shift (scale)
    _inject_scale(y, labels, n_scale, local_mean, rng)

    # 4. Gradual drift
    _inject_drift(y, labels, n_drift, local_mean, rng)

    # 5. Missing (zero)
    _inject_missing(y, labels, n_missing, rng)

    # 6. Stuck sensor
    _inject_stuck(y, labels, n_stuck, rng)

    return y.astype(np.float32), labels


def _random_positions(rng: np.random.RandomState, n_points: int, n_total: int,
                      min_gap: int = 4) -> np.ndarray:
    """Generate non-overlapping anomaly positions with minimum gap."""
    # Simple approach: sample random points, but ensure min_gap
    candidates = np.arange(n_total)
    rng.shuffle(candidates)
    chosen = []
    for c in candidates:
        if all(abs(c - x) > min_gap for x in chosen):
            chosen.append(c)
            if len(chosen) >= n_points:
                break
    return np.array(sorted(chosen))


def _inject_spike(y: np.ndarray, labels: np.ndarray, n: int,
                  local_std: float, local_mean: float, rng: np.random.RandomState):
    if n <= 0:
        return
    pos = _random_positions(rng, n, len(y))
    amps = rng.uniform(3.0, 6.0, size=n) * local_std
    for i, p in enumerate(pos):
        y[p] = max(0, y[p] + amps[i])
        labels[p] = 1


def _inject_drop(y: np.ndarray, labels: np.ndarray, n: int,
                 local_std: float, local_mean: float, rng: np.random.RandomState):
    if n <= 0:
        return
    pos = _random_positions(rng, n, len(y))
    factors = rng.uniform(0.2, 0.5, size=n)
    for i, p in enumerate(pos):
        y[p] = max(0, y[p] * factors[i])
        labels[p] = 1


def _inject_scale(y: np.ndarray, labels: np.ndarray, n: int,
                  local_mean: float, rng: np.random.RandomState):
    if n <= 0:
        return
    pos = _random_positions(rng, n, len(y))
    for p in pos:
        dur = int(rng.randint(3, 13))
        end = min(p + dur, len(y))
        factor = rng.uniform(1.5, 3.0)
        for t in range(p, end):
            y[t] = max(0, y[t] * factor)
            labels[t] = 1


def _inject_drift(y: np.ndarray, labels: np.ndarray, n: int,
                  local_mean: float, rng: np.random.RandomState):
    if n <= 0:
        return
    pos = _random_positions(rng, n, len(y))
    for p in pos:
        dur = int(rng.randint(24, 49))
        end = min(p + dur, len(y))
        rate = rng.uniform(0.005, 0.02)
        for i, t in enumerate(range(p, end)):
            y[t] = max(0, y[t] * (1 + rate) ** i)
            labels[t] = 1


def _inject_missing(y: np.ndarray, labels: np.ndarray, n: int,
                    rng: np.random.RandomState):
    if n <= 0:
        return
    pos = _random_positions(rng, n, len(y))
    for p in pos:
        dur = int(rng.randint(1, 7))
        end = min(p + dur, len(y))
        for t in range(p, end):
            y[t] = 0.0
            labels[t] = 1


def _inject_stuck(y: np.ndarray, labels: np.ndarray, n: int,
                  rng: np.random.RandomState):
    if n <= 0:
        return
    pos = _random_positions(rng, n, len(y))
    for p in pos:
        if p < 1:
            continue
        dur = int(rng.randint(6, 25))
        end = min(p + dur, len(y))
        stuck_val = y[p - 1]
        for t in range(p, end):
            y[t] = stuck_val
            labels[t] = 1
