"""Controlled anomaly injection for test-set evaluation (Scenario D.2).

All injections modify COPIES of test data. Original data is never touched.
"""
import numpy as np


def inject_anomalies(y_test: np.ndarray, injection_rate: float = 0.05,
                     seed: int = 42,
                     return_types: bool = False) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, np.ndarray]:
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
    types = np.array(["normal"] * n, dtype=object)

    # Budget is point-level, not event-level. Duration-based anomalies are
    # capped so the final label prevalence matches injection_rate closely.
    target_points = max(1, int(n * injection_rate))
    occupied = np.zeros(n, dtype=bool)
    anomaly_order = ["spike", "drop", "level_shift", "drift", "missing", "stuck"]

    local_std = max(np.std(y_test) * 0.5, 1e-6)
    order_idx = 0
    attempts = 0
    max_attempts = max(100, target_points * 20)
    while labels.sum() < target_points and attempts < max_attempts:
        attempts += 1
        anomaly_type = anomaly_order[order_idx % len(anomaly_order)]
        order_idx += 1
        remaining = target_points - int(labels.sum())
        start = _choose_start(rng, occupied)
        if start is None:
            break
        _inject_budgeted_event(y, labels, types, occupied, start, anomaly_type,
                               remaining, local_std, rng)

    if return_types:
        return y.astype(np.float32), labels, types
    return y.astype(np.float32), labels


def _choose_start(rng: np.random.RandomState, occupied: np.ndarray,
                  min_gap: int = 4) -> int | None:
    candidates = np.where(~occupied)[0]
    rng.shuffle(candidates)
    for c in candidates:
        left = max(0, c - min_gap)
        right = min(len(occupied), c + min_gap + 1)
        if not occupied[left:right].any():
            return int(c)
    return None


def _available_segment(occupied: np.ndarray, start: int, max_len: int) -> np.ndarray:
    end = min(len(occupied), start + max_len)
    idx = []
    for t in range(start, end):
        if occupied[t]:
            break
        idx.append(t)
    return np.array(idx, dtype=int)


def _inject_budgeted_event(y: np.ndarray, labels: np.ndarray, types: np.ndarray,
                           occupied: np.ndarray, start: int, anomaly_type: str,
                           remaining: int, local_std: float,
                           rng: np.random.RandomState):
    if anomaly_type in ("spike", "drop"):
        duration = 1
    elif anomaly_type == "level_shift":
        duration = int(rng.randint(3, 13))
    elif anomaly_type == "drift":
        duration = int(rng.randint(24, 49))
    elif anomaly_type == "missing":
        duration = int(rng.randint(1, 7))
    else:
        duration = int(rng.randint(6, 25))

    duration = max(1, min(duration, remaining))
    idx = _available_segment(occupied, start, duration)
    if len(idx) == 0:
        return

    if anomaly_type == "spike":
        y[idx[0]] = max(0, y[idx[0]] + rng.uniform(3.0, 6.0) * local_std)
        idx = idx[:1]
    elif anomaly_type == "drop":
        y[idx[0]] = max(0, y[idx[0]] * rng.uniform(0.2, 0.5))
        idx = idx[:1]
    elif anomaly_type == "level_shift":
        factor = rng.uniform(1.5, 3.0)
        y[idx] = np.maximum(0, y[idx] * factor)
    elif anomaly_type == "drift":
        rate = rng.uniform(0.005, 0.02)
        for i, t in enumerate(idx):
            y[t] = max(0, y[t] * (1 + rate) ** i)
    elif anomaly_type == "missing":
        y[idx] = 0.0
    elif anomaly_type == "stuck":
        if idx[0] < 1:
            return
        y[idx] = y[idx[0] - 1]

    for t in idx:
        _mark(labels, types, int(t), anomaly_type)
        occupied[int(t)] = True


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


def _mark(labels: np.ndarray, types: np.ndarray, idx: int, anomaly_type: str):
    labels[idx] = 1
    types[idx] = anomaly_type if types[idx] == "normal" else f"{types[idx]}+{anomaly_type}"


def _inject_spike(y: np.ndarray, labels: np.ndarray, types: np.ndarray, n: int,
                  local_std: float, local_mean: float, rng: np.random.RandomState):
    if n <= 0:
        return
    pos = _random_positions(rng, n, len(y))
    amps = rng.uniform(3.0, 6.0, size=n) * local_std
    for i, p in enumerate(pos):
        y[p] = max(0, y[p] + amps[i])
        _mark(labels, types, p, "spike")


def _inject_drop(y: np.ndarray, labels: np.ndarray, types: np.ndarray, n: int,
                 local_std: float, local_mean: float, rng: np.random.RandomState):
    if n <= 0:
        return
    pos = _random_positions(rng, n, len(y))
    factors = rng.uniform(0.2, 0.5, size=n)
    for i, p in enumerate(pos):
        y[p] = max(0, y[p] * factors[i])
        _mark(labels, types, p, "drop")


def _inject_scale(y: np.ndarray, labels: np.ndarray, types: np.ndarray, n: int,
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
            _mark(labels, types, t, "level_shift")


def _inject_drift(y: np.ndarray, labels: np.ndarray, types: np.ndarray, n: int,
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
            _mark(labels, types, t, "drift")


def _inject_missing(y: np.ndarray, labels: np.ndarray, types: np.ndarray, n: int,
                    rng: np.random.RandomState):
    if n <= 0:
        return
    pos = _random_positions(rng, n, len(y))
    for p in pos:
        dur = int(rng.randint(1, 7))
        end = min(p + dur, len(y))
        for t in range(p, end):
            y[t] = 0.0
            _mark(labels, types, t, "missing")


def _inject_stuck(y: np.ndarray, labels: np.ndarray, types: np.ndarray, n: int,
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
            _mark(labels, types, t, "stuck")
