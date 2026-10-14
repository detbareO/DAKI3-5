import numpy as np


def ap_at_k(rel):
    rel = np.asarray(rel, dtype=float)
    k = rel.shape[1]
    # P@i for i = 1..k
    precision_at_i = np.cumsum(rel, axis=1) / np.arange(1, k + 1)
    n_relevant = rel.sum(axis=1)
    return (precision_at_i * rel).sum(axis=1) / np.maximum(n_relevant, 1)


def p_at_k(rel):
    return np.asarray(rel, dtype=float).mean(axis=1)


def mean_or_nan(values):
    return float(np.mean(values)) if len(values) else float("nan")


def bootstrap_ci(values, n_resamples, seed, level=0.95):
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    picks = rng.integers(0, len(values), size=(n_resamples, len(values)))
    means = values[picks].mean(axis=1)
    tail = (1 - level) / 2 * 100
    low, high = np.percentile(means, [tail, 100 - tail])
    return float(low), float(high)


def paired_bootstrap(values_a, values_b, n_resamples, seed, level=0.95):
    difference = np.asarray(values_a, dtype=float) - np.asarray(values_b, dtype=float)
    low, high = bootstrap_ci(difference, n_resamples, seed, level)
    return mean_or_nan(difference), low, high
