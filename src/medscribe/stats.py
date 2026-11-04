"""Bootstrap confidence intervals (percentile method, fixed seed)."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np


def bootstrap_ci(
    values: Sequence[float], *, n: int = 10_000, alpha: float = 0.05, seed: int = 0
) -> tuple[float, float, float]:
    """Mean and (1-alpha) CI of ``values`` resampled with replacement."""
    x = np.asarray(values, dtype=float)
    if x.size == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = x[rng.integers(0, x.size, size=(n, x.size))].mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return float(x.mean()), float(lo), float(hi)


def paired_bootstrap_diff(
    a: Sequence[float], b: Sequence[float], *, n: int = 10_000, alpha: float = 0.05, seed: int = 0
) -> tuple[float, float, float, float]:
    """Mean of (a - b) over paired items, its CI, and the fraction of resamples <= 0."""
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    rng = np.random.default_rng(seed)
    means = d[rng.integers(0, d.size, size=(n, d.size))].mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return float(d.mean()), float(lo), float(hi), float((means <= 0).mean())


def bootstrap_ratio_ci(
    items: Sequence[object],
    stat: Callable[[Sequence[object]], float],
    *,
    n: int = 2_000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float, float]:
    """CI for a corpus-level statistic (e.g. WER = sum errors / sum words)."""
    rng = np.random.default_rng(seed)
    point = stat(items)
    idx = rng.integers(0, len(items), size=(n, len(items)))
    vals = np.array([stat([items[i] for i in row]) for row in idx])
    lo, hi = np.quantile(vals, [alpha / 2, 1 - alpha / 2])
    return float(point), float(lo), float(hi)
