"""Statistics across the patient sample, with intervals that suit small samples."""

import numpy as np
from scipy import stats


def clopper_pearson(k, n, confidence=0.95):
    a = 1 - confidence
    lo = stats.beta.ppf(a / 2, k, n - k + 1) if k > 0 else 0.0
    hi = stats.beta.ppf(1 - a / 2, k + 1, n - k) if k < n else 1.0
    return float(lo), float(hi)


def bootstrap(fn, n, n_boot=5000, seed=0, confidence=0.95):
    """Percentile interval for a statistic of the patients, resampling patients."""
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        v = fn(idx)
        if np.isfinite(v):
            vals.append(v)
    a = (1 - confidence) / 2
    return float(np.quantile(vals, a)), float(np.quantile(vals, 1 - a))


def spearman(x, y, n_boot=5000, seed=0):
    x, y = np.asarray(x, float), np.asarray(y, float)
    rho, p = stats.spearmanr(x, y)
    ci = bootstrap(lambda i: stats.spearmanr(x[i], y[i])[0], len(x), n_boot, seed)
    return {"rho": float(rho), "p": float(p), "ci": ci, "n": int(len(x))}


def holm(pvalues):
    keys = sorted(pvalues, key=pvalues.get)
    out, running = {}, 0.0
    for rank, k in enumerate(keys):
        running = max(running, min(1.0, (len(keys) - rank) * pvalues[k]))
        out[k] = running
    return out


def bland_altman(a, b):
    """Agreement between two ways of measuring the same thing (Bland and Altman, 1986)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    diff = a - b
    bias, sd = float(diff.mean()), float(diff.std(ddof=1))
    se = sd / np.sqrt(len(diff))
    t = stats.t.ppf(0.975, len(diff) - 1)
    return {"bias": bias, "bias_ci": [bias - t * se, bias + t * se],
            "limits": [bias - 1.96 * sd, bias + 1.96 * sd], "n": int(len(diff))}


def exponential_falloff(distance, dose):
    """Fit log(dose) = a + b * distance by least squares, with 95% intervals.

    exp(b) is the factor by which mean dose changes per extra millimetre
    between tumour and cochlea.
    """
    x, y = np.asarray(distance, float), np.log(np.asarray(dose, float))
    X = np.column_stack([np.ones_like(x), x])
    beta, res, *_ = np.linalg.lstsq(X, y, rcond=None)
    n = len(x)
    resid = y - X @ beta
    s2 = resid @ resid / (n - 2)
    cov = s2 * np.linalg.inv(X.T @ X)
    t = stats.t.ppf(0.975, n - 2)
    se = np.sqrt(np.diag(cov))
    r2 = 1 - (resid @ resid) / np.sum((y - y.mean()) ** 2)
    return {"a": float(beta[0]), "b": float(beta[1]), "b_ci": [float(beta[1] - t * se[1]), float(beta[1] + t * se[1])],
            "factor_per_mm": float(np.exp(beta[1])),
            "factor_ci": [float(np.exp(beta[1] - t * se[1])), float(np.exp(beta[1] + t * se[1]))], "r2": float(r2), "n": n}


def summary(values):
    v = np.asarray(values, float)
    return {"median": float(np.median(v)), "q1": float(np.quantile(v, 0.25)), "q3": float(np.quantile(v, 0.75)),
            "min": float(v.min()), "max": float(v.max()), "n": int(len(v))}
