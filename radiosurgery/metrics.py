"""Plan quality measures used in radiosurgery, computed from the dose grid.

Notation: TV is the target (tumour) volume, PIV the prescription isodose
volume (everything receiving at least the prescription dose), TV_PIV the
part of the tumour inside it.

- Coverage = TV_PIV / TV: how much of the tumour gets the full dose.
- Paddick conformity index = TV_PIV^2 / (TV x PIV) (Paddick, 2000). 1 means
  the prescription dose wraps the tumour exactly; it falls both when tumour
  is missed and when healthy tissue is dosed.
- Gradient index = PIV at half the prescription / PIV (Paddick and Lippitz,
  2006): how quickly dose falls off outside the target. Lower is steeper.
- V12: volume of tissue outside the tumour receiving 12 Gy or more, linked to
  the risk of radiation injury to the brain.
"""

import numpy as np
from scipy.spatial import cKDTree

from .masks import outline_points


def dvh(dose, frac, bins=None):
    """Cumulative dose-volume histogram: share of the structure receiving at least each dose."""
    w = frac.ravel()
    d = dose.ravel()[w > 0]
    w = w[w > 0]
    bins = np.linspace(0, float(dose.max()) * 1.01, 400) if bins is None else bins
    order = np.argsort(d)
    d, w = d[order], w[order]
    above = w.sum() - np.concatenate([[0.0], np.cumsum(w)])
    idx = np.searchsorted(d, bins, side="left")
    return bins, above[idx] / w.sum()


def dose_at_volume(dose, frac, share):
    """D_x: the lowest dose received by the hottest `share` of the structure."""
    d = dose.ravel()
    w = frac.ravel()
    keep = w > 0
    d, w = d[keep], w[keep]
    order = np.argsort(d)[::-1]
    cum = np.cumsum(w[order]) / w.sum()
    return float(d[order][np.searchsorted(cum, share)])


def plan_quality(grid, dose_rx, target):
    d = grid.dose
    cc = grid.voxel_cc
    tv = float(target.sum()) * cc
    piv_mask = d >= dose_rx
    piv = float(piv_mask.sum()) * cc
    tv_piv = float((target * piv_mask).sum()) * cc
    half = float((d >= dose_rx / 2).sum()) * cc
    outside = 1 - target
    return {
        "tumour_cc": tv,
        "coverage": tv_piv / tv,
        "paddick_ci": tv_piv**2 / (tv * piv),
        "gradient_index": half / piv,
        "piv_cc": piv,
        "v12_outside_tumour_cc": float(((d >= 12.0) * outside).sum()) * cc,
        "tumour_mean_gy": float((d * target).sum() / target.sum()),
        "tumour_min_gy": dose_at_volume(d, target, 0.999),
        "max_gy": float(d.max()),
    }


def organ_dose(grid, frac):
    d = grid.dose
    return {
        "mean_gy": float((d * frac).sum() / frac.sum()),
        "max_gy": float(d[frac > 0].max()),
        "d1_gy": dose_at_volume(d, frac, 0.01),
        "volume_cc": float(frac.sum()) * grid.voxel_cc,
    }


def gap_mm(contours_a, contours_b):
    """Closest distance between two outlined structures, 0 if their outlines touch or cross."""
    a, b = outline_points(contours_a), outline_points(contours_b)
    if len(a) == 0 or len(b) == 0:
        return float("nan")
    dist, _ = cKDTree(a).query(b)
    return float(dist.min())


def dice(a, b):
    return float(2 * np.minimum(a, b).sum() / (a.sum() + b.sum()))
