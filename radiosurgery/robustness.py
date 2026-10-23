"""What small positioning errors do to the dose, by Monte Carlo.

If the patient's head sits slightly off where the plan assumed, the whole
dose pattern moves against the anatomy. Each trial draws a random 3D shift
(normal, the same standard deviation on every axis), moves the dose grid by
that much with trilinear interpolation, and remeasures tumour coverage and
cochlea dose. Many trials give the range of outcomes a given setup accuracy
allows.
"""

import numpy as np
from scipy.ndimage import map_coordinates


def shifted_dose(grid, shift_mm, region):
    """Dose inside `region` (a tuple of slices) after moving the dose pattern by shift_mm (z, y, x)."""
    zz, yy, xx = np.meshgrid(*[np.arange(s.start, s.stop, dtype=float) for s in region], indexing="ij")
    coords = [zz - shift_mm[0] / grid.spacing[0], yy - shift_mm[1] / grid.spacing[1], xx - shift_mm[2] / grid.spacing[2]]
    return map_coordinates(grid.dose, coords, order=1, mode="nearest")


def bounding_region(*masks, margin=8):
    nz = np.argwhere(np.max(np.stack(masks), axis=0) > 0)
    lo = np.maximum(nz.min(0) - margin, 0)
    hi = np.minimum(nz.max(0) + margin + 1, masks[0].shape)
    return tuple(slice(a, b) for a, b in zip(lo, hi))


def monte_carlo(grid, dose_rx, target, organ, sigma_mm, trials=200, seed=0):
    region = bounding_region(target, organ)
    t, o = target[region], organ[region]
    rng = np.random.default_rng(seed)
    coverage, organ_mean, shift_len = [], [], []
    for _ in range(trials):
        s = rng.normal(0, sigma_mm, 3)
        d = shifted_dose(grid, s, region)
        coverage.append(float((t * (d >= dose_rx)).sum() / t.sum()))
        organ_mean.append(float((d * o).sum() / o.sum()))
        shift_len.append(float(np.linalg.norm(s)))
    return {"coverage": np.array(coverage), "organ_mean": np.array(organ_mean), "shift_mm": np.array(shift_len)}
