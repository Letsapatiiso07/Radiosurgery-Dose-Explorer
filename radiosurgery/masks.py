"""Turning hand-drawn outlines into 3D masks on the dose grid.

Doctors outline a structure slice by slice on the MRI. Each outline is a
closed polygon at one height. To compare it with the dose, every dose voxel
gets the fraction of it that lies inside the structure:

- In-plane, each voxel is sampled on a fine sub-grid (default 4 x 4) and each
  sample point tested against the polygons with the even-odd rule, so holes
  and islands work.
- Along z, each dose slice takes the outline from the nearest drawn slice, if
  one lies within half the drawing spacing; above and below the structure
  there is nothing.

Fractional voxels matter for small structures: the cochlea is only a few
cubic millimetres, a handful of 1 mm voxels.
"""

import numpy as np
from matplotlib.path import Path as Polygon


def contour_spacing(contours):
    zs = np.unique([z for z, _ in contours])
    return float(np.median(np.diff(zs))) if len(zs) > 1 else 1.0


def inside(points_xy, polygons):
    """Even-odd rule: a point is inside if it falls in an odd number of polygons."""
    count = np.zeros(len(points_xy), dtype=np.int32)
    for poly in polygons:
        count += Polygon(poly).contains_points(points_xy)
    return count % 2 == 1


def rasterise(contours, grid, sub=4):
    """Fraction of each dose voxel inside the structure, shape [z, y, x]."""
    frac = np.zeros(grid.dose.shape, dtype=np.float32)
    if not contours:
        return frac
    by_z = {}
    for z, xy in contours:
        by_z.setdefault(z, []).append(xy)
    slice_z = np.array(sorted(by_z))
    half = contour_spacing(contours) / 2 + 1e-6

    dy, dx = grid.y[1] - grid.y[0], grid.x[1] - grid.x[0]
    offsets = (np.arange(sub) + 0.5) / sub - 0.5
    for k, zk in enumerate(grid.z):
        nearest = slice_z[np.argmin(np.abs(slice_z - zk))]
        if abs(nearest - zk) > half:
            continue
        polys = by_z[nearest]
        lo, hi = np.min([p.min(0) for p in polys], 0), np.max([p.max(0) for p in polys], 0)
        ix = np.flatnonzero((grid.x >= lo[0] - dx) & (grid.x <= hi[0] + dx))
        iy = np.flatnonzero((grid.y >= lo[1] - dy) & (grid.y <= hi[1] + dy))
        if len(ix) == 0 or len(iy) == 0:
            continue
        sx = (grid.x[ix][:, None] + offsets[None, :] * dx).ravel()
        sy = (grid.y[iy][:, None] + offsets[None, :] * dy).ravel()
        SX, SY = np.meshgrid(sx, sy)
        hit = inside(np.column_stack([SX.ravel(), SY.ravel()]), polys).reshape(len(iy), sub, len(ix), sub)
        frac[k, iy[0]:iy[-1] + 1, ix[0]:ix[-1] + 1] = hit.mean(axis=(1, 3))
    return frac


def volume_cc(frac, grid):
    return float(frac.sum()) * grid.voxel_cc


def outline_points(contours, step=0.5):
    """Points every `step` mm along every outline, as (N, 3): for distances between structures."""
    pts = []
    for z, xy in contours:
        closed = np.vstack([xy, xy[:1]])
        for a, b in zip(closed[:-1], closed[1:]):
            n = max(1, int(np.ceil(np.linalg.norm(b - a) / step)))
            s = np.linspace(0, 1, n, endpoint=False)[:, None]
            seg = a + s * (b - a)
            pts.append(np.column_stack([seg, np.full(len(seg), z)]))
    return np.vstack(pts) if pts else np.zeros((0, 3))
