import numpy as np
import pytest

from radiosurgery.dicomrt import DoseGrid
from radiosurgery.masks import outline_points, rasterise, volume_cc


def grid(n=60, spacing=1.0):
    ax = (np.arange(n) - n / 2) * spacing
    return DoseGrid(np.zeros((n, n, n), np.float32), ax.copy(), ax.copy(), ax.copy())


def circle(r, cx=0.0, cy=0.0, n=200):
    a = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.column_stack([cx + r * np.cos(a), cy + r * np.sin(a)])


def sphere_contours(radius, slice_spacing=1.5):
    zs = np.arange(-radius, radius + 1e-9, slice_spacing)
    return [(float(z), circle(np.sqrt(max(radius**2 - z**2, 0)))) for z in zs if radius**2 - z**2 > 0.01]


def test_cylinder_volume_is_exact_with_fractional_voxels():
    g = grid()
    contours = [(float(z), circle(8.0)) for z in np.arange(-10, 10.5, 1.0)]  # 21 slices, 1 mm apart
    v = volume_cc(rasterise(contours, g, sub=8), g)
    assert v == pytest.approx(np.pi * 8**2 * 21 / 1000, rel=0.01)


def test_sphere_drawn_on_1_5_mm_slices():
    # A 6 mm radius sphere, the size of a small schwannoma, outlined every 1.5 mm.
    g = grid()
    v = volume_cc(rasterise(sphere_contours(6.0), g, sub=8), g)
    assert v == pytest.approx(4 / 3 * np.pi * 6**3 / 1000, rel=0.06)


def test_tiny_structure_keeps_its_volume():
    # A cochlea-sized disc stack: radius 2 mm, 3 slices 1.5 mm apart.
    g = grid()
    contours = [(z, circle(2.0, 0.3, -0.2)) for z in (-1.5, 0.0, 1.5)]
    frac = rasterise(contours, g, sub=8)
    expected = np.pi * 4 * 4.5 / 1000  # each drawn slice stands for 1.5 mm
    assert volume_cc(frac, g) == pytest.approx(expected, rel=0.25)
    assert 0 < frac.max() <= 1


def test_holes_follow_the_even_odd_rule():
    g = grid()
    contours = [(float(z), poly) for z in np.arange(-5, 5.5, 1.0) for poly in (circle(10.0), circle(4.0))]
    v = volume_cc(rasterise(contours, g, sub=8), g)
    assert v == pytest.approx(np.pi * (10**2 - 4**2) * 11 / 1000, rel=0.02)


def test_outline_points_are_evenly_spaced():
    pts = outline_points([(2.0, np.array([[0, 0], [10, 0], [10, 10], [0, 10]], float))], step=0.5)
    assert len(pts) == 80
    assert np.all(pts[:, 2] == 2.0)
