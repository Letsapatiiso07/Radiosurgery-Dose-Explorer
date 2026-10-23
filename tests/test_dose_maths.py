import numpy as np
import pytest

from radiosurgery.dicomrt import DoseGrid
from radiosurgery.metrics import dice, dose_at_volume, dvh, gap_mm, organ_dose, plan_quality
from radiosurgery.radiobiology import bed_lq, bed_lql, eqd2
from radiosurgery.robustness import monte_carlo, shifted_dose


def grid_with(dose_fn, n=41):
    ax = np.arange(n) - n // 2.0
    Z, Y, X = np.meshgrid(ax, ax, ax, indexing="ij")
    return DoseGrid(dose_fn(Z, Y, X).astype(np.float32), ax.copy(), ax.copy(), ax.copy()), (Z, Y, X)


def ball(Z, Y, X, r, c=(0, 0, 0)):
    return (((Z - c[0]) ** 2 + (Y - c[1]) ** 2 + (X - c[2]) ** 2) <= r**2).astype(np.float32)


def test_perfectly_conformal_plan_scores_one():
    g, (Z, Y, X) = grid_with(lambda Z, Y, X: 26 * ball(Z, Y, X, 6))  # 26 Gy exactly inside a 6 mm ball
    q = plan_quality(g, 13.0, ball(Z, Y, X, 6))
    assert q["coverage"] == pytest.approx(1.0)
    assert q["paddick_ci"] == pytest.approx(1.0)
    assert q["gradient_index"] == pytest.approx(1.0)


def test_paddick_index_on_a_dose_cloud_twice_the_tumour_volume():
    g, (Z, Y, X) = grid_with(lambda Z, Y, X: 26 * ball(Z, Y, X, 7.56))  # volume about twice a 6 mm ball
    target = ball(Z, Y, X, 6)
    q = plan_quality(g, 13.0, target)
    assert q["coverage"] == pytest.approx(1.0)
    assert q["paddick_ci"] == pytest.approx(target.sum() / ball(Z, Y, X, 7.56).sum(), rel=1e-6)
    assert 0.45 < q["paddick_ci"] < 0.55


def test_gradient_index_of_a_gaussian_dose_cloud():
    # Dose = 26 exp(-r^2 / 2 s^2): prescription 13 Gy at r1, 6.5 Gy at r2, GI = (r2/r1)^3.
    s = 5.0
    g, (Z, Y, X) = grid_with(lambda Z, Y, X: 26 * np.exp(-(Z**2 + Y**2 + X**2) / (2 * s**2)), n=61)
    q = plan_quality(g, 13.0, ball(Z, Y, X, 4))
    r1, r2 = s * np.sqrt(2 * np.log(2)), s * np.sqrt(2 * np.log(4))
    assert q["gradient_index"] == pytest.approx((r2 / r1) ** 3, rel=0.05)


def test_dvh_and_dose_at_volume():
    g, (Z, Y, X) = grid_with(lambda Z, Y, X: 10 + X)  # dose rises along x
    organ = ball(Z, Y, X, 5)
    bins, share = dvh(g.dose, organ, bins=np.array([0, 10, 30]))
    assert share[0] == pytest.approx(1.0)
    assert share[1] == pytest.approx(((10 + X) >= 10)[organ > 0].mean())
    assert share[2] == 0
    assert dose_at_volume(g.dose, organ, 0.01) == pytest.approx(15, abs=1.01)
    assert organ_dose(g, organ)["mean_gy"] == pytest.approx(10.0, abs=1e-5)


def test_gap_between_two_spheres_outlined_on_slices():
    def contours(cx, r):
        out = []
        for z in np.arange(-r, r + 0.01, 0.5):
            rr = np.sqrt(max(r**2 - z**2, 0))
            if rr > 0:
                a = np.linspace(0, 2 * np.pi, 200, endpoint=False)
                out.append((float(z), np.column_stack([cx + rr * np.cos(a), rr * np.sin(a)])))
        return out
    assert gap_mm(contours(0, 5), contours(14, 3)) == pytest.approx(6.0, abs=0.1)


def test_dice_identical_and_disjoint():
    a = np.zeros((5, 5, 5)); a[1:3] = 1
    b = np.zeros((5, 5, 5)); b[3:5] = 1
    assert dice(a, a) == 1 and dice(a, b) == 0


def test_lq_and_lql_known_values():
    # 13 Gy single fraction, alpha/beta 2: BED = 13 (1 + 13/2) = 97.5, EQD2 = 48.75.
    assert float(bed_lq(13, 1, 2.0)) == pytest.approx(97.5)
    assert float(eqd2(97.5, 2.0)) == pytest.approx(48.75)
    # 30 x 2 Gy has EQD2 = 60 whatever alpha/beta is.
    for ab in (2.0, 3.0, 10.0):
        assert float(eqd2(bed_lq(2, 30, ab), ab)) == pytest.approx(60.0)
    # LQ-L equals LQ below the transition and grows linearly with the LQ slope above it.
    assert float(bed_lql(4, 1, 2.0, 6.0)) == pytest.approx(float(bed_lq(4, 1, 2.0)))
    above = float(bed_lql(13, 1, 2.0, 6.0))
    assert above == pytest.approx(6 * (1 + 3) + (1 + 6) * 7)
    assert above < float(bed_lq(13, 1, 2.0))


def test_shifting_the_dose_moves_it_by_whole_voxels_exactly():
    g, (Z, Y, X) = grid_with(lambda Z, Y, X: 10 + X)
    region = (slice(10, 30), slice(10, 30), slice(10, 30))
    moved = shifted_dose(g, (0.0, 0.0, 2.0), region)
    assert np.allclose(moved, g.dose[region] - 2.0)


def test_monte_carlo_with_no_error_changes_nothing_and_errors_lower_coverage():
    s = 5.0
    g, (Z, Y, X) = grid_with(lambda Z, Y, X: 26 * np.exp(-(Z**2 + Y**2 + X**2) / (2 * s**2)), n=61)
    target = ball(Z, Y, X, 5.5)
    organ = ball(Z, Y, X, 1.5, c=(0, 0, 12))
    still = monte_carlo(g, 13.0, target, organ, sigma_mm=0.0, trials=5)
    base = float((target * (g.dose >= 13)).sum() / target.sum())
    assert np.allclose(still["coverage"], base)
    shaky = monte_carlo(g, 13.0, target, organ, sigma_mm=1.5, trials=100)
    assert shaky["coverage"].mean() < base
    assert shaky["shift_mm"].mean() == pytest.approx(1.5 * np.sqrt(8 / np.pi), rel=0.15)  # mean of a 3D normal's length
