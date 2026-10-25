import numpy as np
import pytest

from radiosurgery.cochlea_ml import FEATURES, features, treatment_features
from radiosurgery.dicomrt import Plan, Treatment


def sphere(radius, cx=0.0, spacing=1.0):
    out = []
    for z in np.arange(-radius, radius + 1e-9, spacing):
        r = np.sqrt(max(radius**2 - z**2, 0))
        if r > 0.05:
            a = np.linspace(0, 2 * np.pi, 120, endpoint=False)
            out.append((float(z), np.column_stack([cx + r * np.cos(a), r * np.sin(a)])))
    return out


def test_features_of_two_spheres():
    tumour, cochlea = sphere(8.0), sphere(2.0, cx=14.0)
    f = features(tumour, cochlea, 13.0)
    assert list(f) == FEATURES
    assert f["gap_mm"] == pytest.approx(4.0, abs=0.15)
    assert f["centroid_mm"] == pytest.approx(14.0, abs=0.1)
    assert f["tumour_cc"] == pytest.approx(4 / 3 * np.pi * 8**3 / 1000, rel=0.03)
    assert f["cochlea_cc"] == pytest.approx(4 / 3 * np.pi * 2**3 / 1000, rel=0.15)
    assert 0 < f["tumour_within_5mm_cc"] < f["tumour_within_10mm_cc"] < f["tumour_cc"]
    assert f["prescription_gy"] == 13.0


class Forbidden:
    """Stands in for anything known only after planning: any use of it fails."""

    def __getattr__(self, name):
        raise AssertionError(f"the feature builder read '{name}' of a planning output")

    def __array__(self, *args, **kwargs):
        raise AssertionError("the feature builder read a planning output")

    def __float__(self):
        raise AssertionError("the feature builder read a planning output")

    def __iter__(self):
        raise AssertionError("the feature builder read a planning output")

    def __len__(self):
        raise AssertionError("the feature builder read a planning output")


def test_feature_builder_uses_no_dose_information():
    tumour, cochlea = sphere(6.0), sphere(2.0, cx=11.0)
    plan = Plan(prescription_gy=12.0, target="TV", isodose_percent=Forbidden(), dose_rate=Forbidden(), shots=Forbidden())
    tr = Treatment("P1", plan, Forbidden(), {"TV": tumour, "Cochlea": cochlea, "*Skull": []})
    assert treatment_features(tr) == features(tumour, cochlea, 12.0)


def test_no_cochlea_no_features():
    plan = Plan(prescription_gy=13.0, target="TV", isodose_percent=None, dose_rate=0.0)
    tr = Treatment("P1", plan, Forbidden(), {"TV": sphere(5.0)})
    assert treatment_features(tr) is None
