import json
from pathlib import Path

import numpy as np
import pytest
import torch

from radiosurgery.masks import rasterise, volume_cc
from radiosurgery.seg import splits
from radiosurgery.seg.data import axes_for, geometry, resample
from radiosurgery.seg.evaluate import dice, hd95, keep_largest
from radiosurgery.seg.infer import candidates, sliding_window
from radiosurgery.seg.unet import UNet


def test_splits_share_no_patient():
    everyone = [f"P{i:03d}" for i in range(242)]
    test = everyone[::4][:60]
    s = splits.make(everyone, test)
    assert len(s["test"]) == 60 and len(s["val"]) == splits.N_VAL
    assert not set(s["train"]) & set(s["val"])
    assert not set(s["train"]) & set(s["test"])
    assert not set(s["val"]) & set(s["test"])
    assert sorted(s["train"] + s["val"] + s["test"]) == everyone


def test_saved_splits_share_no_patient_and_test_is_the_dose_sample():
    s = json.loads(splits.FILE.read_text())
    groups = [set(s[k]) for k in ("train", "val", "test")]
    assert sum(len(g) for g in groups) == len(set().union(*groups))
    assert s["test"] == sorted(Path("data/patients.txt").read_text().split())


def circle(r, n=200):
    a = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.column_stack([r * np.cos(a), r * np.sin(a)])


def test_rasterise_on_an_mri_grid():
    # A 6 mm sphere drawn on the MRI's own 1.5 mm slices, measured on its 0.41 mm pixels.
    z = np.arange(-15, 15.1, 1.5)
    xy = -20 + 0.41015625 * np.arange(100)
    contours = [(float(zz), circle(np.sqrt(36 - zz**2))) for zz in z if 36 - zz**2 > 0.01]
    g = geometry(z, xy, xy)
    v = volume_cc(rasterise(contours, g), g)
    drawn = sum(np.pi * (36 - zz**2) * 1.5 for zz in z if 36 - zz**2 > 0.01) / 1000  # each slice stands for 1.5 mm
    assert v == pytest.approx(drawn, rel=0.01)


def test_resampling_a_linear_ramp_is_exact():
    src = [np.arange(10) * 1.5, np.arange(20) * 0.4, np.arange(20) * 0.4]
    Z, Y, X = np.meshgrid(*src, indexing="ij")
    vol = 2 * Z + 3 * Y - X
    dst = axes_for(src, (1.0, 0.5, 0.5))
    out = resample(vol, src, dst)
    Z2, Y2, X2 = np.meshgrid(*dst, indexing="ij")
    assert out.shape == (len(dst[0]), len(dst[1]), len(dst[2]))
    assert np.allclose(out, 2 * Z2 + 3 * Y2 - X2, atol=1e-4)


def test_unet_keeps_the_patch_shape():
    torch.manual_seed(0)
    net = UNet((4, 8, 16), ((1, 3, 3), (3, 3, 3), (3, 3, 3)), ((1, 2, 2), (2, 2, 2)))
    assert net.divisor == (2, 4, 4)
    out = net(torch.randn(2, 1, 8, 32, 32))
    assert out.shape == (2, 1, 8, 32, 32)


def test_sliding_window_averages_to_the_model_output():
    class Constant(torch.nn.Module):
        def forward(self, x):
            return torch.zeros_like(x)  # probability 0.5 everywhere
    p = sliding_window(Constant(), np.random.rand(13, 50, 41), (8, 32, 32), mirror=True)
    assert p.shape == (13, 50, 41) and np.allclose(p, 0.5)


def cube(lo, hi, n=30):
    m = np.zeros((n, n, n), bool)
    m[lo:hi, lo:hi, lo:hi] = True
    return m


def test_dice_and_hd95_of_shifted_cubes():
    a = cube(5, 15)
    b = np.roll(a, 2, axis=2)  # moved 2 voxels along x
    assert dice(a, a) == 1.0
    assert dice(a, b) == pytest.approx(0.8)
    assert hd95(a, a, (1, 1, 1)) == 0.0
    assert hd95(a, b, (1, 1, 1)) == pytest.approx(2.0)
    assert hd95(a, b, (1, 1, 0.5)) == pytest.approx(1.0)  # distances are in mm
    assert np.isnan(hd95(a, np.zeros_like(a), (1, 1, 1)))


def test_keep_largest_and_candidates():
    m = cube(2, 10) | cube(20, 22)
    assert keep_largest(m).sum() == 8**3
    prob = np.zeros((30, 30, 30), np.float32)
    prob[2:4, 2:4, 2:4] = 0.9  # small but sure
    prob[20:26, 20:26, 20:26] = 0.6  # larger: more summed probability
    best = candidates(prob)[0]
    assert [s.start for s in best] == [20, 20, 20]


def test_ai_probability_becomes_outlines_with_the_same_volume():
    from radiosurgery.seg.plans import ai_contours
    axes = [np.arange(30) * 1.5, np.arange(80) * 0.5, np.arange(80) * 0.5]
    Z, Y, X = np.meshgrid(*axes, indexing="ij")
    r = np.sqrt((Z - 22) ** 2 + (Y - 20) ** 2 + (X - 20) ** 2)
    prob = np.clip(8 - r, 0, 1).astype(np.float32)  # a ball of radius 7.5 mm at probability 0.5
    box = (slice(0, 30), slice(0, 80), slice(0, 80))
    contours = ai_contours(prob, box, axes, 0.5, largest=True)
    g = geometry(*axes)
    v = volume_cc(rasterise(contours, g), g)
    assert v == pytest.approx(((prob >= 0.5).sum()) * 1.5 * 0.25 / 1000, rel=0.03)
    assert all(abs(z / 1.5 - round(z / 1.5)) < 1e-6 for z, _ in contours)  # on the MRI's own slices
