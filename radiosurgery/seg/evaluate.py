"""Score the AI tumour outlines against the doctors' outlines on the T1 MRI.

    python -m radiosurgery.seg.evaluate val     # every choice, on the validation patients
    python -m radiosurgery.seg.evaluate test    # once, with the settings validation chose

All scores are on the MRI's own grid (0.41 x 0.41 x 1-1.5 mm). The doctors'
outline is the set of voxels at least half inside their contours.

- Dice: twice the overlap over the sum of the two volumes; 1 is a perfect
  match. It punishes the same 1 mm error far more on a small tumour.
- HD95: the 95th percentile of the distance from each outline's surface to
  the other's, the larger of the two directions, in mm. It shows how far the
  edge strays, ignoring the worst 5% of points.
- Volume: AI against doctor, by Bland-Altman.
- Found: the AI outline overlaps the tumour at all.
"""

import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy import ndimage

from .. import stats
from ..masks import rasterise
from . import data, splits
from .infer import Settings, outline, to_native
from .train import MODELS, build

RESULTS = Path("results")


def surface(mask):
    return mask & ~ndimage.binary_erosion(mask, structure=ndimage.generate_binary_structure(3, 1))


def hd95(a, b, spacing):
    """Symmetric 95th percentile surface distance in mm; NaN if either mask is empty."""
    if not a.any() or not b.any():
        return float("nan")
    nz = np.argwhere(a | b)
    box = tuple(slice(max(lo - 3, 0), hi + 4) for lo, hi in zip(nz.min(0), nz.max(0)))
    a, b = a[box], b[box]
    sa, sb = surface(a), surface(b)
    da = ndimage.distance_transform_edt(~sb, sampling=spacing)[sa]
    db = ndimage.distance_transform_edt(~sa, sampling=spacing)[sb]
    return float(max(np.percentile(da, 95), np.percentile(db, 95)))


def dice(a, b):
    s = a.sum() + b.sum()
    return float(2 * (a & b).sum() / s) if s else 1.0


def keep_largest(mask):
    lab, n = ndimage.label(mask)
    if n <= 1:
        return mask
    sizes = ndimage.sum(mask, lab, index=np.arange(1, n + 1))
    return lab == (1 + int(np.argmax(sizes)))


def truth(patient):
    """The doctors' outline on the native grid: voxels at least half inside, and the voxel volume in cc."""
    m = data.meta(patient)
    axes = data.axes_of(m, "native")
    _, structures = data.t1_outline(data.RAW / patient, m["series_uid"])
    frac = rasterise(structures[m["tumour_name"]], data.geometry(*axes))
    return frac >= 0.5, float(np.prod(m["native_spacing"])) / 1000


def load_models(coarse="best", fine="best"):
    models = []
    for stage, which in [("coarse", coarse), ("fine", fine)]:
        m = build(stage)
        m.load_state_dict(torch.load(MODELS / stage / f"{which}.pt", weights_only=False)["model"])
        models.append(m.eval())
    return models


def predict(patient, models, settings):
    """AI probability on the native grid, inside a box: (probability, box)."""
    m = data.meta(patient)
    ci = data.load_image(patient, "coarse", mmap=False)
    fi = data.load_image(patient, "fine")
    fa = data.axes_of(m, "fine")
    prob, box = outline(*models, ci.astype(np.float32), fi, data.axes_of(m, "coarse"), fa, settings)
    return to_native(prob, box, fa, data.axes_of(m, "native"))


def ai_mask(prob, box, shape, threshold, largest):
    mask = np.zeros(shape, bool)
    mask[box] = prob >= threshold
    return keep_largest(mask) if largest else mask


def score(pred, true, spacing, voxel_cc):
    return {"dice": dice(pred, true), "hd95_mm": hd95(pred, true, spacing), "found": bool((pred & true).any()),
            "ai_cc": float(pred.sum()) * voxel_cc, "doctor_cc": float(true.sum()) * voxel_cc}


THRESHOLDS = (0.3, 0.4, 0.5, 0.6, 0.7)


def validation():
    """Try each choice on the validation patients; keep the one with the best mean Dice."""
    patients = [p for p in splits.load()["val"] if (data.CACHE / p / "meta.json").exists()]
    truths = {p: truth(p) for p in patients}
    trials = []
    for fine_ckpt, mirror, cands in itertools.product(["best", "last"], [False, True], [1, 2]):
        models = load_models("best", fine_ckpt)
        settings = Settings(mirror=mirror, candidates=cands)
        start = time.time()
        probs = {p: predict(p, models, settings) for p in patients}
        took = (time.time() - start) / len(patients)
        for threshold, largest in itertools.product(THRESHOLDS, [False, True]):
            rows = []
            for p in patients:
                true, cc = truths[p]
                spacing = data.meta(p)["native_spacing"]
                rows.append(score(ai_mask(*probs[p], true.shape, threshold, largest), true, spacing, cc))
            trial = {"fine_checkpoint": fine_ckpt, "mirror": mirror, "candidates": cands, "threshold": threshold,
                     "largest_only": largest, "mean_dice": float(np.mean([r["dice"] for r in rows])),
                     "median_hd95_mm": float(np.nanmedian([r["hd95_mm"] for r in rows])),
                     "found": int(sum(r["found"] for r in rows)), "seconds_per_patient": took}
            trials.append(trial)
            print(json.dumps(trial), flush=True)
    # The best mean Dice; among near-ties (within 0.001), the cheaper, simpler setting.
    top = max(t["mean_dice"] for t in trials)
    near = [t for t in trials if t["mean_dice"] >= top - 0.001]
    chosen = min(near, key=lambda t: (t["mirror"], t["candidates"], not t["largest_only"], abs(t["threshold"] - 0.5)))
    out = {"patients": patients, "trials": trials, "chosen": chosen}
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "segmentation_validation.json").write_text(json.dumps(out, indent=1))
    print("chosen:", json.dumps(chosen))


def interval(values, fn=np.mean):
    v = np.asarray(values, float)
    return [float(x) for x in stats.bootstrap(lambda i: fn(v[i]), len(v))]


def test():
    chosen = json.loads((RESULTS / "segmentation_validation.json").read_text())["chosen"]
    settings = Settings(mirror=chosen["mirror"], candidates=chosen["candidates"])
    models = load_models("best", chosen["fine_checkpoint"])
    s = splits.load()
    out_dir = MODELS / "test_probs"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows, missing = [], []
    for p in s["test"]:
        if not (data.CACHE / p / "meta.json").exists():
            missing.append(p)
            continue
        m = data.meta(p)
        prob, box = predict(p, models, settings)
        np.savez_compressed(out_dir / f"{p}.npz", prob=prob.astype(np.float16),
                            box=np.array([[b.start, b.stop] for b in box]))
        true, cc = truth(p)
        pred = ai_mask(prob, box, true.shape, chosen["threshold"], chosen["largest_only"])
        rows.append({"patient": p, "tumour_name": m["tumour_name"],
                     **score(pred, true, m["native_spacing"], cc)})
        print(json.dumps(rows[-1]), flush=True)
    d = np.array([r["dice"] for r in rows])
    found = [r for r in rows if r["found"]]
    h = np.array([r["hd95_mm"] for r in found])
    ai, doc = np.array([r["ai_cc"] for r in rows]), np.array([r["doctor_cc"] for r in rows])
    k = len(found)
    summary = {
        "settings": chosen, "patients": len(rows), "missing": missing,
        "dice": {"mean": float(d.mean()), "mean_ci": interval(d), "sd": float(d.std(ddof=1)),
                 "median": float(np.median(d)), "median_ci": interval(d, np.median), **stats.summary(d)},
        "hd95_mm": {"median": float(np.median(h)), "median_ci": interval(h, np.median), "mean": float(h.mean()),
                    "mean_ci": interval(h), **stats.summary(h), "n": len(h)},
        "volume_cc": {"bland_altman": stats.bland_altman(ai, doc),
                      "relative_error_pct_median": float(np.median(100 * (ai - doc) / doc)),
                      "relative_error_pct_ci": interval(100 * (ai - doc) / doc, np.median)},
        "found": {"n": k, "of": len(rows), "ci": list(stats.clopper_pearson(k, len(rows)))},
        "per_patient": rows,
    }
    (RESULTS / "segmentation_test.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: v for k, v in summary.items() if k != "per_patient"}, indent=1))


if __name__ == "__main__":
    torch.set_num_threads(12)
    {"val": validation, "test": test}[sys.argv[1]]()
