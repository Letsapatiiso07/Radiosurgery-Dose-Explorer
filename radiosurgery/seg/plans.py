"""Would the plan scores change if the AI had drawn the tumour?

    python -m radiosurgery.seg.plans     # after evaluate test

For each test patient in the dose analysis, the AI's tumour probability is
turned into outlines, slice by slice, the way a doctor's outline is stored.
Both outlines then go through the same steps on the same dose grid
(masks.rasterise, then metrics.plan_quality), so any change in coverage,
conformity or gradient index comes from the outline alone.

The doctors' outline here is the one drawn on T1, the image the AI reads.
The dose is the copy registered to T1 where it passes the sanity check of
dicomrt.load, otherwise the other copy (they share one coordinate frame).
"""

import json
from pathlib import Path

import numpy as np
from contourpy import contour_generator
from scipy import ndimage

from .. import stats
from ..dicomrt import load
from ..masks import rasterise
from ..metrics import dice, plan_quality
from . import data
from .evaluate import RESULTS, ai_mask
from .train import MODELS

KEYS = ["tumour_cc", "coverage", "paddick_ci", "gradient_index"]
VIEWER = Path("docs") / "ai_outlines.json"


def ai_contours(prob, box, axes, threshold, largest):
    """The AI outline as closed polygons per MRI slice: [(z, (N, 2) x, y)], like dicomrt.read_structures."""
    shape = tuple(len(a) for a in axes)
    keep = ai_mask(prob, box, shape, threshold, largest)[box]
    p = np.where(ndimage.binary_dilation(keep), prob, 0)  # drop the regions post-processing removed
    p = np.pad(p, 1)  # so every outline closes
    z, y, x = (np.pad(a[s], 1, mode="reflect", reflect_type="odd") for a, s in zip(axes, box))
    out = []
    for k in range(1, len(z) - 1):
        if p[k].max() < threshold:
            continue
        for line in contour_generator(x=x, y=y, z=p[k]).lines(threshold):
            if len(line) >= 4:
                out.append((float(np.round(z[k], 3)), np.asarray(line)))
    return out


def outlines(patient, chosen):
    m = data.meta(patient)
    f = np.load(MODELS / "test_probs" / f"{patient}.npz")
    box = tuple(slice(int(a), int(b)) for a, b in f["box"])
    return ai_contours(f["prob"].astype(np.float32), box, data.axes_of(m, "native"), chosen["threshold"],
                       chosen["largest_only"])


def main():
    chosen = json.loads((RESULTS / "segmentation_validation.json").read_text())["chosen"]
    dose_rows = json.loads((RESULTS / "metrics.json").read_text())["per_patient"]
    rows, viewer = [], {}
    viewer_patients = (Path("data") / "patients.txt").read_text().split()[:5]
    for r in dose_rows:
        p = r["patient"]
        m = data.meta(p)
        try:
            tr, copy = load(data.RAW / p, on_series=m["series_uid"]), "T1"
        except LookupError:
            tr, copy = load(data.RAW / p), "T2 (T1 copy unsound)"
        _, structures = data.t1_outline(data.RAW / p, m["series_uid"])
        doctor = structures[m["tumour_name"]]
        ai = outlines(p, chosen)
        g, rx = tr.dose, tr.plan.prescription_gy
        d_frac, a_frac = rasterise(doctor, g), rasterise(ai, g)
        row = {"patient": p, "dose_copy": copy, "prescription_gy": rx, "dice_on_dose_grid": dice(d_frac, a_frac)}
        if a_frac.sum() == 0:
            row["ai_missed"] = True
            rows.append(row)
            continue
        qd, qa = plan_quality(g, rx, d_frac), plan_quality(g, rx, a_frac)
        for k in KEYS:
            row[f"doctor_{k}"], row[f"ai_{k}"] = qd[k], qa[k]
        rows.append(row)
        if p in viewer_patients:
            viewer[p] = [[z, np.round(xy, 1).tolist()] for z, xy in ai]
        print(json.dumps(row), flush=True)
    scored = [r for r in rows if not r.get("ai_missed")]
    summary = {"patients": len(rows), "ai_missed": [r["patient"] for r in rows if r.get("ai_missed")],
               "dose_copies": {c: sum(r["dose_copy"] == c for r in rows) for c in sorted({r["dose_copy"] for r in rows})}}
    for k in KEYS:
        a = np.array([r[f"ai_{k}"] for r in scored])
        d = np.array([r[f"doctor_{k}"] for r in scored])
        diff = a - d
        worst = sorted(scored, key=lambda r: -abs(r[f"ai_{k}"] - r[f"doctor_{k}"]))[:3]
        summary[k] = {"doctor": stats.summary(d), "ai": stats.summary(a),
                      "difference_median": float(np.median(diff)),
                      "difference_median_ci": list(stats.bootstrap(lambda i: np.median(diff[i]), len(diff))),
                      "difference_q1_q3": [float(np.quantile(diff, 0.25)), float(np.quantile(diff, 0.75))],
                      "bland_altman": stats.bland_altman(a, d),
                      "worst": [{"patient": w["patient"], "doctor": w[f"doctor_{k}"], "ai": w[f"ai_{k}"]} for w in worst]}
    cov = np.array([r["ai_coverage"] - r["doctor_coverage"] for r in scored])
    vol = np.array([r["ai_tumour_cc"] - r["doctor_tumour_cc"] for r in scored])
    summary["coverage_vs_volume_spearman"] = stats.spearman(vol, cov)
    summary["per_patient"] = rows
    (RESULTS / "segmentation_plans.json").write_text(json.dumps(summary, indent=1))
    VIEWER.write_text(json.dumps(viewer, separators=(",", ":")))
    print(json.dumps({k: v for k, v in summary.items() if k != "per_patient"}, indent=1))


if __name__ == "__main__":
    main()
