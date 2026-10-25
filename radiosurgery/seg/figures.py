"""Pictures of the AI outlines for the README.

    python -m radiosurgery.seg.figures     # after evaluate test and plans
"""

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ..mri import load_t1
from . import data
from .evaluate import RESULTS
from .plans import outlines

DOCTOR, AI = "#ef4444", "#d946ef"


def save(fig, name, dpi=120):
    fig.tight_layout()
    fig.savefig(RESULTS / name, dpi=dpi)
    plt.close(fig)
    print(f"  saved {RESULTS / name}")


def draw_case(ax, patient, row, chosen):
    m = data.meta(patient)
    _, structures = data.t1_outline(data.RAW / patient, m["series_uid"])
    doctor = structures[m["tumour_name"]]
    ai = outlines(patient, chosen)
    vol, x, y, z = load_t1(data.RAW / patient)
    areas = {}
    for zz, xy in doctor:  # the slice where the doctors' outline is largest
        areas[zz] = areas.get(zz, 0) + abs(np.dot(xy[:, 0], np.roll(xy[:, 1], 1)) - np.dot(xy[:, 1], np.roll(xy[:, 0], 1))) / 2
    zk = max(areas, key=areas.get)
    k = int(np.argmin(np.abs(z - zk)))
    pts = np.vstack([xy for _, xy in doctor])
    c = (pts.min(0) + pts.max(0)) / 2
    half = max(18.0, (pts.max(0) - pts.min(0)).max() / 2 + 10)
    lo, hi = np.percentile(vol[k], [1, 99.5])
    ax.imshow(vol[k], cmap="gray", vmin=lo, vmax=hi, extent=[x[0], x[-1], y[-1], y[0]])
    for zz, xy in doctor:
        if abs(zz - z[k]) < 0.3:
            ax.plot(*np.vstack([xy, xy[:1]]).T, color=DOCTOR, lw=1.8)
    for zz, xy in ai:
        if abs(zz - z[k]) < 0.3:
            ax.plot(*np.vstack([xy, xy[:1]]).T, color=AI, lw=1.8, ls=(0, (4, 2)))
    ax.set_xlim(c[0] - half, c[0] + half)
    ax.set_ylim(c[1] + half, c[1] - half)
    ax.set_xticks([])
    ax.set_yticks([])
    hd = "" if np.isnan(row["hd95_mm"]) else f", HD95 {row['hd95_mm']:.1f} mm"
    return f"{patient}: Dice {row['dice']:.3f}{hd}\ndoctor {row['doctor_cc']:.2f} cc, AI {row['ai_cc']:.2f} cc"


def cases(test, chosen):
    rows = sorted(test["per_patient"], key=lambda r: r["dice"])
    picks = [("Best", rows[-1]), ("Typical (median Dice)", rows[len(rows) // 2]), ("Worst", rows[0])]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.6))
    for ax, (label, row) in zip(axes, picks):
        ax.set_title(f"{label}\n" + draw_case(ax, row["patient"], row, chosen), fontsize=9)
    fig.legend(handles=[plt.Line2D([], [], color=DOCTOR, lw=1.8, label="Doctors' outline"),
                        plt.Line2D([], [], color=AI, lw=1.8, ls=(0, (4, 2)), label="AI outline")],
               loc="lower center", ncol=2, frameon=False)
    fig.suptitle("AI tumour outlines on test patients the model never saw (T1 MRI with contrast)")
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(RESULTS / "segmentation_cases.png", dpi=110)
    plt.close(fig)
    print(f"  saved {RESULTS / 'segmentation_cases.png'}")


def agreement(test):
    rows = test["per_patient"]
    ai = np.array([r["ai_cc"] for r in rows])
    doc = np.array([r["doctor_cc"] for r in rows])
    d = np.array([r["dice"] for r in rows])
    ba = test["volume_cc"]["bland_altman"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    ax = axes[0]
    ax.scatter(doc, d, s=20, color="#7c3aed")
    ax.set_xscale("log")
    ax.set_xlabel("Tumour volume, doctors' outline (cc, log scale)")
    ax.set_ylabel("Dice (AI against doctors)")
    ax.set_title("Small tumours are harder: the same edge error is a larger share")
    ax = axes[1]
    ax.scatter((ai + doc) / 2, ai - doc, s=20, color="#7c3aed")
    ax.axhline(ba["bias"], color="#1f2937", lw=1.2, label=f"Mean difference {ba['bias']:+.2f} cc")
    for lim in ba["limits"]:
        ax.axhline(lim, color="#6b7280", ls="--", lw=1)
    ax.plot([], [], color="#6b7280", ls="--", lw=1, label=f"95% limits {ba['limits'][0]:+.2f} to {ba['limits'][1]:+.2f} cc")
    ax.axhline(0, color="#d1d5db", lw=0.8)
    ax.set_xlabel("Mean of AI and doctors' volume (cc)")
    ax.set_ylabel("AI minus doctors' volume (cc)")
    ax.set_title("Volume agreement (Bland-Altman)")
    ax.legend(fontsize=8, frameon=False)
    save(fig, "segmentation_agreement.png")


def plan_scores(plans):
    rows = [r for r in plans["per_patient"] if not r.get("ai_missed")]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, key, label, scale in zip(axes, ["coverage", "paddick_ci", "gradient_index"],
                                     ["Coverage (%)", "Paddick conformity index", "Gradient index"], [100, 1, 1]):
        d = np.array([r[f"doctor_{key}"] for r in rows]) * scale
        a = np.array([r[f"ai_{key}"] for r in rows]) * scale
        lo, hi = min(d.min(), a.min()), max(d.max(), a.max())
        pad = (hi - lo) * 0.05
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color="#9ca3af", ls="--", lw=1)
        ax.scatter(d, a, s=18, color="#0d9488")
        ax.set_xlabel(f"{label}, doctors' outline")
        ax.set_ylabel(f"{label}, AI outline")
    fig.suptitle(f"The same {len(rows)} plans scored with the doctors' and the AI's tumour outline")
    save(fig, "segmentation_plans.png")


def main():
    test = json.loads((RESULTS / "segmentation_test.json").read_text())
    chosen = test["settings"]
    cases(test, chosen)
    agreement(test)
    plans = RESULTS / "segmentation_plans.json"
    if plans.exists():
        plan_scores(json.loads(plans.read_text()))


if __name__ == "__main__":
    main()
