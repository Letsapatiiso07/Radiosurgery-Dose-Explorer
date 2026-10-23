"""Measure every sampled Gamma Knife plan, then summarise the cohort.

    python -m radiosurgery.download     # once, about 1.7 GB
    python run.py
"""

import json
import zlib
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from radiosurgery import stats
from radiosurgery.dicomrt import DoseGrid, find, load
from radiosurgery.masks import rasterise
from radiosurgery.metrics import dice, dvh, gap_mm, organ_dose, plan_quality
from radiosurgery.mri import load_t1
from radiosurgery.radiobiology import bed_lq, bed_lql, eqd2, single_fraction_table
from radiosurgery.robustness import monte_carlo

RAW = Path("data") / "raw"
OUT = Path("results")
DOCS = Path("docs")
COCHLEA_LIMIT_GY = 4.0  # mean cochlea dose linked to better hearing preservation in several series
SIGMAS = (0.5, 1.0)  # mm: frame-based setup, and a looser mask or registration setup
TRIALS = 200


def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / name, dpi=130)
    plt.close(fig)
    print(f"  saved {OUT / name}")


def target_name(treatment, plan_target):
    for name in treatment.structures:
        if name.lower() == plan_target.lower():
            return name
    return find(treatment.structures, "tv", "an", exclude=("cochlea",))


def analyse(patient):
    try:
        tr = load(RAW / patient)
    except LookupError as error:
        return None, str(error)
    tname = target_name(tr, tr.plan.target)
    if tname is None:
        return None, f"{patient}: no target outline"
    if tname.lower().startswith("men"):
        return None, f"{patient}: target is a meningioma ('{tname}'), not a schwannoma"
    g, rx = tr.dose, tr.plan.prescription_gy
    target = rasterise(tr.structures[tname], g)
    row = {"patient": patient, "target_name": tname, "prescription_gy": rx, "rejected_copies": tr.notes,
           "stated_isodose_pct": tr.plan.isodose_percent, "shots": len(tr.plan.shots),
           "beam_on_min": float(sum(s.minutes for s in tr.plan.shots)), "dose_rate": tr.plan.dose_rate}
    row.update(plan_quality(g, rx, target))
    row["isodose_pct_from_dose"] = 100 * rx / row["max_gy"]
    # The exported grid misses the true peak between voxels. Rescale the dose
    # so its maximum is the one the plan implies (prescription / stated
    # isodose) and measure again: a sensitivity check, not a correction.
    if tr.plan.isodose_percent:
        scale = (rx / (tr.plan.isodose_percent / 100)) / row["max_gy"]
        rescaled = plan_quality(DoseGrid(g.dose * scale, g.x, g.y, g.z), rx, target)
        row["dose_scale"] = scale
        row["coverage_rescaled"] = rescaled["coverage"]
        row["paddick_ci_rescaled"] = rescaled["paddick_ci"]
    cname = find(tr.structures, "cochlea")
    cochlea = rasterise(tr.structures[cname], g) if cname else None
    if cochlea is not None and cochlea.sum() > 0:
        row.update({f"cochlea_{k}": v for k, v in organ_dose(g, cochlea).items()})
        row["gap_mm"] = gap_mm(tr.structures[tname], tr.structures[cname])
        for sigma in SIGMAS:
            mc = monte_carlo(g, rx, target, cochlea, sigma, trials=TRIALS, seed=zlib.crc32(patient.encode()))
            row[f"mc{sigma}_coverage_p5"] = float(np.quantile(mc["coverage"], 0.05))
            row[f"mc{sigma}_coverage_drop_p5"] = row["coverage"] - row[f"mc{sigma}_coverage_p5"]
            row[f"mc{sigma}_coverage_mean"] = float(mc["coverage"].mean())
            row[f"mc{sigma}_cochlea_p95"] = float(np.quantile(mc["organ_mean"], 0.95))
    iso = find(tr.structures, f"t1+ {rx:g}")
    if iso:
        iso_frac = rasterise(tr.structures[iso], g)
        row["tps_isodose_cc"] = float(iso_frac.sum()) * g.voxel_cc
        row["tps_isodose_dice"] = dice(iso_frac, (g.dose >= rx).astype(np.float32))
    row["_dvh"] = {"target": dvh(g.dose, target)[1].tolist()}
    if cochlea is not None and cochlea.sum() > 0:
        row["_dvh"]["cochlea"] = dvh(g.dose, cochlea, bins=dvh(g.dose, target)[0])[1].tolist()
    row["_dvh"]["dose"] = dvh(g.dose, target)[0].tolist()
    return (row, tr, target, cochlea, tname, cname), None


CONVERGENCE = ["VS-SEG-003", "VS-SEG-020", "VS-SEG-104", "VS-SEG-162", "VS-SEG-235"]


def refine(grid, factor, near, margin=10):
    """The dose on a grid `factor` times finer, by trilinear interpolation, around `near`."""
    from scipy.ndimage import map_coordinates
    nz = np.argwhere(near > 0)
    lo = np.maximum(nz.min(0) - margin, 0)
    hi = np.minimum(nz.max(0) + margin, np.array(grid.dose.shape) - 1)
    axes = [np.arange(lo[i], hi[i] + 1e-9, 1 / factor) for i in range(3)]
    d = map_coordinates(grid.dose, np.meshgrid(*axes, indexing="ij"), order=1)
    at = lambda a, v: np.interp(v, np.arange(len(a)), a)
    return DoseGrid(d.astype(np.float32), at(grid.x, axes[2]), at(grid.y, axes[1]), at(grid.z, axes[0]))


def convergence(cases):
    """Coverage and conformity on 1, 0.5 and 0.25 mm grids: do they settle?"""
    out = []
    for row, tr, target, cochlea, tname, cname in cases:
        rx = row["prescription_gy"]
        entry = {"patient": row["patient"]}
        for f in (1, 2, 4):
            fg = refine(tr.dose, f, target)
            tm = rasterise(tr.structures[tname], fg, sub=4 if f < 4 else 2)
            q = plan_quality(fg, rx, tm)
            entry[f"{1 / f:g}mm"] = {"coverage": q["coverage"], "paddick_ci": q["paddick_ci"], "tumour_cc": q["tumour_cc"]}
        out.append(entry)
    return out


def chart_example(case):
    row, tr, target, cochlea, tname, cname = case
    patient = row["patient"]
    mri = load_t1(RAW / patient)
    g = tr.dose
    k = int(np.argmax(target.sum(axis=(1, 2))))
    z = g.z[k]
    fig, ax = plt.subplots(figsize=(6.5, 6))
    if mri is not None:
        vol, mx, my, mz = mri
        sl = vol[int(np.argmin(np.abs(mz - z)))]
        lo, hi = np.percentile(sl, [1, 99.5])
        ax.imshow(sl, cmap="gray", vmin=lo, vmax=hi, extent=[mx[0], mx[-1], my[-1], my[0]])
    levels = [row["prescription_gy"] / 2, row["prescription_gy"], 0.8 * row["max_gy"]]
    cs = ax.contour(g.x, g.y, g.dose[k], levels=levels, colors=["#60a5fa", "#facc15", "#f97316"], linewidths=1.4)
    ax.clabel(cs, fmt=lambda v: f"{v:.1f} Gy", fontsize=7)
    for name, color in [(tname, "#ef4444"), (cname, "#22d3ee")]:
        if name:
            for zz, xy in tr.structures[name]:
                if abs(zz - z) <= 0.75:
                    ax.plot(*np.vstack([xy, xy[:1]]).T, color=color, lw=1.6)
    cx, cy = np.average(g.x, weights=target[k].sum(0) + 1e-9), np.average(g.y, weights=target[k].sum(1) + 1e-9)
    ax.set_xlim(cx - 35, cx + 35)
    ax.set_ylim(cy + 35, cy - 35)
    ax.set_xlabel("mm")
    ax.set_title(f"{patient}: tumour (red), cochlea (cyan), isodose lines\n"
                 f"{row['prescription_gy']:g} Gy prescription (yellow), half of it (blue), 80% of maximum (orange)", fontsize=9)
    save(fig, "example_slice.png")


def chart_dvh(case):
    row = case[0]
    d = row["_dvh"]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(d["dose"], np.array(d["target"]) * 100, color="#ef4444", label="Tumour")
    if "cochlea" in d:
        ax.plot(d["dose"], np.array(d["cochlea"]) * 100, color="#0891b2", label="Cochlea")
    ax.axvline(row["prescription_gy"], color="#a16207", ls="--", lw=1, label="Prescription")
    ax.set_xlabel("Dose (Gy)")
    ax.set_ylabel("Share of structure receiving at least this dose (%)")
    ax.set_title(f"Dose-volume histogram, {row['patient']}")
    ax.legend()
    save(fig, "dvh_example.png")


def chart_cochlea(rows, fit):
    r = [x for x in rows if "cochlea_mean_gy" in x]
    gap = np.array([x["gap_mm"] for x in r])
    mean = np.array([x["cochlea_mean_gy"] for x in r])
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.scatter(gap, mean, color="#0891b2", s=28, zorder=3)
    xs = np.linspace(0, gap.max() + 1, 100)
    ax.plot(xs, np.exp(fit["a"] + fit["b"] * xs), color="#1f2937", lw=1.5, label=f"Fit: dose x {fit['factor_per_mm']:.2f} per mm")
    ax.axhline(COCHLEA_LIMIT_GY, color="#dc2626", ls="--", lw=1, label=f"{COCHLEA_LIMIT_GY:g} Gy mean dose")
    ax.set_xlabel("Closest distance from tumour to cochlea (mm)")
    ax.set_ylabel("Mean cochlea dose (Gy)")
    ax.set_title("The anatomy sets the hearing risk: cochlea dose against distance")
    ax.legend()
    save(fig, "cochlea_vs_distance.png")


def chart_quality(rows):
    tv = np.array([x["tumour_cc"] for x in rows])
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    for ax, key, label in zip(axes, ["paddick_ci", "gradient_index", "shots"],
                              ["Paddick conformity index", "Gradient index", "Number of shots"]):
        ax.scatter(tv, [x[key] for x in rows], s=20, color="#7c3aed")
        ax.set_xscale("log")
        ax.set_xlabel("Tumour volume (cc, log scale)")
        ax.set_ylabel(label)
    fig.suptitle("Plan quality across the sample")
    save(fig, "quality_vs_volume.png")


def chart_robustness(rows):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    r = [x for x in rows if "mc0.5_coverage_p5" in x]
    for sigma, color in zip(SIGMAS, ["#16a34a", "#dc2626"]):
        axes[0].hist([100 * x[f"mc{sigma}_coverage_drop_p5"] for x in r], bins=15, alpha=0.6, color=color,
                     label=f"{sigma} mm per axis")
        axes[1].scatter([x["cochlea_mean_gy"] for x in r], [x[f"mc{sigma}_cochlea_p95"] for x in r], s=18, color=color,
                        label=f"{sigma} mm per axis")
    axes[0].set_xlabel("Tumour coverage lost in the worst 5% of trials (percentage points)")
    axes[0].set_ylabel("Patients")
    lim = max(x["cochlea_mean_gy"] for x in r) * 1.3
    axes[1].plot([0, lim], [0, lim], color="#9ca3af", ls="--")
    axes[1].set_xlabel("Planned mean cochlea dose (Gy)")
    axes[1].set_ylabel("Mean cochlea dose, worst 5% of trials (Gy)")
    for ax in axes:
        ax.legend(fontsize=8)
    fig.suptitle(f"What positioning errors do, {TRIALS} random shifts per patient")
    save(fig, "robustness.png")


def chart_radiobiology():
    d = np.linspace(0, 25, 200)
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for ab, color in [(2.0, "#b45309"), (3.0, "#2563eb")]:
        ax.plot(d, eqd2(bed_lq(d, 1, ab), ab), color=color, label=f"LQ, alpha/beta {ab:g} Gy")
        ax.plot(d, eqd2(bed_lql(d, 1, ab, 6.0), ab), color=color, ls="--", label=f"LQ-L (transition 6 Gy), alpha/beta {ab:g} Gy")
    ax.axvline(13, color="#6b7280", ls=":", lw=1)
    ax.set_xlabel("Single-session dose (Gy)")
    ax.set_ylabel("Equivalent dose in 2 Gy fractions, EQD2 (Gy)")
    ax.set_title("Why one session of 13 Gy is a large dose")
    ax.legend(fontsize=8)
    save(fig, "radiobiology.png")


def export_viewer(cases):
    """Slice images with outlines and isodose lines, plus numbers, for docs/."""
    (DOCS / "img").mkdir(parents=True, exist_ok=True)
    index = []
    for case in cases:
        row, tr, target, cochlea, tname, cname = case
        mri = load_t1(RAW / row["patient"])
        if mri is None:
            continue
        vol, mx, my, mz = mri
        g = tr.dose
        ks = np.flatnonzero(target.sum(axis=(1, 2)) > 0)
        if cochlea is not None:
            ks = np.union1d(ks, np.flatnonzero(cochlea.sum(axis=(1, 2)) > 0))
        ks = np.arange(max(ks.min() - 4, 0), min(ks.max() + 5, len(g.z)), 2)
        cx = np.average(g.x, weights=target.sum((0, 1)) + 1e-9)
        cy = np.average(g.y, weights=target.sum((0, 2)) + 1e-9)
        files = []
        for k in ks:
            z = g.z[k]
            fig = plt.figure(figsize=(4, 4), dpi=110)
            ax = fig.add_axes([0, 0, 1, 1])
            sl = vol[int(np.argmin(np.abs(mz - z)))]
            lo, hi = np.percentile(sl, [1, 99.5])
            ax.imshow(sl, cmap="gray", vmin=lo, vmax=hi, extent=[mx[0], mx[-1], my[-1], my[0]])
            if g.dose[k].max() >= row["prescription_gy"] / 2:
                ax.contour(g.x, g.y, g.dose[k], levels=[row["prescription_gy"] / 2, row["prescription_gy"]],
                           colors=["#60a5fa", "#facc15"], linewidths=1.3)
            for name, color in [(tname, "#ef4444"), (cname, "#22d3ee")]:
                if name:
                    for zz, xy in tr.structures[name]:
                        if abs(zz - z) <= 0.75:
                            ax.plot(*np.vstack([xy, xy[:1]]).T, color=color, lw=1.5)
            ax.set_xlim(cx - 30, cx + 30)
            ax.set_ylim(cy + 30, cy - 30)
            ax.axis("off")
            name = f"{row['patient']}_{len(files):02d}.png"
            fig.savefig(DOCS / "img" / name)
            plt.close(fig)
            files.append({"file": f"img/{name}", "z_mm": round(float(z), 1)})
        keep = ["patient", "prescription_gy", "tumour_cc", "coverage", "paddick_ci", "gradient_index",
                "shots", "beam_on_min", "cochlea_mean_gy", "cochlea_max_gy", "gap_mm", "max_gy"]
        index.append({**{k: row.get(k) for k in keep}, "slices": files, "dvh": row["_dvh"]})
    (DOCS / "data.json").write_text(json.dumps(index))
    print(f"  saved {len(index)} patients to {DOCS / 'data.json'}")


def main():
    start = time.time()
    OUT.mkdir(exist_ok=True)
    patients = (Path("data") / "patients.txt").read_text().split()
    cases, skipped = [], []
    for p in patients:
        case, why = analyse(p)
        if case:
            cases.append(case)
        else:
            skipped.append(why)
            print("  skipped:", why)
    rows = [c[0] for c in cases]
    print(f"{len(rows)} plans measured in {time.time() - start:.0f} s")

    co = [r for r in rows if "cochlea_mean_gy" in r]
    below = sum(r["cochlea_mean_gy"] < COCHLEA_LIMIT_GY for r in co)
    fit = stats.exponential_falloff([r["gap_mm"] for r in co], [r["cochlea_mean_gy"] for r in co])
    tv = [r["tumour_cc"] for r in rows]
    tests = {k: stats.spearman(tv, [r[k] for r in rows]) for k in ["paddick_ci", "gradient_index", "shots", "beam_on_min"]}
    adj = stats.holm({k: v["p"] for k, v in tests.items()})
    for k in tests:
        tests[k]["p_holm"] = adj[k]
    iso_check = stats.bland_altman([r["isodose_pct_from_dose"] for r in rows if r["stated_isodose_pct"]],
                                   [r["stated_isodose_pct"] for r in rows if r["stated_isodose_pct"]])
    results = {
        "patients_sampled": len(patients), "plans_measured": len(rows), "skipped": skipped,
        "rejected_copies": {r["patient"]: r["rejected_copies"] for r in rows if r["rejected_copies"]},
        "with_cochlea": len(co),
        "prescriptions": {f"{v:g} Gy": sum(r["prescription_gy"] == v for r in rows) for v in sorted({r["prescription_gy"] for r in rows})},
        "summary": {k: stats.summary([r[k] for r in rows if k in r]) for k in
                    ["tumour_cc", "coverage", "coverage_rescaled", "paddick_ci", "paddick_ci_rescaled", "gradient_index",
                     "shots", "beam_on_min", "v12_outside_tumour_cc", "max_gy", "dose_scale"]},
        "grid_convergence": convergence([c for c in cases if c[0]["patient"] in CONVERGENCE]),
        "cochlea": {"mean_gy": stats.summary([r["cochlea_mean_gy"] for r in co]),
                    "max_gy": stats.summary([r["cochlea_max_gy"] for r in co]),
                    "gap_mm": stats.summary([r["gap_mm"] for r in co]),
                    "below_limit": below, "below_limit_share": below / len(co),
                    "below_limit_ci": stats.clopper_pearson(below, len(co)),
                    "distance_fit": fit,
                    "spearman_dose_vs_gap": stats.spearman([r["gap_mm"] for r in co], [r["cochlea_mean_gy"] for r in co])},
        "volume_correlations": tests,
        "isodose_check": iso_check,
        "tps_isodose": [{k: r[k] for k in ("patient", "piv_cc", "tps_isodose_cc", "tps_isodose_dice")} for r in rows if "tps_isodose_cc" in r],
        "robustness": {f"{s} mm": {
            "coverage_p5": stats.summary([r[f"mc{s}_coverage_p5"] for r in co]),
            "coverage_drop_p5": stats.summary([r[f"mc{s}_coverage_drop_p5"] for r in co]),
            "cochlea_p95_minus_planned": stats.summary([r[f"mc{s}_cochlea_p95"] - r["cochlea_mean_gy"] for r in co]),
        } for s in SIGMAS},
        "radiobiology_13gy": single_fraction_table(13.0),
        "radiobiology_12gy": single_fraction_table(12.0),
        "per_patient": [{k: v for k, v in r.items() if not k.startswith("_")} for r in rows],
    }
    (OUT / "metrics.json").write_text(json.dumps(results, indent=1, default=float))

    print("Charts")
    example = next(c for c in cases if c[0]["patient"] in (Path("data") / "patients.txt").read_text().split()[:5] and c[3] is not None)
    chart_example(example)
    chart_dvh(example)
    chart_cochlea(rows, fit)
    chart_quality(rows)
    chart_robustness(rows)
    chart_radiobiology()
    with_mri = (Path("data") / "patients.txt").read_text().split()[:5]
    export_viewer([c for c in cases if c[0]["patient"] in with_mri])
    print(f"Done in {time.time() - start:.0f} s")


if __name__ == "__main__":
    main()
