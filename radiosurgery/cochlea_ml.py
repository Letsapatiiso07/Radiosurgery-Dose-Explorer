"""Predict the cochlea's mean dose from the anatomy, before the plan exists.

    python -m radiosurgery.cochlea_ml     # every patient with a cochlea outline, a few minutes

Every patient in the collection whose plan passes the sanity check of
dicomrt.load and whose cochlea was outlined gives one row. Plans prescribed
outside 10 to 15 Gy are left out: one plan prescribes 100 Gy, which is not a
schwannoma dose. The features come
only from the doctors' outlines and the prescribed dose, all known before
planning starts; nothing is read from the dose grid or the plan's shots.

The target is the mean cochlea dose as a share of the prescription. Gamma
Knife plans scale with the prescription (the prescription isodose wraps the
tumour), so dividing by it puts 12 Gy and 13 Gy plans on one scale.
Predictions are multiplied back by the prescription and scored in Gy.

Models, from simplest to most flexible:

- baseline: log(dose) = a + b x gap, the one-line fit of stats.exponential_falloff;
- ridge regression on the log of the dose;
- gradient boosting and a random forest on the log of the dose.

Each is scored by cross-validation, 5 folds repeated 10 times: the model is
fitted on four fifths of the patients and predicts the fifth it has never
seen. Each patient is one row, so the folds are grouped by patient. Settings
(regularisation, tree depth) are chosen inside each training fold by a
further 5-fold search, so the held-out fifth never steers them.
"""

import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
from scipy import stats as sstats
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV, KFold, RepeatedStratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from . import stats
from .dicomrt import find, load, tumour_name
from .masks import rasterise
from .metrics import gap_mm, organ_dose
from .seg.data import geometry

RAW = Path("data") / "raw"
OUT = Path("results")
LIMIT_GY = 4.0
RX_RANGE = (10, 15)  # Gy; one plan in the collection prescribes 100 Gy, not a schwannoma dose
FOLDS, REPEATS, SEED = 5, 10, 2026
FEATURES = ["tumour_cc", "cochlea_cc", "gap_mm", "centroid_mm", "tumour_within_5mm_cc", "tumour_within_10mm_cc",
            "prescription_gy"]


def features(tumour, cochlea, prescription_gy, step=0.5):
    """Anatomy features from two outlines (lists of (z, xy) contours) and the prescribed dose.

    - tumour_cc, cochlea_cc: volumes.
    - gap_mm: closest distance between the outlines.
    - centroid_mm: distance between their centres.
    - tumour_within_5mm_cc, tumour_within_10mm_cc: how much tumour lies within 5
      and 10 mm of the cochlea, a simple measure of how far it fills the
      internal auditory canal toward the cochlea.
    """
    pts = np.vstack([np.column_stack([xy, np.full(len(xy), z)]) for z, xy in tumour + cochlea])
    lo, hi = pts.min(0) - 2, pts.max(0) + 2
    x, y, z = (np.arange(lo[i], hi[i] + step, step) for i in range(3))
    grid = geometry(z, y, x)
    t, c = rasterise(tumour, grid), rasterise(cochlea, grid)
    cc = step**3 / 1000
    zz, yy, xx = np.meshgrid(z, y, x, indexing="ij")
    centre = lambda f: np.array([(f * a).sum() / f.sum() for a in (xx, yy, zz)])
    dist = ndimage.distance_transform_edt(c < 0.5, sampling=step)
    return {"tumour_cc": float(t.sum()) * cc, "cochlea_cc": float(c.sum()) * cc,
            "gap_mm": gap_mm(tumour, cochlea), "centroid_mm": float(np.linalg.norm(centre(t) - centre(c))),
            "tumour_within_5mm_cc": float(t[dist <= 5].sum()) * cc, "tumour_within_10mm_cc": float(t[dist <= 10].sum()) * cc,
            "prescription_gy": float(prescription_gy)}


def treatment_features(tr):
    """Features of a loaded treatment, from its outlines and prescription only.

    Returns None if there is no schwannoma or cochlea outline.
    """
    tname = tumour_name(tr.structures, tr.plan.target)
    cname = find(tr.structures, "cochlea")
    if tname is None or tname.lower().startswith("men") or cname is None or not tr.structures[cname]:
        return None
    return features(tr.structures[tname], tr.structures[cname], tr.plan.prescription_gy)


def patient_row(patient, t1_series=None):
    """Features and the mean cochlea dose for one patient, or the reason there is none."""
    try:
        tr = load(RAW / patient, on_series=t1_series) if t1_series else load(RAW / patient)
        copy = "T1" if t1_series else "first sound copy"
    except LookupError:
        try:  # the copy drawn on T1 failed the sanity check: use the other one
            tr, copy = load(RAW / patient), "T2 (T1 copy unsound)"
        except LookupError as error:
            return None, str(error)
    if not RX_RANGE[0] <= tr.plan.prescription_gy <= RX_RANGE[1]:
        return None, f"{patient}: prescribed {tr.plan.prescription_gy:g} Gy, outside the {RX_RANGE[0]}-{RX_RANGE[1]} Gy used for schwannomas"
    feats = treatment_features(tr)
    if feats is None:
        return None, f"{patient}: no schwannoma and cochlea outlines"
    cochlea = rasterise(tr.structures[find(tr.structures, "cochlea")], tr.dose)
    if cochlea.sum() == 0:
        return None, f"{patient}: cochlea outside the dose grid"
    return {"patient": patient, "copy": copy, **feats, "cochlea_mean_gy": organ_dose(tr.dose, cochlea)["mean_gy"]}, None


def collect():
    from .seg import data
    rows, skipped = [], []
    for patient in sorted(p.name for p in RAW.iterdir() if p.is_dir()):
        meta = data.CACHE / patient / "meta.json"
        uid = json.loads(meta.read_text())["series_uid"] if meta.exists() else None
        row, why = patient_row(patient, uid)
        if row:
            rows.append(row)
        else:
            skipped.append(why)
    return rows, skipped


class Baseline:
    """The distance-only exponential fit, as a model."""

    def fit(self, X, y):
        f = stats.exponential_falloff(X[:, FEATURES.index("gap_mm")], np.exp(y))
        self.a, self.b = f["a"], f["b"]
        return self

    def predict(self, X):
        return self.a + self.b * X[:, FEATURES.index("gap_mm")]


def design(rows):
    X = np.array([[r[k] for k in FEATURES] for r in rows], float)
    rel = np.array([r["cochlea_mean_gy"] / r["prescription_gy"] for r in rows])
    return X, np.log(rel), np.array([r["prescription_gy"] for r in rows])


def log_volumes(X):
    """For the linear model: volumes on a log scale, distances as they are."""
    X = X.copy()
    for k in ("tumour_cc", "cochlea_cc"):
        X[:, FEATURES.index(k)] = np.log(X[:, FEATURES.index(k)])
    for k in ("tumour_within_5mm_cc", "tumour_within_10mm_cc"):
        X[:, FEATURES.index(k)] = np.log1p(10 * X[:, FEATURES.index(k)])
    return X


def models():
    inner = KFold(5, shuffle=True, random_state=SEED)
    search = lambda est, grid: GridSearchCV(est, grid, cv=inner, scoring="neg_mean_absolute_error")
    return {
        "baseline (gap only)": lambda: Baseline(),
        "ridge": lambda: search(make_pipeline(FunctionTransformer(log_volumes), StandardScaler(), Ridge()),
                                {"ridge__alpha": [0.1, 1, 10, 100]}),
        "gradient boosting": lambda: search(GradientBoostingRegressor(subsample=0.8, random_state=SEED),
                                            {"n_estimators": [100, 300], "max_depth": [2, 3], "learning_rate": [0.03, 0.1]}),
        "random forest": lambda: search(RandomForestRegressor(n_estimators=300, random_state=SEED),
                                        {"min_samples_leaf": [2, 5], "max_features": [0.33, 0.66, 1.0]}),
    }


def cross_validate(rows):
    patients = [r["patient"] for r in rows]
    assert len(set(patients)) == len(patients), "one row per patient, so folds are grouped by patient"
    X, y, rx = design(rows)
    gy = np.exp(y) * rx
    preds = {name: np.zeros((REPEATS, len(y))) for name in models()}
    for rep in range(REPEATS):
        for train, test in KFold(FOLDS, shuffle=True, random_state=SEED + rep).split(X):
            for name, make in models().items():
                m = make().fit(X[train], y[train])
                preds[name][rep, test] = np.exp(m.predict(X[test])) * rx[test]
    out = {}
    base_err = np.abs(preds["baseline (gap only)"] - gy).mean(0)
    for name, p in preds.items():
        abs_err = np.abs(p - gy)  # [repeat, patient]
        sq = ((p - gy) ** 2).mean(0)
        per_patient = abs_err.mean(0)
        r2 = lambda i: 1 - sq[i].mean() / gy[i].var()
        diff = per_patient - base_err
        out[name] = {
            "mae_gy": float(abs_err.mean()), "mae_ci": list(stats.bootstrap(lambda i: per_patient[i].mean(), len(y))),
            "r2": float(np.mean([1 - ((p[r] - gy) ** 2).mean() / gy.var() for r in range(REPEATS)])),
            "r2_ci": list(stats.bootstrap(r2, len(y))),
            "mae_minus_baseline_gy": float(diff.mean()),
            "mae_minus_baseline_ci": list(stats.bootstrap(lambda i: diff[i].mean(), len(y))),
            "wilcoxon_p_vs_baseline": float(sstats.wilcoxon(per_patient, base_err).pvalue) if name != "baseline (gap only)" else None,
        }
    return out, {k: v.mean(0) for k, v in preds.items()}, gy


def effects(rows):
    """Ridge fitted on every patient: % change in cochlea dose per standard deviation of each (log) feature."""
    X, y, _ = design(rows)
    m = models()["ridge"]().fit(X, y)
    coef = m.best_estimator_.named_steps["ridge"].coef_
    return {k: float(100 * (np.exp(c) - 1)) for k, c in zip(FEATURES, coef)}


def classify(rows):
    """Will the mean cochlea dose reach 4 Gy? AUC of a logistic model and boosting, against the gap alone."""
    X, y, rx = design(rows)
    at_risk = (np.exp(y) * rx >= LIMIT_GY).astype(int)
    names = ["gap only", "logistic", "gradient boosting"]
    pred = {k: np.zeros((REPEATS, len(y))) for k in names}
    cv = RepeatedStratifiedKFold(n_splits=FOLDS, n_repeats=REPEATS, random_state=SEED)
    for i, (train, test) in enumerate(cv.split(X, at_risk)):
        rep = i // FOLDS
        logit = GridSearchCV(make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
                             {"logisticregression__C": [0.1, 1, 10]}, cv=5, scoring="roc_auc").fit(log_volumes(X[train]), at_risk[train])
        gb = GridSearchCV(GradientBoostingClassifier(subsample=0.8, random_state=SEED),
                          {"n_estimators": [100, 300], "max_depth": [2, 3], "learning_rate": [0.03, 0.1]},
                          cv=5, scoring="roc_auc").fit(X[train], at_risk[train])
        pred["gap only"][rep, test] = -X[test, FEATURES.index("gap_mm")]
        pred["logistic"][rep, test] = logit.predict_proba(log_volumes(X[test]))[:, 1]
        pred["gradient boosting"][rep, test] = gb.predict_proba(X[test])[:, 1]
    out = {"at_risk": int(at_risk.sum()), "of": int(len(at_risk))}
    for k in names:
        auc = lambda i: (np.mean([roc_auc_score(at_risk[i], p[i]) for p in pred[k]])
                         if 0 < at_risk[i].sum() < len(i) else np.nan)
        out[k] = {"auc": float(auc(np.arange(len(y)))), "auc_ci": list(stats.bootstrap(auc, len(y), n_boot=2000))}
    return out


def chart(oof, gy, rows, cv):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    lim = max(gy.max(), max(v.max() for v in oof.values())) * 1.08
    for ax, name, color in zip(axes, ["baseline (gap only)", min((k for k in cv if k != "baseline (gap only)"), key=lambda k: cv[k]["mae_gy"])],
                               ["#0d9488", "#7c3aed"]):
        ax.scatter(gy, oof[name], s=18, color=color, alpha=0.8)
        ax.plot([0, lim], [0, lim], color="#9ca3af", ls="--", lw=1)
        ax.axhline(LIMIT_GY, color="#dc2626", ls=":", lw=1)
        ax.axvline(LIMIT_GY, color="#dc2626", ls=":", lw=1)
        ax.set_xlim(0, lim)
        ax.set_ylim(0, lim)
        ax.set_xlabel("Mean cochlea dose in the plan (Gy)")
        ax.set_ylabel("Predicted before planning (Gy)")
        ax.set_title(f"{name}: MAE {cv[name]['mae_gy']:.2f} Gy, R² {cv[name]['r2']:.2f}", fontsize=10)
    fig.suptitle(f"Cochlea dose predicted from anatomy, {len(rows)} patients, cross-validated")
    fig.tight_layout()
    fig.savefig(OUT / "cochlea_ml.png", dpi=120)
    plt.close(fig)


def main():
    start = time.time()
    rows, skipped = collect()
    print(f"{len(rows)} patients with a cochlea outline and a sound plan ({len(skipped)} skipped) in {time.time() - start:.0f} s")
    cv, oof, gy = cross_validate(rows)
    for k, v in cv.items():
        print(k, json.dumps(v))
    cls = classify(rows)
    print(json.dumps(cls))
    fit = stats.exponential_falloff([r["gap_mm"] for r in rows], [r["cochlea_mean_gy"] for r in rows])
    out = {"patients": len(rows), "skipped": skipped, "features": FEATURES, "folds": FOLDS, "repeats": REPEATS,
           "copies": {c: sum(r["copy"] == c for r in rows) for c in sorted({r["copy"] for r in rows})},
           "prescriptions": {f"{v:g} Gy": sum(r["prescription_gy"] == v for r in rows) for v in sorted({r["prescription_gy"] for r in rows})},
           "cochlea_mean_gy": stats.summary(gy), "distance_fit_all": fit, "regression": cv, "classifier_4gy": cls,
           "ridge_percent_per_sd": effects(rows),
           "per_patient": [{**r, **{f"pred_{k}": float(v[i]) for k, v in oof.items()}} for i, r in enumerate(rows)]}
    OUT.mkdir(exist_ok=True)
    (OUT / "cochlea_ml.json").write_text(json.dumps(out, indent=1))
    chart(oof, gy, rows, cv)
    print(f"done in {time.time() - start:.0f} s")


if __name__ == "__main__":
    main()
