"""The T1 MRI and the doctors' tumour outline drawn on it, ready for the network.

Each patient has two copies of the outlines, one drawn on the contrast T1
MRI and one on the T2 MRI. The model learns from, and is scored against,
the copy drawn on T1, the image it sees.

Every MRI is put on two fixed grids so that all patients look alike to the
network:

- coarse, 2 mm in every direction, to find roughly where the tumour is;
- fine, 0.5 mm in-plane and 1.5 mm between slices (close to the scans'
  own 0.41 x 0.41 x 1.5 mm), to outline it.

The labels are the share of each voxel inside the doctors' outline, from
masks.rasterise with sub-voxel sampling, so the boundary is not rounded to
whole voxels.

    python -m radiosurgery.seg.data      # all patients in data/raw, about 15 minutes
"""

import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pydicom
from scipy.ndimage import gaussian_filter

from ..dicomrt import DoseGrid, read_structures, referenced_series, tumour_name
from ..masks import rasterise
from ..mri import load_t1

RAW = Path("data") / "raw"
CACHE = Path("data") / "cache" / "seg"
FINE = (1.5, 0.5, 0.5)  # mm, (z, y, x)
COARSE = (2.0, 2.0, 2.0)


def geometry(z, y, x):
    """A grid with no dose, for rasterise: it only needs the axes and the shape."""
    return DoseGrid(np.broadcast_to(np.float32(0), (len(z), len(y), len(x))), np.asarray(x), np.asarray(y), np.asarray(z))


def axes_for(native, spacing):
    """Axes with the given spacing covering the same extent as the native axes."""
    return [a[0] + s * np.arange(int(np.floor((a[-1] - a[0]) / s + 1e-6)) + 1) for a, s in zip(native, spacing)]


def resample(vol, src, dst):
    """Linear interpolation from axes src (z, y, x) to axes dst, one axis at a time.

    Points outside the source take the nearest edge value.
    """
    out = np.asarray(vol, np.float32)
    for axis, (a, b) in enumerate(zip(src, dst)):
        pos = np.interp(b, a, np.arange(len(a)))
        i0 = np.minimum(np.floor(pos).astype(int), len(a) - 2) if len(a) > 1 else np.zeros(len(b), int)
        w = (pos - i0).astype(np.float32) if len(a) > 1 else np.zeros(len(b), np.float32)
        shape = [1, 1, 1]
        shape[axis] = len(b)
        w = w.reshape(shape)
        out = np.take(out, i0, axis=axis) * (1 - w) + np.take(out, np.minimum(i0 + 1, len(a) - 1), axis=axis) * w
    return out


def normalise(vol):
    """Clip the head's intensities to their 0.5 and 99.5 percentiles, then scale to zero mean, unit spread."""
    head = vol > vol.mean()
    lo, hi = np.percentile(vol[head], [0.5, 99.5])
    v = np.clip(vol, lo, hi)
    return ((v - v[head].mean()) / v[head].std()).astype(np.float32)


def t1_outline(patient_dir, series_uid):
    """The structure set drawn on this MRI series: (tumour name, all structures)."""
    patient_dir = Path(patient_dir)
    plans = [pydicom.dcmread(f) for f in sorted(patient_dir.glob("RTPLAN_*/*.dcm"))]
    for f in sorted(patient_dir.glob("RTSTRUCT_*/*.dcm")):
        ds = pydicom.dcmread(f)
        if referenced_series(ds) != series_uid:
            continue
        structures = read_structures(ds)
        target = ""
        for p in plans:
            if p.ReferencedStructureSetSequence[0].ReferencedSOPInstanceUID == ds.SOPInstanceUID:
                ref = p.DoseReferenceSequence[0]
                target = str(ref.get("DoseReferenceDescription", "")).split(":", 1)[-1].strip()
        return tumour_name(structures, target), structures
    raise LookupError(f"{patient_dir.name}: no structure set drawn on the T1 MRI")


def load_case(patient_dir):
    """Native T1 volume, its axes (z, y, x), the series UID, the tumour name and contours."""
    mri = load_t1(patient_dir, with_uid=True)
    if mri is None:
        raise LookupError(f"{Path(patient_dir).name}: no axial T1 MRI")
    vol, x, y, z, uid = mri
    name, structures = t1_outline(patient_dir, uid)
    if name is None or not structures.get(name):
        raise LookupError(f"{Path(patient_dir).name}: no tumour outline on the T1 MRI")
    return {"image": vol, "axes": (z, y, x), "uid": uid, "tumour": name, "contours": structures[name],
            "structures": structures}


def prepare(patient):
    out = CACHE / patient
    if (out / "meta.json").exists():
        return patient, "cached"
    try:
        case = load_case(RAW / patient)
    except LookupError as error:
        return patient, str(error)
    native = case["axes"]
    spacing = np.array([np.median(np.diff(a)) for a in native])
    img = normalise(case["image"])
    out.mkdir(parents=True, exist_ok=True)
    meta = {"patient": patient, "series_uid": case["uid"], "tumour_name": case["tumour"],
            "native_spacing": spacing.tolist(), "native_shape": list(img.shape),
            "native_axes": [np.round(a, 4).tolist() for a in native]}
    for kind, step in [("fine", FINE), ("coarse", COARSE)]:
        ax = axes_for(native, step)
        sigma = [0.4 * s / n if s / n > 1.5 else 0 for s, n in zip(step, spacing)]  # against aliasing
        src = gaussian_filter(img, sigma) if any(sigma) else img
        np.save(out / f"{kind}_img.npy", resample(src, native, ax).astype(np.float16))
        label = rasterise(case["contours"], geometry(*ax), sub=4 if kind == "fine" else 8)
        np.save(out / f"{kind}_lbl.npy", np.round(label * 255).astype(np.uint8))
        meta[f"{kind}_axes"] = [[float(a[0]), float(a[-1]), len(a)] for a in ax]
        meta[f"{kind}_tumour_cc"] = float(label.sum()) * float(np.prod(step)) / 1000
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    return patient, "prepared"


def axes_of(meta, kind):
    """The (z, y, x) axes of the native, fine or coarse grid stored in meta.json."""
    if kind == "native":
        return [np.array(a) for a in meta["native_axes"]]
    return [np.linspace(a, b, n) for a, b, n in meta[f"{kind}_axes"]]


def load_cached(patient, kind, mmap=True):
    """(image, label) on the fine or coarse grid."""
    mode = "r" if mmap else None
    return load_image(patient, kind, mmap), np.load(CACHE / patient / f"{kind}_lbl.npy", mmap_mode=mode)


def load_image(patient, kind, mmap=True):
    """The image alone: what the models see when they outline a patient."""
    return np.load(CACHE / patient / f"{kind}_img.npy", mmap_mode="r" if mmap else None)


def meta(patient):
    return json.loads((CACHE / patient / "meta.json").read_text())


def main():
    patients = sorted(p.name for p in RAW.iterdir() if p.is_dir())
    CACHE.mkdir(parents=True, exist_ok=True)
    failed = {}
    with ProcessPoolExecutor(int(sys.argv[1]) if len(sys.argv) > 1 else 6) as pool:
        for patient, status in pool.map(prepare, patients):
            print(patient, status, flush=True)
            if status not in ("cached", "prepared"):
                failed[patient] = status
    (CACHE / "failed.json").write_text(json.dumps(failed, indent=1))
    print(f"{len(patients) - len(failed)} of {len(patients)} patients ready")


if __name__ == "__main__":
    main()
