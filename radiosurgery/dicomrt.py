"""Reading a Gamma Knife treatment from DICOM-RT files.

Each patient folder can hold more than one copy of the plan, structure set
and dose (one per MRI sequence the plan was registered to). A consistent
set is one where the dose refers to the plan and the plan refers to the
structure set, all by their unique identifiers.
"""

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pydicom


@dataclass
class DoseGrid:
    dose: np.ndarray  # Gy, indexed [z, y, x], axes ascending
    x: np.ndarray  # mm, patient coordinates
    y: np.ndarray
    z: np.ndarray

    @property
    def spacing(self):
        return np.array([self.z[1] - self.z[0], self.y[1] - self.y[0], self.x[1] - self.x[0]])

    @property
    def voxel_cc(self):
        return float(np.prod(self.spacing)) / 1000.0


@dataclass
class Shot:
    name: str
    position: np.ndarray  # mm
    minutes: float


@dataclass
class Plan:
    prescription_gy: float
    target: str  # the structure the prescription refers to
    isodose_percent: float | None
    dose_rate: float  # Gy per minute, the machine's calibration dose rate
    shots: list = field(default_factory=list)


@dataclass
class Treatment:
    patient: str
    plan: Plan
    dose: DoseGrid
    structures: dict  # ROI name -> list of (z, (N, 2) array of x, y) in mm
    notes: list = field(default_factory=list)  # copies rejected on the way, and why


def read_dose(ds):
    if [round(float(v)) for v in ds.ImageOrientationPatient] != [1, 0, 0, 0, 1, 0]:
        raise ValueError("only axial dose grids in the standard orientation are supported")
    dose = ds.pixel_array.astype(np.float32) * float(ds.DoseGridScaling)
    origin = np.array(ds.ImagePositionPatient, float)
    row, col = (float(v) for v in ds.PixelSpacing)
    x = origin[0] + np.arange(ds.Columns) * col
    y = origin[1] + np.arange(ds.Rows) * row
    z = origin[2] + np.array(ds.GridFrameOffsetVector, float)
    if z[1] < z[0]:  # store with ascending z
        z, dose = z[::-1], dose[::-1]
    return DoseGrid(np.ascontiguousarray(dose), x, y, z.copy())


def read_plan(ds):
    ref = ds.DoseReferenceSequence[0]
    isodose = None
    for elem in ref:  # Leksell GammaPlan stores the prescription isodose in a private element
        if elem.tag.is_private and elem.VR in ("UN", "LO", "DS", "IS"):
            try:
                isodose = float(elem.value.decode() if isinstance(elem.value, bytes) else elem.value)
            except (ValueError, AttributeError):
                pass
    shots = [Shot(b.BeamName, np.array(b.ControlPointSequence[0].IsocenterPosition, float),
                  float(b.FinalCumulativeMetersetWeight)) for b in ds.BeamSequence]
    rate = float(ds.BeamSequence[0].ControlPointSequence[0].DoseRateSet)
    target = str(ref.get("DoseReferenceDescription", "")).split(":", 1)[-1].strip()
    return Plan(float(ref.TargetPrescriptionDose), target, isodose, rate, shots)


def read_structures(ds):
    names = {r.ROINumber: r.ROIName for r in ds.StructureSetROISequence}
    out = {}
    for roi in ds.ROIContourSequence:
        contours = []
        for c in roi.get("ContourSequence", []):
            if c.ContourGeometricType != "CLOSED_PLANAR":
                continue
            pts = np.array(c.ContourData, float).reshape(-1, 3)
            contours.append((float(np.round(pts[:, 2].mean(), 3)), pts[:, :2]))
        out[names[roi.ReferencedROINumber].strip()] = contours
    return out


def referenced_series(struct_ds):
    """The MRI series a structure set was drawn on."""
    frame = struct_ds.ReferencedFrameOfReferenceSequence[0]
    return frame.RTReferencedStudySequence[0].RTReferencedSeriesSequence[0].SeriesInstanceUID


def load(patient_dir, tolerance=0.05, on_series=None):
    """The consistent plan, structure set and dose that pass a sanity check.

    Some patient folders hold extra copies, and not all of them are sound:
    plans with no shots, or doses whose maximum does not fit the
    prescription. A Gamma Knife plan prescribes to an isodose line, a stated
    percentage of the maximum dose, so prescription / maximum dose must equal
    that percentage. A copy is accepted only if it agrees within
    `tolerance`. Rejected copies are listed in Treatment.notes.

    With `on_series`, only a copy whose structure set was drawn on that MRI
    series is accepted.
    """
    patient_dir = Path(patient_dir)
    read = lambda kind: [pydicom.dcmread(f) for f in sorted(patient_dir.glob(f"{kind}_*/*.dcm"))]
    plans = {p.SOPInstanceUID: p for p in read("RTPLAN")}
    structs = {s.SOPInstanceUID: s for s in read("RTSTRUCT")}
    notes = []
    for d in read("RTDOSE"):
        plan_ds = plans.get(d.ReferencedRTPlanSequence[0].ReferencedSOPInstanceUID)
        if plan_ds is None:
            notes.append("dose refers to a plan that is not present")
            continue
        if "BeamSequence" not in plan_ds:
            notes.append("plan copy has no shots")
            continue
        struct = structs.get(plan_ds.ReferencedStructureSetSequence[0].ReferencedSOPInstanceUID)
        if struct is None:
            notes.append("plan refers to a structure set that is not present")
            continue
        if on_series and referenced_series(struct) != on_series:
            notes.append("structure set drawn on another MRI series")
            continue
        plan, dose = read_plan(plan_ds), read_dose(d)
        if plan.isodose_percent:
            implied = 100 * plan.prescription_gy / float(dose.dose.max())
            if abs(implied / plan.isodose_percent - 1) > tolerance:
                notes.append(f"dose maximum {dose.dose.max():.2f} Gy does not fit {plan.prescription_gy:g} Gy "
                             f"at the {plan.isodose_percent:g}% isodose")
                continue
        t = Treatment(patient_dir.name, plan, dose, read_structures(struct))
        t.notes = notes
        return t
    raise LookupError(f"{patient_dir.name}: no sound plan, structure set and dose ({'; '.join(notes)})")


def find(structures, *keywords, exclude=()):
    """The structure whose name contains a keyword (case-insensitive)."""
    for name in structures:
        low = name.lower()
        if any(k in low for k in keywords) and not any(x in low for x in exclude):
            return name
    return None


def tumour_name(structures, plan_target=""):
    """The tumour outline: the structure the plan prescribes to, else one named like a target."""
    for name in structures:
        if name.lower() == plan_target.lower():
            return name
    return find(structures, "tv", "an", exclude=("cochlea",))
