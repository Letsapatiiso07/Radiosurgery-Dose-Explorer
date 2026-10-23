"""The DICOM-RT reader on small hand-built datasets."""

import numpy as np
import pytest
from pydicom.dataset import Dataset
from pydicom.sequence import Sequence

from radiosurgery.dicomrt import read_dose, read_plan, read_structures


def dose_dataset(values, offsets, scaling=0.001):
    ds = Dataset()
    ds.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
    ds.ImagePositionPatient = [-10.0, -20.0, 5.0]
    ds.PixelSpacing = [2.0, 1.0]  # row spacing (y), column spacing (x)
    ds.GridFrameOffsetVector = offsets
    ds.DoseGridScaling = scaling
    frames, rows, cols = values.shape
    ds.NumberOfFrames, ds.Rows, ds.Columns = frames, rows, cols
    ds.BitsAllocated, ds.BitsStored, ds.HighBit, ds.PixelRepresentation = 16, 16, 15, 0
    ds.SamplesPerPixel, ds.PhotometricInterpretation = 1, "MONOCHROME2"
    ds.PixelData = values.astype("<u2").tobytes()
    ds.file_meta = Dataset()
    ds.file_meta.TransferSyntaxUID = "1.2.840.10008.1.2.1"
    return ds


def test_dose_scaling_axes_and_descending_frames():
    values = np.arange(2 * 3 * 4, dtype=np.uint16).reshape(2, 3, 4) * 1000
    g = read_dose(dose_dataset(values, [0.0, -1.5]))  # frames stored top-down
    assert np.allclose(g.x, [-10, -9, -8, -7])
    assert np.allclose(g.y, [-20, -18, -16])
    assert np.allclose(g.z, [3.5, 5.0])  # sorted upwards
    assert g.dose[1, 0, 0] == pytest.approx(0.0)  # the frame at z = 5.0 is the first one stored
    assert g.dose[0, 2, 3] == pytest.approx(values[1, 2, 3] * 0.001)
    assert g.voxel_cc == pytest.approx(1.5 * 2 * 1 / 1000)


def test_rotated_grids_are_refused():
    ds = dose_dataset(np.zeros((2, 2, 2), np.uint16), [0.0, 1.0])
    ds.ImageOrientationPatient = [0, 1, 0, 1, 0, 0]
    with pytest.raises(ValueError):
        read_dose(ds)


def test_plan_reads_prescription_isodose_and_shots():
    ref = Dataset()
    ref.TargetPrescriptionDose = 13
    ref.DoseReferenceDescription = "A:TV"
    ref.add_new(0x30090010, "LO", "VENDOR")
    ref.add_new(0x30091001, "UN", b"50")  # the private isodose element
    plan = Dataset()
    plan.DoseReferenceSequence = Sequence([ref])
    beams = []
    for name, pos, minutes in [("A1", [1.0, 2.0, 3.0], 2.5), ("A2", [4.0, 5.0, 6.0], 1.25)]:
        cp = Dataset()
        cp.IsocenterPosition = pos
        cp.DoseRateSet = 2.62
        b = Dataset()
        b.BeamName, b.FinalCumulativeMetersetWeight = name, minutes
        b.ControlPointSequence = Sequence([cp])
        beams.append(b)
    plan.BeamSequence = Sequence(beams)
    p = read_plan(plan)
    assert p.prescription_gy == 13 and p.target == "TV" and p.isodose_percent == 50
    assert p.dose_rate == pytest.approx(2.62)
    assert [s.minutes for s in p.shots] == [2.5, 1.25]
    assert np.allclose(p.shots[1].position, [4, 5, 6])


def test_structures_keep_closed_planar_contours_only():
    roi = Dataset()
    roi.ROINumber, roi.ROIName = 1, " TV "
    contour = Dataset()
    contour.ContourGeometricType = "CLOSED_PLANAR"
    contour.ContourData = [0, 0, 7.5, 1, 0, 7.5, 1, 1, 7.5]
    point = Dataset()
    point.ContourGeometricType = "POINT"
    point.ContourData = [0, 0, 7.5]
    rc = Dataset()
    rc.ReferencedROINumber = 1
    rc.ContourSequence = Sequence([contour, point])
    ds = Dataset()
    ds.StructureSetROISequence = Sequence([roi])
    ds.ROIContourSequence = Sequence([rc])
    s = read_structures(ds)
    assert list(s) == ["TV"]
    assert len(s["TV"]) == 1 and s["TV"][0][0] == 7.5
    assert s["TV"][0][1].shape == (3, 2)
