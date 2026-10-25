"""The planning MRI: for pictures, and as the input of the tumour outlining model.

Every dose analysis number comes from the dose grid.
"""

from pathlib import Path

import numpy as np
import pydicom


def load_t1(patient_dir, with_uid=False):
    """Axial T1 slices sorted by height: (volume [z, y, x], x, y, z) in mm.

    With `with_uid`, the series' unique identifier is returned as well, to
    find the structure set drawn on this MRI.
    """
    for series in sorted(Path(patient_dir).glob("MR_*")):
        files = [pydicom.dcmread(f) for f in series.glob("*.dcm")]
        if not files or not str(files[0].SeriesDescription).lower().startswith("t1"):
            continue
        files.sort(key=lambda d: float(d.ImagePositionPatient[2]))
        first = files[0]
        if [round(float(v)) for v in first.ImageOrientationPatient] != [1, 0, 0, 0, 1, 0]:
            continue
        vol = np.stack([f.pixel_array.astype(np.float32) * float(getattr(f, "RescaleSlope", 1))
                        + float(getattr(f, "RescaleIntercept", 0)) for f in files])
        row, col = (float(v) for v in first.PixelSpacing)
        x = float(first.ImagePositionPatient[0]) + np.arange(first.Columns) * col
        y = float(first.ImagePositionPatient[1]) + np.arange(first.Rows) * row
        z = np.array([float(f.ImagePositionPatient[2]) for f in files])
        return (vol, x, y, z, str(first.SeriesInstanceUID)) if with_uid else (vol, x, y, z)
    return None
