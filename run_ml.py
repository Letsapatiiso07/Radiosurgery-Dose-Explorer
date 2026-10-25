"""The machine learning steps, in order. Each step can also be run alone.

    python -m radiosurgery.download --all    # once, about 21 GB (add --gcs if Amazon cannot be reached)
    python run_ml.py

Training takes about four and a half hours on an 8-core CPU and stops and resumes
cleanly: run the same command again. Model weights go to models/ and are
not in git; this script rebuilds them.
"""

import subprocess
import sys

STEPS = [
    ["radiosurgery.seg.data"],  # the T1 MRI on the model grids
    ["radiosurgery.seg.train", "coarse", "--threads", "12"],
    ["radiosurgery.seg.train", "fine", "--threads", "12"],
    ["radiosurgery.seg.evaluate", "val"],  # every choice, on the validation patients
    ["radiosurgery.seg.evaluate", "test"],  # once
    ["radiosurgery.seg.plans"],  # plan scores with the AI outline
    ["radiosurgery.seg.figures"],
    ["radiosurgery.cochlea_ml"],
]

if __name__ == "__main__":
    for step in STEPS:
        print(">", " ".join(step), flush=True)
        subprocess.run([sys.executable, "-m", *step], check=True)
    print("> python run.py  (to redraw the viewer with the AI outlines)")
