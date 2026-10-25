"""Which patients the model learns from, which tune it, and which test it.

The test set is the 60 patients of data/patients.txt, drawn at random for
the dose analysis before any model existed. The other patients are split at
random (fixed seed) into training and validation. Every choice about the
model is made on validation; the test set is scored once, at the end.

    python -m radiosurgery.seg.splits    # writes data/splits.json
"""

import json
import random
from pathlib import Path

N_VAL = 20
SEED = 2026
FILE = Path("data") / "splits.json"


def make(all_patients, test_patients, n_val=N_VAL, seed=SEED):
    test = sorted(set(test_patients))
    missing = set(test) - set(all_patients)
    if missing:
        raise ValueError(f"test patients not in the collection: {sorted(missing)}")
    rest = sorted(set(all_patients) - set(test))
    val = sorted(random.Random(seed).sample(rest, n_val))
    train = sorted(set(rest) - set(val))
    return {"train": train, "val": val, "test": test}


def load():
    return json.loads(FILE.read_text())


def main():
    patients = sorted(p.name for p in (Path("data") / "raw").iterdir() if p.is_dir())
    test = (Path("data") / "patients.txt").read_text().split()
    s = make(patients, test)
    FILE.write_text(json.dumps(s, indent=1) + "\n")
    print({k: len(v) for k, v in s.items()})


if __name__ == "__main__":
    main()
