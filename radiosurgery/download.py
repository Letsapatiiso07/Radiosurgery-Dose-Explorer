"""Fetch the treatment files for a random sample of patients.

Vestibular-Schwannoma-SEG (Shapey et al., 2021, Scientific Data 8:286;
TCIA DOI 10.7937/TCIA.9YTJ-5Q73) holds 242 patients treated with Gamma Knife
radiosurgery at King's College Hospital, London, under the CC BY 4.0
licence. It is served as open data by the NCI Imaging Data Commons.

For every sampled patient: the plan, the structure set and the dose. For a
few: the T1 MRI used for planning, for pictures and the web viewer.

With --all: the plan, structure set, dose and contrast T1 MRI for every
patient in the collection (about 21 GB), for the machine learning steps.
Files already on disk are skipped. Add --gcs to fetch from the Google Cloud
copy of the data instead of the Amazon one, if Amazon cannot be reached.
"""

import random
import sys
from pathlib import Path

from idc_index import IDCClient

COLLECTION = "vestibular_schwannoma_seg"
SAMPLE = 60
WITH_MRI = 5
SEED = 2026
RAW = Path("data") / "raw"


def choose(client):
    patients = sorted(client.sql_query(
        f"SELECT DISTINCT PatientID FROM index WHERE collection_id = '{COLLECTION}'")["PatientID"])
    rng = random.Random(SEED)
    sample = sorted(rng.sample(patients, SAMPLE))
    return sample, sample[:WITH_MRI]


def series_for(client, patients, modalities, description_like=None):
    ids = ", ".join(f"'{p}'" for p in patients)
    mods = ", ".join(f"'{m}'" for m in modalities)
    where = f"collection_id = '{COLLECTION}' AND PatientID IN ({ids}) AND Modality IN ({mods})"
    if description_like:
        where += f" AND SeriesDescription LIKE '{description_like}'"
    return client.sql_query(f"SELECT SeriesInstanceUID FROM index WHERE {where}")["SeriesInstanceUID"].tolist()


def fetch_all(client, bucket="aws"):
    patients = sorted(client.sql_query(
        f"SELECT DISTINCT PatientID FROM index WHERE collection_id = '{COLLECTION}'")["PatientID"])
    rt = series_for(client, patients, ["RTPLAN", "RTSTRUCT", "RTDOSE"])
    mri = series_for(client, patients, ["MR"], "t1%")
    print(f"{len(patients)} patients, {len(rt)} treatment series, {len(mri)} MRI series", flush=True)
    RAW.mkdir(parents=True, exist_ok=True)
    client.download_from_selection(seriesInstanceUID=rt + mri, downloadDir=str(RAW), use_s5cmd_sync=True,
                                   source_bucket_location=bucket, dirTemplate="%PatientID/%Modality_%SeriesInstanceUID", quiet=True)
    print("done", flush=True)


def main():
    client = IDCClient()
    if "--all" in sys.argv[1:]:
        return fetch_all(client, "gcs" if "--gcs" in sys.argv[1:] else "aws")
    sample, with_mri = choose(client)
    (Path("data") / "patients.txt").write_text("\n".join(sample) + "\n")
    rt = series_for(client, sample, ["RTPLAN", "RTSTRUCT", "RTDOSE"])
    mri = series_for(client, with_mri, ["MR"], "t1%")
    print(f"{len(sample)} patients, {len(rt)} treatment series, {len(mri)} MRI series", flush=True)
    RAW.mkdir(parents=True, exist_ok=True)
    client.download_from_selection(seriesInstanceUID=rt + mri, downloadDir=str(RAW),
                                   dirTemplate="%PatientID/%Modality_%SeriesInstanceUID", quiet=True)
    print("done", flush=True)


if __name__ == "__main__":
    sys.exit(main())
