#!/usr/bin/env python3
"""Audit HFO detector-candidate vs. doctor-annotation counts for every omni_ieeg recording.

`foundry.data.pipelines.omni_ieeg._load_hfo_events` walks a per-detector pointer into `derivatives/hfo/<recording>.csv` in the same order as the doctor labels in `derivatives/hfo_annotation/{train,test}/<recording>.parquet`, 
assuming there are at least as many candidate rows as doctor-labeled rows for every detector. 
This script checks that assumption across the whole raw dataset and reports where it's violated (found for zurich's "ste" detector; see omni_ieeg.py's `_load_hfo_events` handling).

Usage:
    uv run --frozen --active python scripts/audit_hfo_detector_shortage.py \
        --raw-dir /capstor/scratch/cscs/davalos/data/raw/omni_ieeg
"""
import argparse
from pathlib import Path

import pandas as pd

DEFAULT_RAW_DIR = Path("/capstor/scratch/cscs/davalos/data/raw/omni_ieeg")


def find_recordings(raw_dir: Path):
    for csv_path in sorted((raw_dir / "derivatives" / "hfo").rglob("*.csv")):
        edf_name = csv_path.with_suffix(".edf").name
        for split in ("train", "test"):
            parquet_path = (
                raw_dir / "derivatives" / "hfo_annotation" / split / f"{edf_name}.parquet"
            )
            if parquet_path.exists():
                yield csv_path, parquet_path
                break


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    args = parser.parse_args()

    rows = []
    for candidates_path, annotation_path in find_recordings(args.raw_dir):
        candidates = pd.read_csv(candidates_path)
        doctor_labels = pd.read_parquet(annotation_path)
        cand_counts = candidates["detector"].value_counts().to_dict()
        doc_counts = doctor_labels["detector"].value_counts().to_dict()
        recording_id = candidates_path.stem
        site = recording_id.removeprefix("sub-").split("_")[0].rstrip("0123456789")
        for detector, need in doc_counts.items():
            have = cand_counts.get(detector, 0)
            rows.append({
                "recording_id": recording_id,
                "site": site,
                "detector": detector,
                "need": need,
                "have": have,
                "short": have < need,
            })

    df = pd.DataFrame(rows)
    short = df[df["short"]]

    print(f"{len(df)} (recording, detector) pairs checked across {df['recording_id'].nunique()} recordings")
    print(f"{len(short)} short pairs found\n")
    print("Shortages by detector:")
    print(short["detector"].value_counts())
    print("\nShortages by site:")
    print(short["site"].value_counts())
    print("\nFull shortage list:")
    with pd.option_context("display.max_rows", None):
        print(short.sort_values(["detector", "site", "recording_id"]).to_string(index=False))

    csv_out = Path(__file__).parent / "hfo_detector_shortage_report.csv"
    df.to_csv(csv_out, index=False)
    print(f"\nFull report written to {csv_out}")


if __name__ == "__main__":
    main()
