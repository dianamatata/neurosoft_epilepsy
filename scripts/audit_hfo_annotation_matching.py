#!/usr/bin/env python3
"""Audit how well `derivatives/hfo_annotation` filenames match omni_ieeg recordings.

`_find_annotation_path` (foundry/data/pipelines/omni_ieeg.py) looks up a
recording's doctor-annotated HFO parquet by its raw EDF filename, with one
known fallback: some sites (found for UCLA) drop the "openieeg" prefix in the
derivatives filename. This script re-derives that match for every recording
that has `derivatives/hfo` candidates, so a future naming inconsistency shows
up here instead of silently returning "no annotations" for an already
recorded doctor label.

Three outcomes per recording:
  - matched (exact): raw EDF name found bytewise in `hfo_annotation/{train,test}`.
  - matched (fallback): only found after applying the known "openieeg" strip.
  - unmatched: no match by either method. Any leftover `hfo_annotation` file
    whose filename doesn't correspond one-to-one with a BIDS recording ID
    (seen for Multicenter's `Pt11_MO_bipolar_mne_10min.edf`-style names,
    which may even be a different channel montage/reference than what this
    pipeline processes) is reported separately as "unclaimed" for manual
    review -- this script does not attempt to auto-link those.

Usage:
    uv run --frozen --active python scripts/audit_hfo_annotation_matching.py \
        --raw-dir /capstor/scratch/cscs/davalos/data/raw/omni_ieeg
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from foundry.data.pipelines.omni_ieeg import _find_annotation_path  # noqa: E402

DEFAULT_RAW_DIR = Path("/capstor/scratch/cscs/davalos/data/raw/omni_ieeg")


def all_annotation_files(raw_dir: Path) -> set[Path]:
    files = set()
    for split in ("train", "test"):
        files.update((raw_dir / "derivatives" / "hfo_annotation" / split).glob("*.parquet"))
    return files


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    args = parser.parse_args()

    rows = []
    claimed = set()
    for csv_path in sorted((args.raw_dir / "derivatives" / "hfo").rglob("*.csv")):
        edf_name = csv_path.with_suffix(".edf").name
        recording_id = csv_path.stem
        site = recording_id.removeprefix("sub-").split("_")[0].rstrip("0123456789")

        exact_path = (
            args.raw_dir / "derivatives" / "hfo_annotation" / "train" / f"{edf_name}.parquet"
        )
        exact_path_test = (
            args.raw_dir / "derivatives" / "hfo_annotation" / "test" / f"{edf_name}.parquet"
        )
        matched_path = _find_annotation_path(args.raw_dir, edf_name)

        if matched_path is None:
            outcome = "unmatched"
        elif matched_path in (exact_path, exact_path_test):
            outcome = "matched (exact)"
        else:
            outcome = "matched (fallback)"

        if matched_path is not None:
            claimed.add(matched_path)

        rows.append({
            "recording_id": recording_id,
            "site": site,
            "outcome": outcome,
        })

    df = pd.DataFrame(rows)
    unclaimed = sorted(all_annotation_files(args.raw_dir) - claimed)

    print(f"{len(df)} recordings with HFO candidates checked\n")
    print(df["outcome"].value_counts().to_string())
    print()
    print("By site:")
    print(df.groupby(["site", "outcome"]).size().unstack(fill_value=0).to_string())

    print(f"\n{len(unclaimed)} annotation file(s) in derivatives/hfo_annotation "
          f"that no recording's candidates CSV claimed (needs manual review, "
          f"not auto-linked):")
    for path in unclaimed:
        print(f"  UNCLAIMED  {path.relative_to(args.raw_dir)}")

    csv_out = Path(__file__).parent / "hfo_annotation_matching_report.csv"
    df.to_csv(csv_out, index=False)
    print(f"\nFull report written to {csv_out}")


if __name__ == "__main__":
    main()
