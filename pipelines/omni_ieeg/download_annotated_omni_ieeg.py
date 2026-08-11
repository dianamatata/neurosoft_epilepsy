#!/usr/bin/env python3
"""Download Omni-iEEG raw subjects that have event annotations but are missing on disk.

Reads `participants.tsv` from the raw Omni-iEEG dir, selects participants with
`event_annotation == 1`, skips those whose folder already exists, and downloads
the rest directly into that same raw dir via the Omni-iEEG HuggingFace downloader.

Run with the Omni-iEEG project's env (has huggingface_hub/tqdm):
    uv run --project Omni-iEEG python scripts/download_annotated_omni_ieeg.py
    uv run --active python3 /capstor/scratch/cscs/davalos/neurosoft_epilepsy/scripts/download_annotated_omni_ieeg.py --dry-run 2>&1 | tail -50
    uv run /Users/avalos/Documents/Programming/neurosoft_epilepsy/pipelines/omni_ieeg/download_annotated_omni_ieeg.py  --dry-run
"""
import argparse
import csv
import sys
from pathlib import Path
from omni_ieeg.dataloader.download_dataset import download_and_extract

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "Omni-iEEG"))


DEFAULT_RAW_DIR = Path("/capstor/scratch/cscs/davalos/data/raw/omni_ieeg")
DEFAULT_RAW_DIR = Path("/Users/avalos/Documents/Programming/neurosoft_epilepsy/data/raw/omni_ieeg")


def annotated_participants(raw_dir: Path) -> list[str]:
    with open(raw_dir / "participants.tsv", newline="") as f:
        rows = csv.DictReader(f, delimiter="\t")
        return [row["participant_id"] for row in rows if row["event_annotation"] == "1"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    annotated = annotated_participants(args.raw_dir)
    missing = [pid for pid in annotated if not (args.raw_dir / pid).exists()]
    print(f"{len(annotated)} participants have event annotations; {len(missing)} missing on disk.")

    failed = []
    for i, pid in enumerate(missing, 1):
        print(f"[{i}/{len(missing)}] {pid}")
        if args.dry_run:
            continue
        try:
            download_and_extract(
                repo_id="Omni-iEEG/Omni-iEEG",
                output_dir=str(args.raw_dir),
                delete_tars=True,
                folder_path=pid,
            )
        except Exception as e:
            print(f"  FAILED: {pid}: {e}")
            failed.append(pid)

    if failed:
        print(f"\n{len(failed)} subject(s) failed: {failed}")
        sys.exit(1)


if __name__ == "__main__":
    main()
