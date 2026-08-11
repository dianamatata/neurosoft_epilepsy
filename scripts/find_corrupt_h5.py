#!/usr/bin/env python3
"""Find corrupt/incomplete .h5 files in a brainsets processed directory.

A pipeline crash during `process()` happens after `h5py.File(path, "w")` has
already created the file and after some groups were already written -- the
file still gets closed cleanly (the `with` block's __exit__ runs even on
exception), so it's missing whatever the pipeline hadn't gotten around to
writing yet, but it still `.exists()`. That's exactly what let corrupt,
multi-GB partial files hide as "already processed" on a rerun.

Detects two kinds of broken files, with no pipeline-specific schema hardcoded:
  - Unreadable / truncated: h5py can't even open the file.
  - Incomplete: the file opens fine but is missing top-level groups that
    every other file in the directory has (found by majority vote across
    the directory, so this works for any BrainsetPipeline's output).

Usage:
    uv run --frozen --active python scripts/find_corrupt_h5.py /path/to/processed/omni_ieeg
    uv run --frozen --active python scripts/find_corrupt_h5.py /path/to/processed/omni_ieeg --delete
"""
import argparse
from collections import Counter
from pathlib import Path
from typing import Optional

import h5py


def top_level_keys(path: Path) -> Optional[frozenset]:
    try:
        with h5py.File(path, "r") as f:
            return frozenset(f.keys())
    except OSError:
        return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("processed_dir", type=Path)
    parser.add_argument(
        "--delete", action="store_true", help="Delete broken files after reporting them"
    )
    args = parser.parse_args()

    h5_files = sorted(args.processed_dir.glob("*.h5"))
    if not h5_files:
        print(f"No .h5 files found under {args.processed_dir}")
        return

    keys_by_file = {}
    unreadable = []
    for path in h5_files:
        keys = top_level_keys(path)
        if keys is None:
            unreadable.append(path)
        else:
            keys_by_file[path] = keys

    if not keys_by_file:
        print("No readable .h5 files to derive a reference schema from.")
        return

    # The most common complete key-set across all readable files is treated
    # as the reference schema for this brainset -- no hardcoded assumptions
    # about what fields a Data object should have.
    reference, ref_count = Counter(keys_by_file.values()).most_common(1)[0]
    print(
        f"Reference schema ({ref_count}/{len(keys_by_file)} readable files): "
        f"{sorted(reference)}\n"
    )

    incomplete = {
        path: keys for path, keys in keys_by_file.items() if keys != reference
    }

    print(f"{len(h5_files)} files checked")
    print(f"{len(unreadable)} unreadable/truncated")
    print(f"{len(incomplete)} incomplete (missing top-level keys)\n")

    for path in unreadable:
        print(f"UNREADABLE  {path.name}")
    for path, keys in incomplete.items():
        missing = reference - keys
        extra = keys - reference
        detail = f"missing={sorted(missing)}"
        if extra:
            detail += f" extra={sorted(extra)}"
        print(f"INCOMPLETE  {path.name}  {detail}")

    broken = unreadable + list(incomplete.keys())
    if args.delete and broken:
        for path in broken:
            path.unlink()
        print(f"\nDeleted {len(broken)} broken file(s).")
    elif broken:
        print(f"\n{len(broken)} broken file(s) found. Rerun with --delete to remove them.")
    else:
        print("\nNo broken files found.")


if __name__ == "__main__":
    main()
